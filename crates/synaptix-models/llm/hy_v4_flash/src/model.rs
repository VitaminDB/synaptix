//! Прямой проход Hy4-flash (CUDA). Потоки iHC `[T, 4, H]` — F32; проекции —
//! BF16 (квант-веса MXFP8/NVFP4), эксперты — F16 через общий [`MoeFfn`].
//!
//! Внимание MLA считается в поглощённой форме по латентному кэшу (норм. латент
//! 512 + k_rot 64 на токен): `q_lat = q_nope·W_uk`, оценки по 576 каналам,
//! выход `o_lat·W_uvᵀ`. Ключей на запрос не больше 2048: окно SWA 512, у
//! индексированных слоёв — до 512 групп по 4 токена.

use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::config::Activation;
use synaptix_llm_common::moe::{ExpertCache, ExpertSource, MoeConfig, MoeExt, MoeFfn};
use synaptix_llm_common::{ModelError, QLinear, WeightSource};

use crate::config::{AttnKind, Hy4Config, MlaDims};
use crate::engram::{table_rows, EngramHash, EngramLookup};
use crate::kern::{self, fe, f32c, Keys, R};
use crate::loader::Hy4Weights;

/// Проекции — в BF16 (диапазон шире F16; квант-ядра читают его напрямую).
const ACT: DType = DType::BF16;

fn lin(w: &dyn WeightSource, key: &str, dev: Device) -> R<QLinear> {
    if let Some(q) = w.quant(key, dev) {
        return Ok(QLinear::Quant(q?));
    }
    let t = w.tensor(key, dev, ACT)?;
    QLinear::build(t, ACT, ACT)
}

fn f32t(w: &dyn WeightSource, key: &str, dev: Device) -> R<Tensor> {
    let t = w.tensor(key, dev, DType::F32)?;
    kern::contig(&t)
}

/// Плотная матрица F32 (деквант, если в бандле квантована).
fn dense_f32(w: &dyn WeightSource, key: &str, dev: Device) -> R<Tensor> {
    if let Some(q) = w.quant(key, dev) {
        let q = q?;
        return q.dequantize(DType::BF16).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.contiguous()).map_err(fe);
    }
    f32t(w, key, dev)
}

fn fwd(l: &QLinear, x: &Tensor) -> R<Tensor> {
    l.forward(x)
}

/// iHC-смеситель подслоя: `fn [NM, M·H]`, `base [NM]`, `scale [2]` (у головы — `[1]`).
pub struct Hc {
    fn_w: Tensor,
    base: Tensor,
    scale: Tensor,
}

impl Hc {
    fn load(w: &dyn WeightSource, p: &str, dev: Device) -> R<Self> {
        Ok(Self { fn_w: f32t(w, &format!("{p}_fn"), dev)?, base: f32t(w, &format!("{p}_base"), dev)?, scale: f32t(w, &format!("{p}_scale"), dev)? })
    }

    fn mixes(&self, x: &Tensor) -> R<Tensor> {
        let (t, m, h) = (x.dims()[0], x.dims()[1], x.dims()[2]);
        let flat = x.reshape(vec![t, m * h]).map_err(fe)?;
        kern::contig(&flat.linear(&self.fn_w).map_err(fe)?)
    }
}

/// Сжатый индексатор слоя с разреженным вниманием.
pub struct Indexer {
    q_proj: QLinear,
    w_proj: QLinear,
    k_proj: QLinear,
    k_norm: Tensor,
    wv: Tensor,
    wg: Tensor,
    ape: Tensor,
    key_norm: Tensor,
}

pub struct Attn {
    dims: MlaDims,
    kind: AttnKind,
    q_a: QLinear,
    q_a_norm: Tensor,
    q_b: QLinear,
    kv_a: QLinear,
    kv_a_norm: Tensor,
    /// `[NH, qk_nope, L]` BF16: `q_lat = q_nope · W_uk`.
    w_uk: Tensor,
    /// `[NH, L, V]` BF16: `v = o_lat · W_uvᵀ`.
    w_uv_t: Tensor,
    gate: QLinear,
    o_proj: QLinear,
    sinks: Tensor,
    post_norm: Tensor,
    indexer: Option<Indexer>,
}

pub enum Mlp {
    Dense { gate: QLinear, up: QLinear, down: QLinear, post_norm: Tensor },
    Moe { moe: MoeFfn, pre: QLinear, post: QLinear, sh_gate: QLinear, sh_up: QLinear, sh_down: QLinear, combine_norm: Tensor },
}

pub struct EngramLayer {
    lookup: EngramLookup,
    /// Ключи потоков и value: `[(M+1)·H, width]` F32 и смещение `[(M+1)·H]`.
    proj_w: Tensor,
    proj_b: Tensor,
    key_norm: Tensor,
    query_norm: Tensor,
    pre_gamma: Tensor,
    hc_scale: Tensor,
}

pub struct Layer {
    attn_hc: Hc,
    ffn_hc: Hc,
    in_norm: Tensor,
    ffn_norm: Tensor,
    attn: Attn,
    mlp: Mlp,
    engram: Option<EngramLayer>,
}

/// KV одного слоя: латенты `[CAP, L+R]`; у индексированного — ещё ключи
/// токенов индексатора `[CAP, C]` и ключи групп `[CAP/G, C]`.
pub struct LayerCache {
    pub lat: Tensor,
    pub tok: Option<Tensor>,
    pub grp: Option<Tensor>,
    /// Сколько групп уже свёрнуто.
    pub pooled: usize,
}

pub struct ModelCache {
    pub layers: Vec<LayerCache>,
    /// Токены, уже прошедшие через модель (история для engram).
    pub ids: Vec<u32>,
    pub max_seq: usize,
}

impl ModelCache {
    pub fn seq_len(&self) -> usize {
        self.ids.len()
    }

    /// Откатить к первым `n` токенам (KV дальше просто перепишется).
    pub fn truncate(&mut self, n: usize) {
        if n < self.ids.len() {
            self.ids.truncate(n);
            for l in &mut self.layers {
                l.pooled = l.pooled.min(n / 4);
            }
        }
    }

    /// Перенести тензоры кэша на `dev`; возвращает прежнее устройство.
    pub fn move_to(&mut self, dev: Device) -> R<Device> {
        let prev = self.layers.first().map(|l| l.lat.device()).unwrap_or(dev);
        for l in &mut self.layers {
            l.lat = l.lat.to_device(dev).map_err(fe)?;
            if let Some(t) = &l.tok {
                l.tok = Some(t.to_device(dev).map_err(fe)?);
            }
            if let Some(t) = &l.grp {
                l.grp = Some(t.to_device(dev).map_err(fe)?);
            }
        }
        Ok(prev)
    }

    pub fn device_bytes(&self) -> usize {
        let b = |t: &Tensor| t.dtype().bytes_for_numel(t.numel());
        self.layers.iter().map(|l| b(&l.lat) + l.tok.as_ref().map_or(0, b) + l.grp.as_ref().map_or(0, b)).sum()
    }
}

pub struct Hy4Model {
    pub config: Hy4Config,
    pub device: Device,
    /// Таблица эмбеддингов в RAM (BF16-биты строк).
    embed: Vec<u16>,
    layers: Vec<Layer>,
    head_hc: Hc,
    norm: Tensor,
    lm_head: QLinear,
    expert_cache: Option<Arc<ExpertCache>>,
    /// Кэш латентов индексированных слоёв — F32 (BF16 — `HY4_KV=bf16`).
    kv_dtype: DType,
    /// Ёмкость кольца SWA сверх окна (чанк префилла).
    swa_extra: usize,
}

fn moe_config(cfg: &Hy4Config, bias: Vec<f32>, chunk: usize) -> MoeConfig {
    MoeConfig {
        hidden_size: cfg.moe_hidden,
        moe_intermediate_size: cfg.moe_intermediate,
        num_experts: cfg.n_experts,
        num_experts_per_tok: cfg.top_k,
        shared_intermediate_size: 0,
        norm_topk_prob: cfg.norm_topk,
        chunk,
        skip_below: 0.0,
        router_key: None,
        per_expert_scale: None,
        activation: if cfg.swiglu_limit > 0.0 { Activation::SiluClamp(cfg.swiglu_limit as u32) } else { Activation::Silu },
        ext: MoeExt { sigmoid_scoring: true, select_bias: Some(bias), routed_scale: cfg.routed_scale, shared_ungated: true, router_hidden: Some(cfg.hidden_size) },
    }
}

/// Строки BF16 таблицы эмбеддингов в RAM.
fn load_embed(w: &Hy4Weights, cfg: &Hy4Config) -> R<Vec<u16>> {
    let key = "model.embed_tokens.weight";
    let t = match w.quant(key, Device::Cpu) {
        Some(q) => q?.dequantize(DType::BF16).and_then(|t| t.to_dtype(DType::F32)).map_err(fe)?,
        None => w.tensor(key, Device::Cpu, DType::F32)?,
    };
    let v = t.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(fe)?;
    if v.len() != cfg.vocab_size * cfg.hidden_size {
        return Err(ModelError::Load(format!("{key}: {} элементов", v.len())));
    }
    Ok(v.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect())
}

impl Hy4Model {
    pub fn build(w: &Hy4Weights, device: Device, expert_cache: Option<Arc<ExpertCache>>, expert_source: Option<Arc<dyn ExpertSource>>, chunk: usize) -> R<Self> {
        let cfg = w.config.clone();
        let dev = device;
        let token_map: Arc<Vec<u32>> = match &cfg.engram {
            Some(e) => Arc::new(w.host_i32(&format!("model.layers.{}.engram_layer.lookup_table", e.layers[0]))?.into_iter().map(|x| x as u32).collect()),
            None => Arc::new(Vec::new()),
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let dims = *cfg.dims(l);
            let kind = cfg.attn[l];
            let a = format!("{p}.self_attn");
            // W_uk / W_uv из kv_b_proj [NH·(nope+v), L].
            let kvb = dense_f32(w, &format!("{a}.kv_b_proj.weight"), dev)?;
            let (nh, nope, vd, lat) = (dims.heads, dims.qk_nope, dims.v, cfg.kv_lora_rank);
            let kvb = kvb.reshape(vec![nh, nope + vd, lat]).map_err(fe)?;
            let w_uk = kern::to(&kvb.narrow(1, 0, nope).map_err(fe)?, ACT)?;
            let w_uv_t = kern::to(&kvb.narrow(1, nope, vd).and_then(|t| t.transpose(1, 2)).map_err(fe)?, ACT)?;
            drop(kvb);
            let indexer = if kind == AttnKind::Indexed {
                let i = format!("{a}.compressed_indexer");
                Some(Indexer {
                    q_proj: lin(w, &format!("{i}.query_projection.weight"), dev)?,
                    w_proj: lin(w, &format!("{i}.head_weight_projection.weight"), dev)?,
                    k_proj: lin(w, &format!("{i}.token_key_projection.weight"), dev)?,
                    k_norm: f32t(w, &format!("{i}.token_key_norm.weight"), dev)?,
                    wv: f32t(w, &format!("{i}.key_value_projection.weight"), dev)?,
                    wg: f32t(w, &format!("{i}.key_gate_projection.weight"), dev)?,
                    ape: f32t(w, &format!("{i}.intra_group_ape"), dev)?,
                    key_norm: f32t(w, &format!("{i}.key_norm.weight"), dev)?,
                })
            } else {
                None
            };
            let attn = Attn {
                dims,
                kind,
                q_a: lin(w, &format!("{a}.q_a_proj.weight"), dev)?,
                q_a_norm: f32t(w, &format!("{a}.q_a_layernorm.weight"), dev)?,
                q_b: lin(w, &format!("{a}.q_b_proj.weight"), dev)?,
                kv_a: lin(w, &format!("{a}.kv_a_proj_with_mqa.weight"), dev)?,
                kv_a_norm: f32t(w, &format!("{a}.kv_a_layernorm.weight"), dev)?,
                w_uk,
                w_uv_t,
                gate: lin(w, &format!("{a}.linear_gate.weight"), dev)?,
                o_proj: lin(w, &format!("{a}.o_proj.weight"), dev)?,
                sinks: f32t(w, &format!("{a}.learnable_sink_param"), dev)?,
                post_norm: f32t(w, &format!("{a}.post_proj_norm.weight"), dev)?,
                indexer,
            };
            let m = format!("{p}.mlp");
            let mlp = if cfg.dense[l] {
                Mlp::Dense {
                    gate: lin(w, &format!("{m}.gate_proj.weight"), dev)?,
                    up: lin(w, &format!("{m}.up_proj.weight"), dev)?,
                    down: lin(w, &format!("{m}.down_proj.weight"), dev)?,
                    post_norm: f32t(w, &format!("{m}.post_fc2_norm.weight"), dev)?,
                }
            } else {
                let bias = w.host_f32(&format!("{m}.gate.e_score_correction_bias"))?;
                let mc = moe_config(&cfg, bias, chunk);
                let moe = match (&expert_cache, &expert_source) {
                    (Some(c), Some(s)) => MoeFfn::load_lazy(w, &m, mc, dev, DType::F16, DType::NVFP4, c.clone(), l, s.clone())?,
                    (Some(c), None) => MoeFfn::load_offloaded(w, &m, mc, dev, DType::F16, DType::NVFP4, c.clone(), l)?,
                    (None, _) => MoeFfn::load(w, &m, mc, dev, DType::F16, DType::NVFP4)?,
                };
                Mlp::Moe {
                    moe,
                    pre: lin(w, &format!("{m}.routed_expert_down_proj.weight"), dev)?,
                    post: lin(w, &format!("{m}.routed_expert_up_proj.weight"), dev)?,
                    sh_gate: lin(w, &format!("{m}.shared_experts.gate_proj.weight"), dev)?,
                    sh_up: lin(w, &format!("{m}.shared_experts.up_proj.weight"), dev)?,
                    sh_down: lin(w, &format!("{m}.shared_experts.down_proj.weight"), dev)?,
                    combine_norm: f32t(w, &format!("{m}.combine_output_norm.weight"), dev)?,
                }
            };
            let engram = match cfg.engram.as_ref().and_then(|e| e.index_of(l).map(|i| (e, i))) {
                Some((e, i)) => {
                    let g = format!("{p}.engram_layer");
                    let mc = cfg.hc_mult;
                    let hash = EngramHash::new(e, i, token_map.clone(), w.engram_multipliers[i]);
                    let table = table_rows(&w.raw, &format!("{g}.multi_head_embedding.embedding.weight"), e.head_dim)?;
                    let mut head_norm = Vec::with_capacity(e.width());
                    for h in 0..e.heads() {
                        head_norm.extend(w.host_f32(&format!("{g}.lookup_per_head_norms.{h}.weight"))?);
                    }
                    let mut pw = Vec::new();
                    let mut pb = Vec::new();
                    for c in 0..mc {
                        pw.push(dense_f32(w, &format!("{g}.key_projs.{c}.weight"), dev)?);
                        pb.push(f32t(w, &format!("{g}.key_projs.{c}.bias"), dev)?);
                    }
                    pw.push(dense_f32(w, &format!("{g}.value_proj.weight"), dev)?);
                    pb.push(f32t(w, &format!("{g}.value_proj.bias"), dev)?);
                    let cat = |v: &[Tensor]| -> R<Tensor> { kern::contig(&Tensor::cat(&v.iter().collect::<Vec<_>>(), 0).map_err(fe)?) };
                    let stack = |name: &str| -> R<Tensor> {
                        let v: Vec<Tensor> = (0..mc).map(|c| f32t(w, &format!("{g}.{name}.{c}.weight"), dev)).collect::<R<_>>()?;
                        cat(&v)
                    };
                    let hcs: Vec<Tensor> = (0..mc).map(|c| f32t(w, &format!("{g}.unified_per_hc_scales.{c}"), dev)).collect::<R<_>>()?;
                    Some(EngramLayer {
                        lookup: EngramLookup { hash, table, head_norm },
                        proj_w: cat(&pw)?,
                        proj_b: cat(&pb)?,
                        key_norm: stack("norm_key")?,
                        query_norm: stack("norm_query")?,
                        pre_gamma: f32t(w, &format!("{g}.unified_pre_gamma"), dev)?,
                        hc_scale: cat(&hcs)?,
                    })
                }
                None => None,
            };
            layers.push(Layer {
                attn_hc: Hc::load(w, &format!("{p}.hc_attn_layer.hc_pre.hc"), dev)?,
                ffn_hc: Hc::load(w, &format!("{p}.hc_mlp_layer.hc_pre.hc"), dev)?,
                in_norm: f32t(w, &format!("{p}.input_layernorm.weight"), dev)?,
                ffn_norm: f32t(w, &format!("{p}.post_attention_layernorm.weight"), dev)?,
                attn,
                mlp,
                engram,
            });
            if std::env::var_os("HY4_VERBOSE").is_some() {
                eprintln!("[hy4] слой {l} поднят");
            }
        }
        let head_hc = Hc { fn_w: f32t(w, "model.hc_head.hc_head_fn", dev)?, base: f32t(w, "model.hc_head.hc_head_base", dev)?, scale: f32t(w, "model.hc_head.hc_head_scale", dev)? };
        let kv_dtype = if std::env::var("HY4_KV").as_deref() == Ok("bf16") { DType::BF16 } else { DType::F32 };
        Ok(Self {
            embed: load_embed(w, &cfg)?,
            norm: f32t(w, "model.norm.weight", dev)?,
            lm_head: lin(w, "lm_head.weight", dev)?,
            head_hc,
            layers,
            config: cfg,
            device,
            expert_cache,
            kv_dtype,
            swa_extra: chunk,
        })
    }

    pub fn expert_cache(&self) -> Option<&Arc<ExpertCache>> {
        self.expert_cache.as_ref()
    }

    pub fn make_cache(&self, max_seq: usize) -> R<ModelCache> {
        let c = &self.config;
        let d = c.kv_lora_rank + c.indexed.qk_rope;
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for l in 0..c.num_hidden_layers {
            let (cap, dt) = match c.attn[l] {
                AttnKind::Sliding => ((c.sliding_window + self.swa_extra).min(max_seq.max(1)), DType::F32),
                AttnKind::Indexed => (max_seq.max(1), self.kv_dtype),
            };
            let lat = kern::zeros(vec![cap, d], dt, self.device)?;
            let (tok, grp) = if c.attn[l] == AttnKind::Indexed {
                (
                    Some(kern::zeros(vec![max_seq.max(1), c.index_head_dim], DType::F32, self.device)?),
                    Some(kern::zeros(vec![max_seq.div_ceil(c.index_kpool).max(1), c.index_head_dim], DType::F32, self.device)?),
                )
            } else {
                (None, None)
            };
            layers.push(LayerCache { lat, tok, grp, pooled: 0 });
        }
        Ok(ModelCache { layers, ids: Vec::new(), max_seq })
    }

    /// Байт KV на токен (индексированные слои растут с длиной, SWA — кольцо).
    pub fn kv_bytes_per_token(&self) -> usize {
        let c = &self.config;
        let n = c.attn.iter().filter(|a| **a == AttnKind::Indexed).count();
        n * ((c.kv_lora_rank + c.indexed.qk_rope) * self.kv_dtype.bytes_for_numel(1) + c.index_head_dim * 4 + c.index_head_dim)
    }

    fn embed_rows(&self, ids: &[u32]) -> R<Tensor> {
        let h = self.config.hidden_size;
        let mut v = vec![0f32; ids.len() * h];
        for (i, id) in ids.iter().enumerate() {
            let r = (*id as usize).min(self.config.vocab_size - 1);
            for (o, b) in v[i * h..(i + 1) * h].iter_mut().zip(&self.embed[r * h..(r + 1) * h]) {
                *o = f32::from_bits((*b as u32) << 16);
            }
        }
        Tensor::from_vec(v, vec![ids.len(), h], self.device).map_err(fe)
    }

    /// Прогнать новые токены `ids` через модель, дописав кэш. Возвращает
    /// скрытое после финальной нормы `[T, H]` (BF16) — для `rows` последних
    /// строк (`None` — все).
    pub fn forward(&self, ids: &[u32], cache: &mut ModelCache, rows: Option<usize>) -> R<Tensor> {
        let c = &self.config;
        let (t, h, m) = (ids.len(), c.hidden_size, c.hc_mult);
        let p0 = cache.ids.len();
        if p0 + t > cache.max_seq {
            return Err(ModelError::Forward(format!("hy4: контекст {} > ёмкости {}", p0 + t, cache.max_seq)));
        }
        cache.ids.extend_from_slice(ids);
        let res = self.forward_inner(ids, p0, cache, rows);
        if res.is_err() {
            cache.truncate(p0);
        }
        let _ = (t, h, m);
        res
    }

    fn forward_inner(&self, ids: &[u32], p0: usize, cache: &mut ModelCache, rows: Option<usize>) -> R<Tensor> {
        let c = &self.config;
        let (t, h, m) = (ids.len(), c.hidden_size, c.hc_mult);
        let dev = self.device;
        let pos: Vec<i32> = (p0..p0 + t).map(|p| p as i32).collect();
        let pos = Tensor::from_vec(pos, vec![t], dev).map_err(fe)?;
        let emb = self.embed_rows(ids)?;
        let x = kern::contig(&emb.unsqueeze(1).and_then(|e| e.expand(vec![t, m, h])).map_err(fe)?)?;
        let all_ids = cache.ids.clone();
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(e) = &layer.engram {
                self.engram(e, &x, &all_ids, p0)?;
            }
            let (hin, post) = kern::hc_pre(&x, &layer.attn_hc.mixes(&x)?, &layer.attn_hc.base, &layer.attn_hc.scale, false, c.rms_norm_eps, c.hc_eps, c.hc_magnitude)?;
            let hn = kern::rms(&hin, Some(&layer.in_norm), c.rms_norm_eps, ACT)?;
            stage(l, "x", &x);
            stage(l, "attn_hn", &hn);
            let y = self.attention(l, &layer.attn, &hn, &pos, p0, &mut cache.layers[l])?;
            let y = kern::rms(&y, Some(&layer.attn.post_norm), c.rms_norm_eps, DType::F32)?;
            stage(l, "attn_y", &y);
            kern::hc_post(&x, post.as_ref().expect("post"), &y)?;
            let (hin, post) = kern::hc_pre(&x, &layer.ffn_hc.mixes(&x)?, &layer.ffn_hc.base, &layer.ffn_hc.scale, false, c.rms_norm_eps, c.hc_eps, c.hc_magnitude)?;
            let hn = kern::rms(&hin, Some(&layer.ffn_norm), c.rms_norm_eps, ACT)?;
            stage(l, "mlp_hn", &hn);
            let z = self.mlp(&layer.mlp, &hn)?;
            stage(l, "mlp_z", &z);
            kern::hc_post(&x, post.as_ref().expect("post"), &z)?;
            trace(l, &x);
        }
        let x = match rows {
            Some(r) if r < t => kern::contig(&x.narrow(0, t - r, r).map_err(fe)?)?,
            _ => x,
        };
        let (hh, _) = kern::hc_pre(&x, &self.head_hc.mixes(&x)?, &self.head_hc.base, &self.head_hc.scale, true, c.rms_norm_eps, c.hc_eps, 1.0)?;
        kern::rms(&hh, Some(&self.norm), c.rms_norm_eps, ACT)
    }

    pub fn logits(&self, hidden: &Tensor) -> R<Tensor> {
        f32c(&fwd(&self.lm_head, hidden)?)
    }

    fn engram(&self, e: &EngramLayer, x: &Tensor, all: &[u32], p0: usize) -> R<()> {
        let (emb, live) = e.lookup.embed(all, p0)?;
        let t = live.len();
        if !live.iter().any(|v| *v) {
            return Ok(());
        }
        let width = emb.len() / t.max(1);
        let et = Tensor::from_vec(emb, vec![t, width], self.device).map_err(fe)?;
        let proj = et.linear(&e.proj_w).and_then(|p| p.broadcast_add(&e.proj_b)).map_err(fe)?;
        let proj = kern::contig(&proj)?;
        let live: Vec<u8> = live.iter().map(|v| *v as u8).collect();
        let live = Tensor::from_vec(live, vec![t], self.device).map_err(fe)?;
        kern::engram_gate(x, &proj, &e.key_norm, &e.query_norm, &e.pre_gamma, &e.hc_scale, &live, f32::EPSILON)
    }

    fn mlp(&self, mlp: &Mlp, hn: &Tensor) -> R<Tensor> {
        let eps = self.config.rms_norm_eps;
        match mlp {
            Mlp::Dense { gate, up, down, post_norm } => {
                let g = fwd(gate, hn)?;
                let u = fwd(up, hn)?;
                let a = g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|g| g.mul(&u))).map_err(fe)?;
                let z = fwd(down, &a)?;
                kern::rms(&z, Some(post_norm), eps, DType::F32)
            }
            Mlp::Moe { moe, pre, post, sh_gate, sh_up, sh_down, combine_norm } => {
                let lat = kern::to(&fwd(pre, hn)?, DType::F16)?;
                let routed = moe.forward_routed(&lat, Some(hn))?;
                let routed = fwd(post, &kern::to(&routed, ACT)?)?;
                let g = fwd(sh_gate, hn)?;
                let u = fwd(sh_up, hn)?;
                let a = g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|g| g.mul(&u))).map_err(fe)?;
                let s = fwd(sh_down, &a)?;
                let sum = f32c(&routed)?.add(&f32c(&s)?).map_err(fe)?;
                kern::rms(&sum, Some(combine_norm), eps, DType::F32)
            }
        }
    }

    fn attention(&self, _l: usize, a: &Attn, hn: &Tensor, pos: &Tensor, p0: usize, lc: &mut LayerCache) -> R<Tensor> {
        let c = &self.config;
        let d = &a.dims;
        let (t, nh, lat) = (hn.dims()[0], d.heads, c.kv_lora_rank);
        let eps = c.rms_norm_eps;
        let gate = fwd(&a.gate, hn)?;
        let qr = kern::rms(&fwd(&a.q_a, hn)?, Some(&a.q_a_norm), eps, ACT)?;
        let q = f32c(&fwd(&a.q_b, &qr)?)?;
        kern::rope(&q, pos, nh * d.qk(), nh, d.qk(), d.qk_nope, d.qk_rope, d.rope_theta)?;
        let ckv = f32c(&fwd(&a.kv_a, hn)?)?;
        let latent = kern::rms_cols(&ckv, lat, Some(&a.kv_a_norm), eps, DType::F32)?;
        let krot = kern::contig(&ckv.narrow(1, lat, d.qk_rope).map_err(fe)?)?;
        kern::rope(&krot, pos, d.qk_rope, 1, d.qk_rope, 0, d.qk_rope, d.rope_theta)?;
        let row = kern::contig(&Tensor::cat(&[&latent, &krot], 1).map_err(fe)?)?;
        kern::cache_put(&row, &lc.lat, pos)?;
        // Поглощённый запрос: q_lat = q_nope · W_uk по головам.
        let q3 = q.reshape(vec![t, nh, d.qk()]).map_err(fe)?;
        let q_nope = kern::to(&q3.narrow(2, 0, d.qk_nope).and_then(|x| x.transpose(0, 1)).map_err(fe)?, ACT)?;
        let q_lat = f32c(&q_nope.matmul(&a.w_uk).map_err(fe)?)?; // [NH, T, L]
        let q_lat = kern::contig(&q_lat.transpose(0, 1).map_err(fe)?)?; // [T, NH, L]
        let q_rot = kern::contig(&q3.narrow(2, d.qk_nope, d.qk_rope).map_err(fe)?)?;
        let qa = kern::contig(&Tensor::cat(&[&q_lat, &q_rot], 2).map_err(fe)?)?;
        let scale = 1.0 / (d.qk() as f32).sqrt();
        let o_lat = match a.kind {
            AttnKind::Sliding => kern::mla_attn(&qa, &lc.lat, &a.sinks, pos, lat, Keys::Window(c.sliding_window), scale)?,
            AttnKind::Indexed => {
                let idx = a.indexer.as_ref().expect("индексатор");
                let groups = self.indexer(_l, idx, hn, &qr, pos, p0, t, lc)?;
                match &groups {
                    Some(g) => kern::mla_attn(&qa, &lc.lat, &a.sinks, pos, lat, Keys::Groups(g, c.index_kpool), scale)?,
                    None => kern::mla_attn(&qa, &lc.lat, &a.sinks, pos, lat, Keys::Window(usize::MAX), scale)?,
                }
            }
        };
        // v = o_lat · W_uvᵀ по головам.
        let o = kern::to(&o_lat.transpose(0, 1).map_err(fe)?, ACT)?; // [NH, T, L]
        let v = f32c(&o.matmul(&a.w_uv_t).map_err(fe)?)?; // [NH, T, V]
        let v = kern::contig(&v.transpose(0, 1).map_err(fe)?)?.reshape(vec![t, nh * d.v]).map_err(fe)?;
        stage(_l, "attn_o", &v);
        let og = kern::gate_mul(&v, &gate, ACT)?;
        fwd(&a.o_proj, &og)
    }

    /// Ключи индексатора в кэш, свёртка закрытых групп и, если хоть одному
    /// запросу мало плотного окна, выбор групп `[T, k]`.
    #[allow(clippy::too_many_arguments)]
    fn indexer(&self, l: usize, idx: &Indexer, hn: &Tensor, qr: &Tensor, pos: &Tensor, p0: usize, t: usize, lc: &mut LayerCache) -> R<Option<Tensor>> {
        let c = &self.config;
        let (g, ch) = (c.index_kpool, c.index_head_dim);
        let rope_w = c.indexed.qk_rope;
        let theta = c.indexed.rope_theta;
        let tok = kern::rms(&fwd(&idx.k_proj, hn)?, Some(&idx.k_norm), c.rms_norm_eps, DType::F32)?;
        let tok_cache = lc.tok.as_ref().expect("ключи токенов");
        let grp_cache = lc.grp.as_ref().expect("ключи групп");
        kern::cache_put(&tok, tok_cache, pos)?;
        let complete = (p0 + t) / g;
        kern::idx_pool(tok_cache, &idx.wv, &idx.wg, &idx.ape, &idx.key_norm, grp_cache, lc.pooled, complete, g, rope_w, c.rms_norm_eps, theta)?;
        lc.pooled = complete;
        let last_own = (p0 + t - 1) / g;
        if last_own <= c.index_groups() {
            return Ok(None);
        }
        let q = f32c(&fwd(&idx.q_proj, qr)?)?;
        kern::rope(&q, pos, c.index_heads * ch, c.index_heads, ch, ch - rope_w, rope_w, theta)?;
        let q = q.reshape(vec![t, c.index_heads, ch]).map_err(fe)?;
        let wscale = 1.0 / ((c.index_heads as f32).sqrt() * (ch as f32).sqrt());
        let w = f32c(&fwd(&idx.w_proj, hn)?)?.mul_scalar(wscale).map_err(fe)?;
        let w = kern::contig(&w)?;
        let ng = complete.max(1);
        let scores = kern::idx_score(&q, &w, grp_cache, pos, ng, g)?;
        let valid: Vec<u32> = (p0..p0 + t).map(|p| ((p / g) as u32).min(ng as u32)).collect();
        let valid = Tensor::from_vec(valid, vec![t], self.device).map_err(fe)?;
        let k = c.index_groups().min(ng);
        let sel = scores.topk_wide(&valid, k).map_err(fe)?;
        if std::env::var_os("HY4_DUMP").is_some() {
            stage(l, "idx_scores", &scores);
            if let Ok(v) = sel.to_vec2::<u32>() {
                let f: Vec<f32> = v.into_iter().flatten().map(|x| x as f32).collect();
                if let Ok(t) = Tensor::from_vec(f, vec![t, k], Device::Cpu) {
                    stage(l, "idx_sel", &t);
                }
            }
        }
        Ok(Some(sel))
    }
}

/// `HY4_DUMP=каталог`: промежуточные стадии слоя `HY4_STAGE_LAYER` (0) — `l0_<имя>.f32`.
fn stage(l: usize, name: &str, x: &Tensor) {
    let want = std::env::var("HY4_STAGE_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(0usize);
    if l != want {
        return;
    }
    let Ok(dir) = std::env::var("HY4_DUMP") else { return };
    if let Ok(v) = x.to_dtype(DType::F32).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()) {
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/l0_{name}.f32"), bytes);
    }
}

/// `HY4_DUMP=каталог`: потоки после каждого слоя в `.f32` (сверка с эталоном).
fn trace(l: usize, x: &Tensor) {
    let Ok(dir) = std::env::var("HY4_DUMP") else { return };
    if let Ok(v) = x.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/layer{l:02}.f32"), bytes);
    }
}
