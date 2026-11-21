//! Прямой проход Inkling-2 (схема — llama.cpp `inkling2.cpp` + `inkling2-loop.cpp`).
//!
//! Поток между слоями и блоки attention-residuals — F32; проекции — BF16-веса с
//! BF16-активацией (как в эталоне); KV — BF16; эксперты — квант-веса движка с
//! плотной BF16-активацией (только веса квантованы, как у modelopt-эталона).

use synaptix_core::backend::Ink2Launch;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::{ExpertTable, QuantWeight};
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::ModelError;
use synaptix_nn::linear::Linear;
use synaptix_nn::Module;

use crate::config::Inkling2Config;
use crate::kern::{self, addr, bf16c, empty, f32c, fe, launch, R};
use crate::loader::{ExpertStack, Inkling2Weights, LoadError, LLM};

/// Проекция: BF16-вес `[out, in]` либо квант-вес бандла с общим множителем.
/// Наибольший чанк одного прогона: кольцо KV SWA-слоя — окно + столько.
pub const MAX_CHUNK: usize = 512;

pub enum Lin {
    Dense(Linear),
    Quant(synaptix_llm_common::QLinear, f32),
}

impl Lin {
    fn load(w: &Inkling2Weights, name: &str, dev: Device) -> Result<Self, LoadError> {
        if let Some((q, alpha)) = w.quant(name, dev)? {
            return Ok(Self::Quant(synaptix_llm_common::QLinear::Quant(q), alpha));
        }
        let t = w.tensor(name, dev, DType::BF16)?;
        Ok(Self::Dense(Linear::new(t, None).map_err(|e| LoadError::Weights(format!("{name}: {e}")))?))
    }

    /// Плотные BF16-матрицы с общим входом, склеенные по выходам — один GEMV вместо нескольких.
    fn cat(w: &Inkling2Weights, names: &[String], dev: Device) -> Result<Self, LoadError> {
        let ts: Vec<Tensor> = names.iter().map(|n| w.tensor(n, dev, DType::BF16)).collect::<Result<_, _>>()?;
        let refs: Vec<&Tensor> = ts.iter().collect();
        let t = Tensor::cat(&refs, 0).map_err(lerr)?.contiguous().map_err(lerr)?;
        Ok(Self::Dense(Linear::new(t, None).map_err(lerr)?))
    }

    /// `x [T, in]` (любой dtype) → `[T, out]` в dtype движка (BF16).
    fn raw(&self, x: &Tensor) -> R<Tensor> {
        let x = bf16c(x)?;
        match self {
            Self::Dense(l) => l.forward(&x).map_err(fe),
            Self::Quant(q, a) => {
                let y = q.forward(&x)?;
                if *a == 1.0 { Ok(y) } else { y.mul_scalar(*a).map_err(fe) }
            }
        }
    }

    /// `x [T, in]` → `[T, out]` F32.
    fn f32(&self, x: &Tensor) -> R<Tensor> {
        f32c(&self.raw(x)?)
    }
}

/// Контроллер входа повторного прохода: `(1 − tanh(F·d))·x + tanh(G·d)·m`,
/// `d = Down·[rms(x); rms(m)]`. Веса F32: down `[R, 2H]`, gain/forget транспонированы в `[R, H]`.
struct Ctrl {
    down: Tensor,
    gain: Tensor,
    forget: Tensor,
    rank: usize,
}

struct LoopLayer {
    /// Запросы сообщений `[4, H]`: q, k, v, mlp.
    msg_q: Tensor,
    ctrl: [Ctrl; 4],
}

struct Experts {
    /// Таблицы адресов весов экспертов на карте (плотная активация).
    gate_up: ExpertTable,
    down: ExpertTable,
    alpha_gate_up: Vec<f32>,
    /// Множители на карте `[E]`: gate_up — для активации, down — в вес маршрута.
    alpha_gate_up_dev: Tensor,
    alpha_down_dev: Tensor,
    /// Держит веса живыми (таблицы — только адреса).
    _weights: (Vec<QuantWeight>, Vec<QuantWeight>),
}

pub struct Layer {
    swa: bool,
    attn_norm: Option<Tensor>,
    qkv_norm: Option<[Tensor; 3]>,
    /// Слитые проекции с общим входом: `pa` = q | r | gate | sink (вход q),
    /// `pk` = k | отводы k-свёртки, `pv` = v | отводы v-свёртки.
    pa: Lin,
    pk: Lin,
    pv: Lin,
    wo: Lin,

    k_conv: Tensor,
    v_conv: Tensor,
    dyn_conv: bool,
    q_clip: Tensor,
    r_clip: Tensor,
    rel_proj: Tensor,
    extent: usize,
    attn_res_q: Option<Tensor>,
    ffn_res_q: Tensor,
    attn_post_conv: Tensor,
    ffn_post_conv: Tensor,
    ffn_norm: Tensor,
    router: Tensor,
    router_bias: Tensor,
    experts: Experts,
    lp: Option<LoopLayer>,
}

/// Кэш одного слота (физический слой на своём проходе цикла).
pub struct SlotCache {
    /// Кольцо KV SWA-слоя (окно + запас на чанк префилла); 0 — линейный кэш на весь контекст.
    ring: usize,
    k: Tensor,
    v: Tensor,
    k_state: Tensor,
    v_state: Tensor,
    attn_state: Tensor,
    ffn_state: Tensor,
}

/// CUDA-граф шага декода. Сырые указатели драйвера не `Send`; граф запускается
/// только через `&mut ModelCache` — по одному потоку за раз.
pub struct GraphHandle(std::sync::Arc<cudarc::driver::CudaGraph>);
// SAFETY: доступ исключительный (через `&mut` кэша), драйвер допускает вызовы из разных потоков по очереди.
unsafe impl Send for GraphHandle {}
unsafe impl Sync for GraphHandle {}

pub struct ModelCache {
    pub slots: Vec<SlotCache>,
    pub seq_len: usize,
    pub capacity: usize,
    /// Позиция начала прогона (U32 `[1]`): ядра читают её с карты, поэтому шаг
    /// декода от позиции не зависит и записывается в граф один раз.
    pos_dev: Tensor,
    /// Токен шага декода (U32 `[1]`) и логиты шага (F32 `[1, V]`) — постоянные буферы графа.
    tok_dev: Tensor,
    out_dev: Option<Tensor>,
    graph: Option<GraphHandle>,
    warm: u32,
}

impl ModelCache {
    /// Граф держит адреса KV и состояний: после их замены он негоден.
    fn drop_graph(&mut self) {
        self.graph = None;
        self.warm = 0;
    }

    pub fn reset(&mut self) -> R<()> {
        self.drop_graph();
        self.seq_len = 0;
        for s in self.slots.iter_mut() {
            for t in [&mut s.k_state, &mut s.v_state, &mut s.attn_state, &mut s.ffn_state] {
                *t = Tensor::zeros(t.dims().to_vec(), DType::F32, t.device()).map_err(fe)?;
            }
        }
        Ok(())
    }
}

/// Готовые эмбеддинги (F32 `[len, H]` на карте) вместо строк чанка с `row`.
pub struct EmbedPatch {
    pub row: usize,
    pub embeds: Tensor,
}

pub struct Inkling2Model {
    pub cfg: Inkling2Config,
    pub device: Device,
    embed: Tensor,
    embed_norm: Tensor,
    unembed: Lin,
    norm: Tensor,
    output_q: Tensor,
    layers: Vec<Layer>,
}

fn lerr(e: impl std::fmt::Display) -> LoadError {
    LoadError::Weights(e.to_string())
}

/// Тензор F32 на карте из host-вектора.
fn dev_f32(v: Vec<f32>, shape: Vec<usize>, dev: Device) -> Result<Tensor, LoadError> {
    Tensor::from_vec(v, shape, dev).map_err(lerr)
}

/// Матрица `[r, c]` F32 → транспонированная `[c, r]` на карте.
fn dev_t(v: &[f32], r: usize, c: usize, dev: Device) -> Result<Tensor, LoadError> {
    let mut t = vec![0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = v[i * c + j];
        }
    }
    dev_f32(t, vec![c, r], dev)
}

/// Веса свёртки → `[C, K]`, последний отвод — текущий токен.
/// `conv1d.weight [C, 1, K]` уже так; `static_w [K, C]` — отвод по расстоянию назад.
fn conv_w(w: &Inkling2Weights, name: &str, k: usize, dev: Device) -> Result<Tensor, LoadError> {
    let v = w.host_f32(name)?;
    if name.ends_with("static_w") {
        let c = v.len() / k;
        let mut out = vec![0f32; c * k];
        for lag in 0..k {
            for ch in 0..c {
                out[ch * k + (k - 1 - lag)] = v[lag * c + ch];
            }
        }
        dev_f32(out, vec![c, k], dev)
    } else {
        let c = v.len() / k;
        dev_f32(v, vec![c, k], dev)
    }
}

impl Inkling2Model {
    pub fn load(w: &Inkling2Weights, dev: Device) -> Result<Self, LoadError> {
        let cfg = w.config.clone();
        let t = &cfg.text;
        let h = t.hidden_size;
        let k = t.sconv_kernel_size;
        let f32t = |name: &str| -> Result<Tensor, LoadError> { w.tensor(name, dev, DType::F32) };
        let embed = w.tensor(&format!("{LLM}.embed.weight"), dev, DType::BF16)?;
        let embed_norm = f32t(&format!("{LLM}.embed_norm.weight"))?;
        let unembed = Lin::load(w, &format!("{LLM}.unembed.weight"), dev)?;
        let norm = f32t(&format!("{LLM}.norm.weight"))?;
        let output_q = f32t(&format!("{LLM}.attn_res.output_query"))?.reshape(vec![1, h]).map_err(lerr)?;
        let (cs, ce) = cfg.core();
        let msg = if t.loop_config.is_some() {
            Some((
                w.host_f32(&format!("{LLM}.middle_cycle_cross_pass_queries"))?,
                w.host_f32(&format!("{LLM}.middle_cycle_cross_pass_kv_queries"))?,
            ))
        } else {
            None
        };
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for i in 0..t.num_hidden_layers {
            let p = format!("{LLM}.layers.{i}");
            let a = format!("{p}.attn");
            let swa = cfg.is_swa(i);
            let extent = cfg.rel_extent(i);
            let (attn_norm, qkv_norm) = if swa {
                (Some(f32t(&format!("{p}.attn_norm.weight"))?), None)
            } else {
                (
                    None,
                    Some([
                        f32t(&format!("{p}.q_input_norm.weight"))?,
                        f32t(&format!("{p}.k_input_norm.weight"))?,
                        f32t(&format!("{p}.v_input_norm.weight"))?,
                    ]),
                )
            };
            let (k_conv, v_conv, k_conv_proj, v_conv_proj) = if swa {
                (conv_w(w, &format!("{a}.k_sconv.weight"), k, dev)?, conv_w(w, &format!("{a}.v_sconv.weight"), k, dev)?, None, None)
            } else {
                (
                    conv_w(w, &format!("{a}.k_sconv.static_w"), k, dev)?,
                    conv_w(w, &format!("{a}.v_sconv.static_w"), k, dev)?,
                    Some(format!("{a}.k_sconv.weight_proj")),
                    Some(format!("{a}.v_sconv.weight_proj")),
                )
            };
            let attn_res_q = if i > 0 {
                let q = w.host_f32(&format!("{LLM}.attn_res.attn_queries.{}", i - 1))?;
                let nq = q.len() / h;
                Some(dev_f32(q, vec![nq, h], dev)?)
            } else {
                None
            };
            let ffn_res_q = dev_f32(w.host_f32(&format!("{LLM}.attn_res.mlp_queries.{i}"))?, vec![1, h], dev)?;
            let router = w.host_f32(&format!("{p}.mlp.gate.weight"))?;
            let e = t.n_routed_experts;
            let router = dev_f32(router, vec![e, h], dev)?;
            let router_bias = dev_f32(w.host_f32(&format!("{p}.mlp.gate.bias"))?, vec![e], dev)?;
            let ExpertStack { gate_up, down, alpha_gate_up, alpha_down } = w.experts(i, dev)?;
            let lp = if let (Some((q, kv)), true) = (&msg, cfg.loop_layer(i)) {
                let j = i - cs;
                let mut mq = Vec::with_capacity(4 * h);
                for row in [&q[2 * j * h..(2 * j + 1) * h], &kv[2 * j * h..(2 * j + 1) * h], &kv[(2 * j + 1) * h..(2 * j + 2) * h], &q[(2 * j + 1) * h..(2 * j + 2) * h]] {
                    mq.extend_from_slice(row);
                }
                let rank = t.loop_config.as_ref().map(|l| l.gate_rank).unwrap_or(64);
                let ctrl = |name: &str| -> Result<Ctrl, LoadError> {
                    let c = format!("{p}.controllers.{name}");
                    Ok(Ctrl {
                        down: dev_f32(w.host_f32(&format!("{c}.down_weight"))?, vec![rank, 2 * h], dev)?,
                        gain: dev_t(&w.host_f32(&format!("{c}.gain_weight"))?, h, rank, dev)?,
                        forget: dev_t(&w.host_f32(&format!("{c}.forget_weight"))?, h, rank, dev)?,
                        rank,
                    })
                };
                Some(LoopLayer { msg_q: dev_f32(mq, vec![4, h], dev)?, ctrl: [ctrl("attn")?, ctrl("k")?, ctrl("v")?, ctrl("mlp")?] })
            } else {
                None
            };
            let _ = ce;
            layers.push(Layer {
                swa,
                attn_norm,
                qkv_norm,
                pa: Lin::cat(w, &[format!("{a}.wq_du.weight"), format!("{a}.wr_du.weight"), format!("{a}.gate_attn.weight"), format!("{a}.attn_sink_proj.weight")], dev)?,
                pk: Lin::cat(w, &[&[format!("{a}.wk_dv.weight")][..], k_conv_proj.as_slice()].concat(), dev)?,
                pv: Lin::cat(w, &[&[format!("{a}.wv_dv.weight")][..], v_conv_proj.as_slice()].concat(), dev)?,
                wo: Lin::load(w, &format!("{a}.wo_ud.weight"), dev)?,

                k_conv,
                v_conv,
                dyn_conv: !swa,
                q_clip: f32t(&format!("{a}.q_l2_clip.scale"))?,
                r_clip: f32t(&format!("{a}.rel_l2_clip.scale"))?,
                rel_proj: f32t(&format!("{a}.rel_logits_proj.proj"))?,
                extent,
                attn_res_q,
                ffn_res_q,
                attn_post_conv: conv_w(w, &format!("{p}.attn_sconv.weight"), k, dev)?,
                ffn_post_conv: conv_w(w, &format!("{p}.mlp_sconv.weight"), k, dev)?,
                ffn_norm: f32t(&format!("{p}.mlp_norm.weight"))?,
                router,
                router_bias,
                experts: {
                    let gu_refs: Vec<&QuantWeight> = gate_up.iter().collect();
                    let dn_refs: Vec<&QuantWeight> = down.iter().collect();
                    let gt = ExpertTable::build_dense(&gu_refs).ok_or_else(|| LoadError::Weights(format!("слой {i}: таблица gate_up")))?;
                    let dt = ExpertTable::build_dense(&dn_refs).ok_or_else(|| LoadError::Weights(format!("слой {i}: таблица down")))?;
                    Experts {
                        gate_up: gt,
                        down: dt,
                        alpha_gate_up_dev: dev_f32(alpha_gate_up.clone(), vec![e], dev)?,
                        alpha_down_dev: dev_f32(alpha_down, vec![e], dev)?,
                        alpha_gate_up,
                        _weights: (gate_up, down),
                    }
                },
                lp,
            });
        }
        Ok(Self { cfg, device: dev, embed, embed_norm, unembed, norm, output_q, layers })
    }

    /// Физический слой слота кэша.
    fn slot_phys(&self, slot: usize) -> usize {
        let n = self.cfg.text.num_hidden_layers;
        if slot < n {
            return slot;
        }
        let (s, e) = self.cfg.core();
        s + (slot - n) % (e - s)
    }

    /// Кольцо KV слота: у SWA-слоёв — окно + наибольший чанк префилла, если это меньше контекста.
    fn slot_ring(&self, slot: usize, capacity: usize) -> usize {
        let ring = self.cfg.text.sliding_window_size + MAX_CHUNK;
        if self.cfg.is_swa(self.slot_phys(slot)) && ring < capacity { ring } else { 0 }
    }

    pub fn make_cache(&self, capacity: usize) -> R<ModelCache> {
        let t = &self.cfg.text;
        let (nkv, d, h, k) = (t.num_key_value_heads, t.head_dim, t.hidden_size, t.sconv_kernel_size);
        let dev = self.device;
        let z = |shape: Vec<usize>, dt: DType| Tensor::zeros(shape, dt, dev).map_err(fe);
        let mut slots = Vec::with_capacity(self.cfg.num_slots());
        for si in 0..self.cfg.num_slots() {
            let ring = self.slot_ring(si, capacity);
            let cap = if ring > 0 { ring } else { capacity };
            slots.push(SlotCache {
                ring,
                k: z(vec![nkv, cap, d], DType::BF16)?,
                v: z(vec![nkv, cap, d], DType::BF16)?,
                k_state: z(vec![k - 1, nkv * d], DType::F32)?,
                v_state: z(vec![k - 1, nkv * d], DType::F32)?,
                attn_state: z(vec![k - 1, h], DType::F32)?,
                ffn_state: z(vec![k - 1, h], DType::F32)?,
            });
        }
        Ok(ModelCache {
            slots,
            seq_len: 0,
            capacity,
            pos_dev: z(vec![1], DType::U32)?,
            tok_dev: z(vec![1], DType::U32)?,
            out_dev: None,
            graph: None,
            warm: 0,
        })
    }

    /// Расширить кэш до `capacity` позиций: KV копируется, состояния свёрток те же.
    pub fn grow_cache(&self, cache: &mut ModelCache, capacity: usize) -> R<()> {
        if capacity <= cache.capacity {
            return Ok(());
        }
        cache.drop_graph();
        let used = cache.seq_len;
        for (si, s) in cache.slots.iter_mut().enumerate() {
            if s.ring > 0 {
                continue;
            }
            let ring = self.slot_ring(si, capacity);
            if ring > 0 {
                // Слот, бывший линейным на малом контексте, становится кольцом: переложить хвост окна.
                let (nkv, d) = (s.k.dims()[0], s.k.dims()[2]);
                let lo = used.saturating_sub(ring);
                for kv in [&mut s.k, &mut s.v] {
                    let host: Vec<half::bf16> = kv.to_device(Device::Cpu).map_err(fe)?.flatten_all().map_err(fe)?.to_vec1::<half::bf16>().map_err(fe)?;
                    let cap_old = kv.dims()[1];
                    let mut buf = vec![half::bf16::ZERO; nkv * ring * d];
                    for h in 0..nkv {
                        for p in lo..used {
                            let src = (h * cap_old + p) * d;
                            let dst = (h * ring + p % ring) * d;
                            buf[dst..dst + d].copy_from_slice(&host[src..src + d]);
                        }
                    }
                    *kv = Tensor::from_vec(buf, vec![nkv, ring, d], self.device).map_err(fe)?;
                }
                s.ring = ring;
                continue;
            }
            for kv in [&mut s.k, &mut s.v] {
                let (nkv, d) = (kv.dims()[0], kv.dims()[2]);
                if used > 0 {
                    let head = kv.narrow(1, 0, used).map_err(fe)?;
                    let pad = Tensor::zeros(vec![nkv, capacity - used, d], DType::BF16, self.device).map_err(fe)?;
                    *kv = Tensor::cat(&[&head, &pad], 1).map_err(fe)?.contiguous().map_err(fe)?;
                } else {
                    *kv = Tensor::zeros(vec![nkv, capacity, d], DType::BF16, self.device).map_err(fe)?;
                }
            }
        }
        cache.capacity = capacity;
        Ok(())
    }

    /// Прогон токенов `ids` с позиции `cache.seq_len`. Возвращает логиты
    /// `[1, V]` последнего токена (или всех `[T, V]` при `all_logits`), F32,
    /// словарь без паддинга.
    pub fn forward(&self, ids: &[u32], cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        self.forward_with_embeds(ids, &[], cache, all_logits)
    }

    /// Как [`Self::forward`], но строки `patches` заменяются готовыми
    /// эмбеддингами (признаки картинки) — после `embed_norm`, без нормировки.
    /// `patches[i].row` — строка относительно всего `ids`; длинный прогон
    /// режется на куски по [`MAX_CHUNK`], патчи — по кускам.
    pub fn forward_with_embeds(&self, ids: &[u32], patches: &[EmbedPatch], cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        if ids.len() <= MAX_CHUNK {
            return self.forward_chunk(ids, patches, cache, all_logits);
        }
        let mut outs = Vec::new();
        let mut off = 0;
        while off < ids.len() {
            let end = (off + MAX_CHUNK).min(ids.len());
            let mut part = Vec::new();
            for p in patches {
                let len = p.embeds.dims()[0];
                let (a, b) = (p.row.max(off), (p.row + len).min(end));
                if a < b {
                    part.push(EmbedPatch { row: a - off, embeds: p.embeds.narrow(0, a - p.row, b - a).map_err(fe)? });
                }
            }
            outs.push(self.forward_chunk(&ids[off..end], &part, cache, all_logits)?);
            off = end;
        }
        if !all_logits {
            return Ok(outs.pop().expect("непустой прогон"));
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        Tensor::cat(&refs, 0).map_err(fe)
    }

    fn forward_chunk(&self, ids: &[u32], patches: &[EmbedPatch], cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        let n = ids.len();
        let past = cache.seq_len;
        if past + n > cache.capacity {
            return Err(ModelError::Forward(format!("inkling2: кэш {} < {}", cache.capacity, past + n)));
        }
        set_u32(&cache.pos_dev, past as u32)?;
        let idt = Tensor::from_vec(ids.to_vec(), vec![n], self.device).map_err(fe)?;
        let out = self.forward_dev_with(&idt, patches, cache, all_logits)?;
        cache.seq_len = past + n;
        Ok(out)
    }

    /// Шаг декода одного токена: после прогрева — проигрыш CUDA-графа.
    /// Возвращает логиты `[1, V]` (буфер графа: годен до следующего шага).
    pub fn decode(&self, tok: u32, cache: &mut ModelCache) -> R<Tensor> {
        let past = cache.seq_len;
        if past + 1 > cache.capacity {
            return Err(ModelError::Forward(format!("inkling2: кэш {} < {}", cache.capacity, past + 1)));
        }
        let graph_on = std::env::var("SYN_INKLING2_GRAPH").as_deref() != Ok("0") && self.device.is_cuda();
        set_u32(&cache.pos_dev, past as u32)?;
        set_u32(&cache.tok_dev, tok)?;
        if let (true, Some(g)) = (graph_on, &cache.graph) {
            g.0.launch().map_err(|e| ModelError::Forward(format!("граф декода: {e:?}")))?;
            cache.seq_len = past + 1;
            return Ok(cache.out_dev.clone().expect("буфер графа"));
        }
        if !graph_on || cache.warm < 2 {
            let tokt = cache.tok_dev.clone();
            let out = self.forward_dev(&tokt, cache, false)?;
            cache.warm += 1;
            cache.seq_len = past + 1;
            return Ok(out);
        }
        let Device::Cuda(ord) = self.device else { unreachable!() };
        let vu = self.cfg.text.unpadded_vocab_size;
        if cache.out_dev.is_none() {
            cache.out_dev = Some(Tensor::zeros(vec![1, vu], DType::F32, self.device).map_err(fe)?);
        }
        let stream = synaptix_core::device::cuda::default_stream(ord).map_err(fe)?;
        let tokt = cache.tok_dev.clone();
        let out_buf = cache.out_dev.clone().expect("буфер");
        let mut err: Option<ModelError> = None;
        let mut cap = synaptix_infer::graph_capture::GraphCapturer::new(0);
        let res = cap.capture_with(&stream, |_| {
            let r = self.forward_dev(&tokt, cache, false).and_then(|l| copy_into(&out_buf, &l));
            r.map_err(|e| {
                let m = e.to_string();
                err = Some(e);
                synaptix_infer::InferError::Other(m)
            })
        });
        match (res, err) {
            (Ok(g), None) => {
                g.launch().map_err(|e| ModelError::Forward(format!("граф декода: {e:?}")))?;
                cache.graph = Some(GraphHandle(g));
                cache.seq_len = past + 1;
                Ok(out_buf)
            }
            (_, Some(e)) => Err(e),
            (Err(e), None) => Err(ModelError::Forward(format!("граф декода: захват: {e}"))),
        }
    }

    /// Прогон по токенам на карте (`idt` U32 `[T]`) с позиции из `cache.pos_dev`;
    /// хост в нём не участвует, кроме маршрута MoE префилла (T > 1).
    fn forward_dev(&self, idt: &Tensor, cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        self.forward_dev_with(idt, &[], cache, all_logits)
    }

    fn forward_dev_with(&self, idt: &Tensor, patches: &[EmbedPatch], cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        let t = &self.cfg.text;
        let n = idt.dims()[0];
        let past = cache.seq_len;
        let dev = self.device;
        let x = if std::env::var_os("INK2_OLD_EMBED").is_some() {
            synaptix_ops::embed::token_embedding(idt, &self.embed).map_err(fe)?
        } else {
            self.embed.embed_gather(idt).map_err(fe)?
        };
        let x = f32c(&x.reshape(vec![n, t.hidden_size]).map_err(fe)?)?;
        let mut cur = kern::rms(&x, Some(&self.embed_norm), t.rms_norm_eps, false)?;
        for p in patches {
            let len = p.embeds.dims()[0];
            if p.row + len > n || p.embeds.dims().get(1) != Some(&t.hidden_size) {
                return Err(ModelError::Shape(format!("inkling2: эмбеддинги {:?} с строки {} в чанк {n}", p.embeds.dims(), p.row)));
            }
            copy_into(&cur.narrow(0, p.row, len).map_err(fe)?, &f32c(&p.embeds)?)?;
        }

        let (cs, ce) = self.cfg.core();
        let passes = self.cfg.num_passes();
        let mut blocks: Vec<Tensor> = Vec::new();
        let first_loop = if t.loop_config.is_some() { cs } else { t.num_hidden_layers };
        for ip in 0..first_loop {
            cur = self.layer(ip, ip, cur, &mut blocks, None, cache, past)?;
        }
        if t.loop_config.is_some() {
            let prelude = cur.clone();
            let prelude_blocks = blocks.clone();
            for pass in 0..passes {
                let mut bank: Vec<Tensor> = Vec::new();
                if pass > 0 {
                    bank.extend(blocks[prelude_blocks.len() + 1..].iter().cloned());
                    bank.push(cur.clone());
                    cur = prelude.clone();
                    blocks = prelude_blocks.clone();
                }
                for ip in cs..ce {
                    let slot = self.cfg.slot(ip, pass);
                    let messages = if pass > 0 {
                        let lp = self.layers[ip].lp.as_ref().ok_or_else(|| ModelError::Forward("inkling2: нет контроллеров".into()))?;
                        let refs: Vec<&Tensor> = bank.iter().collect();
                        Some(kern::mix(&refs, &lp.msg_q)?)
                    } else {
                        None
                    };
                    cur = self.layer(ip, slot, cur, &mut blocks, messages.as_ref(), cache, past)?;
                }
            }
            for ip in ce..t.num_hidden_layers {
                cur = self.layer(ip, ip, cur, &mut blocks, None, cache, past)?;
            }
        }
        let mut states: Vec<&Tensor> = blocks.iter().collect();
        states.push(&cur);
        let out = kern::pick(&kern::mix(&states, &self.output_q)?, 0)?;
        let out = f32c(&out)?;
        let out = if all_logits { out } else { f32c(&out.narrow(0, n - 1, 1).map_err(fe)?)? };
        let rows = out.dims()[0];
        let xn = kern::rms(&out, Some(&self.norm), t.rms_norm_eps, true)?;
        let logits = bf16c(&self.unembed.raw(&xn)?)?;
        let vu = t.unpadded_vocab_size;
        let res = empty(vec![rows, vu], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&logits)?;
        a.a[1] = addr(&res)?;
        a.u[0] = rows as u32;
        a.u[1] = logits.dims()[1] as u32;
        a.u[2] = vu as u32;
        a.u[3] = 1;
        a.f[0] = t.final_logit_softcapping;
        launch(dev, "ink2_logits", &a, ((vu as u32).div_ceil(256), rows as u32, 1), 256)?;
        Ok(res)
    }

    fn control(&self, x: &Tensor, m: &Tensor, c: &Ctrl) -> R<Tensor> {
        if std::env::var_os("INK2_OLD_CTRL").is_some() {
            let xr = kern::rms(x, None, 1e-6, false)?;
            let mr = kern::rms(m, None, 1e-6, false)?;
            let cat = Tensor::cat(&[&xr, &mr], 1).map_err(fe)?;
            let down_t = c.down.transpose(0, 1).map_err(fe)?.contiguous().map_err(fe)?;
            let d = cat.matmul(&down_t).map_err(fe)?;
            let gain = d.matmul(&c.gain).map_err(fe)?.tanh().map_err(fe)?;
            let forget = d.matmul(&c.forget).map_err(fe)?.tanh().map_err(fe)?;
            let keep = x.sub(&forget.mul(x).map_err(fe)?).map_err(fe)?;
            return f32c(&keep.add(&gain.mul(m).map_err(fe)?).map_err(fe)?);
        }
        let (n, h) = (x.dims()[0], x.dims()[1]);
        let dev = self.device;
        let d = empty(vec![n, c.rank], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(x)?;
        a.a[1] = addr(m)?;
        a.a[2] = addr(&c.down)?;
        a.a[3] = addr(&d)?;
        a.u[0] = n as u32;
        a.u[1] = h as u32;
        a.u[2] = c.rank as u32;
        launch(dev, "ink2_ctrl_down", &a, (n as u32, c.rank as u32, 1), 256)?;
        let out = empty(vec![n, h], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(x)?;
        a.a[1] = addr(m)?;
        a.a[2] = addr(&d)?;
        a.a[3] = addr(&c.gain)?;
        a.a[4] = addr(&c.forget)?;
        a.a[5] = addr(&out)?;
        a.u[0] = n as u32;
        a.u[1] = h as u32;
        a.u[2] = c.rank as u32;
        launch(dev, "ink2_ctrl_out", &a, ((h as u32).div_ceil(256), n as u32, 1), 256)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(&self, ip: usize, slot: usize, cur: Tensor, blocks: &mut Vec<Tensor>, messages: Option<&Tensor>, cache: &mut ModelCache, past: usize) -> R<Tensor> {
        let t = &self.cfg.text;
        let l = &self.layers[ip];
        let n = cur.dims()[0];
        let h = t.hidden_size;
        let (nh, nkv, d, dr) = (t.num_attention_heads, t.num_key_value_heads, t.head_dim, t.d_rel);
        let eps = t.rms_norm_eps;
        let dev = self.device;
        let inp_sa = cur.clone();

        let mut inputs = [cur.clone(), cur.clone(), cur.clone()];
        if let Some(q) = &l.attn_res_q {
            let mut states: Vec<&Tensor> = blocks.iter().collect();
            states.push(&cur);
            let mixed = kern::stage("mix", dev, || kern::mix(&states, q))?;
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = f32c(&kern::pick(&mixed, if l.swa { 0 } else { j })?)?;
            }
        }
        if let Some(m) = messages {
            let lp = l.lp.as_ref().expect("контроллеры есть, раз есть сообщения");
            for (j, inp) in inputs.iter_mut().enumerate() {
                let msg = f32c(&kern::pick(m, j)?)?;
                *inp = kern::stage("control", dev, || self.control(inp, &msg, &lp.ctrl[j]))?;
            }
        }
        let normed: [Tensor; 3] = if l.swa {
            let w = l.attn_norm.as_ref().expect("SWA-норма");
            if messages.is_some() {
                [kern::rms(&inputs[0], Some(w), eps, true)?, kern::rms(&inputs[1], Some(w), eps, true)?, kern::rms(&inputs[2], Some(w), eps, true)?]
            } else {
                let x = kern::rms(&inputs[0], Some(w), eps, true)?;
                [x.clone(), x.clone(), x]
            }
        } else {
            let w = l.qkv_norm.as_ref().expect("нормы q/k/v");
            [kern::rms(&inputs[0], Some(&w[0]), eps, true)?, kern::rms(&inputs[1], Some(&w[1]), eps, true)?, kern::rms(&inputs[2], Some(&w[2]), eps, true)?]
        };

        let sc = &mut cache.slots[slot];
        let k_scale = 1.0 / ((t.sconv_kernel_size + 1) as f32).sqrt();
        let kd_scale = 1.0 / ((2 * t.sconv_kernel_size + 1) as f32).sqrt();
        let col = |y: &Tensor, off: usize, len: usize| -> R<Tensor> { y.narrow(1, off, len).map_err(fe)?.contiguous().map_err(fe) };
        let ya = if std::env::var_os("INK2_UNFUSED").is_some() {
            // Отладка: те же веса по частям (срезы склейки) — сверка с раздельными GEMV.
            let Lin::Dense(lin) = &l.pa else { unreachable!() };
            let w = lin.weight();
            let mut outs = Vec::new();
            for (off, len) in [(0, nh * d), (nh * d, nh * dr), (nh * d + nh * dr, nh * d), (2 * nh * d + nh * dr, nh)] {
                let wp = w.narrow(0, off, len).map_err(fe)?.contiguous().map_err(fe)?;
                outs.push(Lin::Dense(Linear::new(wp, None).map_err(fe)?).f32(&normed[0])?);
            }
            let refs: Vec<&Tensor> = outs.iter().collect();
            f32c(&Tensor::cat(&refs, 1).map_err(fe)?)?
        } else {
            kern::stage("proj_a", dev, || l.pa.f32(&normed[0]))?
        };
        let (q, r, gate, sink) = (col(&ya, 0, nh * d)?, col(&ya, nh * d, nh * dr)?, col(&ya, nh * d + nh * dr, nh * d)?, col(&ya, 2 * nh * d + nh * dr, nh)?);
        let (yk, yv) = kern::stage("proj_kv", dev, || -> R<_> { Ok((l.pk.f32(&normed[1])?, l.pv.f32(&normed[2])?)) })?;
        let kk = t.sconv_kernel_size;
        let (kc, vc) = if l.dyn_conv {
            let (kraw, kdyn) = (col(&yk, 0, nkv * d)?, col(&yk, nkv * d, nkv * kk)?);
            let (vraw, vdyn) = (col(&yv, 0, nkv * d)?, col(&yv, nkv * d, nkv * kk)?);
            (
                kern::conv(&kraw, &l.k_conv, &sc.k_state, Some((&kdyn, nkv)), kd_scale)?,
                kern::conv(&vraw, &l.v_conv, &sc.v_state, Some((&vdyn, nkv)), kd_scale)?,
            )
        } else {
            (kern::conv(&yk, &l.k_conv, &sc.k_state, None, k_scale)?, kern::conv(&yv, &l.v_conv, &sc.v_state, None, k_scale)?)
        };
        let q_out = empty(vec![n, nh * d], DType::F32, dev)?;
        let r_out = empty(vec![n, nh * dr], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&q)?;
        a.a[1] = addr(&kc)?;
        a.a[2] = addr(&vc)?;
        a.a[3] = addr(&r)?;
        a.a[4] = addr(&l.q_clip)?;
        a.a[5] = addr(&l.r_clip)?;
        a.a[6] = addr(&q_out)?;
        a.a[7] = addr(&r_out)?;
        a.a[8] = addr(&sc.k)?;
        a.a[9] = addr(&sc.v)?;
        a.u[0] = n as u32;
        a.u[1] = nh as u32;
        a.u[2] = nkv as u32;
        a.u[3] = d as u32;
        a.u[4] = dr as u32;
        a.u[5] = sc.k.dims()[1] as u32;
        a.u[6] = past as u32;
        a.u[7] = (!l.swa) as u32;
        a.u[9] = sc.ring as u32;
        a.u[8] = t.log_scaling_n_floor as u32;
        a.f[0] = t.qk_l2_clip_threshold;
        a.f[1] = t.rel_l2_clip_threshold;
        a.f[2] = eps;
        a.f[3] = t.log_scaling_alpha;
        if std::env::var_os("INK2_HOST_POS").is_none() { a.a[10] = addr(&cache.pos_dev)?; }
        kern::stage("k_qkv", dev, || launch(dev, "ink2_qkv", &a, (n as u32, (nh + 2 * nkv) as u32, 1), d as u32))?;

        let att = empty(vec![n, nh * d], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&q_out)?;
        a.a[1] = addr(&r_out)?;
        a.a[2] = addr(&l.rel_proj)?;
        a.a[3] = addr(&sink)?;
        a.a[4] = addr(&sc.k)?;
        a.a[5] = addr(&sc.v)?;
        a.a[6] = addr(&gate)?;
        a.a[7] = addr(&att)?;
        a.u[0] = n as u32;
        a.u[1] = nh as u32;
        a.u[2] = nkv as u32;
        a.u[3] = d as u32;
        a.u[4] = dr as u32;
        a.u[5] = sc.k.dims()[1] as u32;
        a.u[6] = past as u32;
        a.u[7] = if l.swa { t.sliding_window_size as u32 } else { 0 };
        a.u[10] = sc.ring as u32;
        a.u[8] = l.extent as u32;
        a.f[0] = 1.0 / (d as f32).sqrt();
        if std::env::var_os("INK2_HOST_POS").is_none() { a.a[10] = addr(&cache.pos_dev)?; }
        // Части по ключам на декоде: число фиксировано от ёмкости окна (граф), ~64 позиции на часть.
        let span = if l.swa { t.sliding_window_size.min(cache.capacity) } else { cache.capacity };
        let splits = if n == 1 { span.div_ceil(64).clamp(1, 32) } else { 1 };
        let parts = if splits > 1 { Some(empty(vec![n * nh * splits * 130], DType::F32, dev)?) } else { None };
        if let Some(p) = &parts {
            a.a[8] = addr(p)?;
            a.u[9] = splits as u32;
        }
        kern::stage("k_attn", dev, || -> R<()> {
            launch(dev, "ink2_attn", &a, (n as u32, nh as u32, splits as u32), 128)?;
            if parts.is_some() {
                launch(dev, "ink2_attn_merge", &a, (n as u32, nh as u32, 1), 128)?;
            }
            Ok(())
        })?;
        let o = kern::stage("proj_o", dev, || l.wo.f32(&att))?;
        let mut cur = kern::conv(&o, &l.attn_post_conv, &sc.attn_state, None, k_scale)?;

        if ip % t.n_layers_per_attn_res_block == 0 {
            blocks.push(inp_sa.clone());
        } else {
            cur = f32c(&inp_sa.add(&cur).map_err(fe)?)?;
        }
        let ffn_inp = cur.clone();
        let mut states: Vec<&Tensor> = blocks.iter().collect();
        states.push(&cur);
        let mut m = f32c(&kern::pick(&kern::mix(&states, &l.ffn_res_q)?, 0)?)?;
        if let Some(msg) = messages {
            let lp = l.lp.as_ref().expect("контроллеры");
            let mm = f32c(&kern::pick(msg, 3)?)?;
            m = self.control(&m, &mm, &lp.ctrl[3])?;
        }
        let xn = kern::rms(&m, Some(&l.ffn_norm), eps, true)?;
        let y = kern::stage("moe", dev, || self.moe(l, &xn))?;
        let sc = &mut cache.slots[slot];
        let y = kern::conv(&y, &l.ffn_post_conv, &sc.ffn_state, None, k_scale)?;
        let _ = h;
        f32c(&ffn_inp.add(&y).map_err(fe)?)
    }

    /// MoE: `xn [T, H]` BF16 → `[T, H]` F32. Декод (T = 1) — целиком на карте,
    /// без чтения маршрута хостом; префилл — групповой GEMM по экспертам.
    fn moe(&self, l: &Layer, xn: &Tensor) -> R<Tensor> {
        let t = &self.cfg.text;
        let n = xn.dims()[0];
        let (e, k, hdim, inter) = (t.n_routed_experts, t.num_experts_per_tok, t.hidden_size, t.intermediate_size);
        let dev = self.device;
        let ex = &l.experts;
        let xf = f32c(xn)?;
        let logits = empty(vec![n, e], DType::F32, dev)?;
        if std::env::var_os("INK2_OLD_ROUTER").is_some() {
            let rt = l.router.transpose(0, 1).map_err(fe)?.contiguous().map_err(fe)?;
            let lg = f32c(&xf.matmul(&rt).map_err(fe)?)?;
            return self.moe_after_logits(l, xn, lg);
        }
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&xf)?;
        a.a[1] = addr(&l.router)?;
        a.a[2] = addr(&logits)?;
        a.u[0] = n as u32;
        a.u[1] = hdim as u32;
        a.u[2] = e as u32;
        launch(dev, "ink2_router", &a, (n as u32, (e as u32).div_ceil(8), 1), 256)?;
        self.moe_after_logits(l, xn, logits)
    }

    fn moe_after_logits(&self, l: &Layer, xn: &Tensor, logits: Tensor) -> R<Tensor> {
        let t = &self.cfg.text;
        let n = xn.dims()[0];
        let (e, k, hdim, inter) = (t.n_routed_experts, t.num_experts_per_tok, t.hidden_size, t.intermediate_size);
        let dev = self.device;
        let ex = &l.experts;
        let idx = empty(vec![n * k], DType::U32, dev)?;
        let wts = empty(vec![n * k], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&logits)?;
        a.a[1] = addr(&l.router_bias)?;
        a.a[2] = addr(&idx)?;
        a.a[3] = addr(&wts)?;
        a.a[4] = addr(&ex.alpha_down_dev)?;
        a.u[0] = n as u32;
        a.u[1] = e as u32;
        a.u[2] = k as u32;
        a.f[0] = t.route_scale;
        if std::env::var_os("INK2_OLD_ROUTE").is_some() { launch(dev, "ink2_route_blk", &a, (n as u32, 1, 1), 128)?; } else { launch(dev, "ink2_route", &a, (n as u32, 1, 1), 32)?; }
        if n == 1 {
            let gu = ex.gate_up.gemv_indexed_dense(&idx, xn, false).map_err(fe)?;
            let h = self.act(&gu, &ex.alpha_gate_up_dev, Some(&idx), inter)?;
            let mut acc = Tensor::zeros(vec![hdim], DType::F32, dev).map_err(fe)?;
            ex.down.gemv_indexed_dense_acc(&idx, &h, true, &wts, &mut acc).map_err(fe)?;
            return acc.reshape(vec![1, hdim]).map_err(fe);
        }
        let idx_h: Vec<u32> = idx.to_device(Device::Cpu).map_err(fe)?.to_vec1::<u32>().map_err(fe)?;
        let pairs = n * k;
        let mut order: Vec<usize> = (0..pairs).collect();
        order.sort_by_key(|&p| (idx_h[p], p));
        let mut rowmap = vec![0u32; pairs];
        let mut rows = Vec::with_capacity(pairs);
        let mut alpha_rows = Vec::with_capacity(pairs);
        let mut segments: Vec<(u32, u32, u32)> = Vec::new();
        for (pos, &p) in order.iter().enumerate() {
            let ei = idx_h[p];
            rowmap[p] = pos as u32;
            rows.push((p / k) as u32);
            alpha_rows.push(ex.alpha_gate_up[ei as usize]);
            match segments.last_mut() {
                Some(s) if s.0 == ei => s.2 = pos as u32 + 1,
                _ => segments.push((ei, pos as u32, pos as u32 + 1)),
            }
        }
        let row_idx = Tensor::from_vec(rows, vec![pairs], dev).map_err(fe)?;
        let gu = ex.gate_up.gemm_grouped_dense_rows(xn, Some(&row_idx), &segments, false).map_err(fe)?;
        let at = Tensor::from_vec(alpha_rows, vec![pairs], dev).map_err(fe)?;
        let h = self.act(&gu, &at, None, inter)?;
        let y = bf16c(&ex.down.gemm_grouped_dense_rows(&h, None, &segments, false).map_err(fe)?)?;
        let rm = Tensor::from_vec(rowmap, vec![pairs], dev).map_err(fe)?;
        let out = empty(vec![n, hdim], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&y)?;
        a.a[1] = addr(&wts)?;
        a.a[2] = addr(&rm)?;
        a.a[3] = addr(&out)?;
        a.u[0] = n as u32;
        a.u[1] = hdim as u32;
        a.u[2] = k as u32;
        launch(dev, "ink2_combine", &a, ((hdim as u32).div_ceil(256), n as u32, 1), 256)?;
        Ok(out)
    }

    /// Активация экспертов: `gu [R, 2I]`, множитель — `alpha[r]` либо `alpha[idx[r]]` → `[R, I]` BF16.
    fn act(&self, gu: &Tensor, alpha: &Tensor, idx: Option<&Tensor>, inter: usize) -> R<Tensor> {
        let gu = bf16c(gu)?;
        let rows = gu.dims()[0];
        let out = empty(vec![rows, inter], DType::BF16, self.device)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&gu)?;
        a.a[1] = addr(alpha)?;
        a.a[2] = addr(&out)?;
        a.a[3] = match idx { Some(i) => addr(i)?, None => 0 };
        a.u[0] = rows as u32;
        a.u[1] = inter as u32;
        a.f[0] = self.cfg.text.mlp_gate_softcap;
        a.f[1] = self.cfg.text.mlp_up_softcap;
        launch(self.device, "ink2_act", &a, ((inter as u32).div_ceil(256), rows as u32, 1), 256)?;
        Ok(out)
    }
}

/// D2D-копия `src` в постоянный буфер `dst` той же формы (на default-стриме).
fn copy_into(dst: &Tensor, src: &Tensor) -> R<()> {
    let src = if src.is_contiguous() { src.clone() } else { src.contiguous().map_err(fe)? };
    let (d, dl) = dst.cuda_region().map_err(fe)?;
    let (s, sl) = src.cuda_region().map_err(fe)?;
    if dl != sl {
        return Err(ModelError::Shape("inkling2: размеры буферов графа".into()));
    }
    let Device::Cuda(ord) = dst.device() else { return Err(ModelError::Forward("не CUDA".into())) };
    let ds = synaptix_core::device::cuda::default_stream(ord).map_err(fe)?;
    synaptix_core::device::cuda::dtod_into_region(&ds, d, s, dl).map_err(fe)
}

/// Записать число в буфер на карте по порядку потока: синхронная копия с хоста
/// не ждёт ядер, которые ещё читают прежнее значение (прошлый чанк префилла).
fn set_u32(dst: &Tensor, v: u32) -> R<()> {
    let src = Tensor::from_vec(vec![v], vec![1], dst.device()).map_err(fe)?;
    copy_into(dst, &src)
}
