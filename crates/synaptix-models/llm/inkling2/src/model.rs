//! Прямой проход Inkling-2 (схема — llama.cpp `inkling2.cpp` + `inkling2-loop.cpp`).
//!
//! Поток между слоями и блоки attention-residuals — F32; проекции — BF16-веса с
//! BF16-активацией (как в эталоне); KV — BF16; эксперты — квант-веса движка с
//! плотной BF16-активацией (только веса квантованы, как у modelopt-эталона).

use synaptix_core::backend::Ink2Launch;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::ModelError;
use synaptix_nn::linear::Linear;
use synaptix_nn::Module;

use crate::config::Inkling2Config;
use crate::kern::{self, addr, bf16c, empty, f32c, fe, launch, R};
use crate::loader::{ExpertStack, Inkling2Weights, LoadError, LLM};

/// Плотная проекция с BF16-весом `[out, in]`.
pub struct Lin(Linear);

impl Lin {
    fn load(w: &Inkling2Weights, name: &str, dev: Device) -> Result<Self, LoadError> {
        let t = w.tensor(name, dev, DType::BF16)?;
        Ok(Self(Linear::new(t, None).map_err(|e| LoadError::Weights(format!("{name}: {e}")))?))
    }

    /// `x [T, in]` (любой dtype) → `[T, out]` F32.
    fn f32(&self, x: &Tensor) -> R<Tensor> {
        let x = bf16c(x)?;
        f32c(&self.0.forward(&x).map_err(fe)?)
    }

    fn bf16(&self, x: &Tensor) -> R<Tensor> {
        let x = bf16c(x)?;
        bf16c(&self.0.forward(&x).map_err(fe)?)
    }
}

/// Контроллер входа повторного прохода: `(1 − tanh(F·d))·x + tanh(G·d)·m`,
/// `d = Down·[rms(x); rms(m)]`. Веса F32, транспонированы под `x · W`.
struct Ctrl {
    down_t: Tensor,
    gain_t: Tensor,
    forget_t: Tensor,
}

struct LoopLayer {
    /// Запросы сообщений `[4, H]`: q, k, v, mlp.
    msg_q: Tensor,
    ctrl: [Ctrl; 4],
}

struct Experts {
    gate_up: Vec<QuantWeight>,
    down: Vec<QuantWeight>,
    alpha_gate_up: Vec<f32>,
    alpha_down: Vec<f32>,
}

pub struct Layer {
    swa: bool,
    attn_norm: Option<Tensor>,
    qkv_norm: Option<[Tensor; 3]>,
    wq: Lin,
    wk: Lin,
    wv: Lin,
    wo: Lin,
    wr: Lin,
    wgate: Lin,
    wsink: Lin,
    k_conv: Tensor,
    v_conv: Tensor,
    k_conv_proj: Option<Lin>,
    v_conv_proj: Option<Lin>,
    q_clip: Tensor,
    r_clip: Tensor,
    rel_proj: Tensor,
    extent: usize,
    attn_res_q: Option<Tensor>,
    ffn_res_q: Tensor,
    attn_post_conv: Tensor,
    ffn_post_conv: Tensor,
    ffn_norm: Tensor,
    router_t: Tensor,
    router_bias: Tensor,
    experts: Experts,
    lp: Option<LoopLayer>,
}

/// Кэш одного слота (физический слой на своём проходе цикла).
pub struct SlotCache {
    k: Tensor,
    v: Tensor,
    k_state: Tensor,
    v_state: Tensor,
    attn_state: Tensor,
    ffn_state: Tensor,
}

pub struct ModelCache {
    pub slots: Vec<SlotCache>,
    pub seq_len: usize,
    pub capacity: usize,
}

impl ModelCache {
    pub fn reset(&mut self) -> R<()> {
        self.seq_len = 0;
        for s in self.slots.iter_mut() {
            for t in [&mut s.k_state, &mut s.v_state, &mut s.attn_state, &mut s.ffn_state] {
                *t = Tensor::zeros(t.dims().to_vec(), DType::F32, t.device()).map_err(fe)?;
            }
        }
        Ok(())
    }
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
                    Some(Lin::load(w, &format!("{a}.k_sconv.weight_proj"), dev)?),
                    Some(Lin::load(w, &format!("{a}.v_sconv.weight_proj"), dev)?),
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
            let router_t = dev_t(&router, e, h, dev)?;
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
                        down_t: dev_t(&w.host_f32(&format!("{c}.down_weight"))?, rank, 2 * h, dev)?,
                        gain_t: dev_t(&w.host_f32(&format!("{c}.gain_weight"))?, h, rank, dev)?,
                        forget_t: dev_t(&w.host_f32(&format!("{c}.forget_weight"))?, h, rank, dev)?,
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
                wq: Lin::load(w, &format!("{a}.wq_du.weight"), dev)?,
                wk: Lin::load(w, &format!("{a}.wk_dv.weight"), dev)?,
                wv: Lin::load(w, &format!("{a}.wv_dv.weight"), dev)?,
                wo: Lin::load(w, &format!("{a}.wo_ud.weight"), dev)?,
                wr: Lin::load(w, &format!("{a}.wr_du.weight"), dev)?,
                wgate: Lin::load(w, &format!("{a}.gate_attn.weight"), dev)?,
                wsink: Lin::load(w, &format!("{a}.attn_sink_proj.weight"), dev)?,
                k_conv,
                v_conv,
                k_conv_proj,
                v_conv_proj,
                q_clip: f32t(&format!("{a}.q_l2_clip.scale"))?,
                r_clip: f32t(&format!("{a}.rel_l2_clip.scale"))?,
                rel_proj: f32t(&format!("{a}.rel_logits_proj.proj"))?,
                extent,
                attn_res_q,
                ffn_res_q,
                attn_post_conv: conv_w(w, &format!("{p}.attn_sconv.weight"), k, dev)?,
                ffn_post_conv: conv_w(w, &format!("{p}.mlp_sconv.weight"), k, dev)?,
                ffn_norm: f32t(&format!("{p}.mlp_norm.weight"))?,
                router_t,
                router_bias,
                experts: Experts { gate_up, down, alpha_gate_up, alpha_down },
                lp,
            });
        }
        Ok(Self { cfg, device: dev, embed, embed_norm, unembed, norm, output_q, layers })
    }

    pub fn make_cache(&self, capacity: usize) -> R<ModelCache> {
        let t = &self.cfg.text;
        let (nkv, d, h, k) = (t.num_key_value_heads, t.head_dim, t.hidden_size, t.sconv_kernel_size);
        let dev = self.device;
        let z = |shape: Vec<usize>, dt: DType| Tensor::zeros(shape, dt, dev).map_err(fe);
        let mut slots = Vec::with_capacity(self.cfg.num_slots());
        for _ in 0..self.cfg.num_slots() {
            slots.push(SlotCache {
                k: z(vec![nkv, capacity, d], DType::BF16)?,
                v: z(vec![nkv, capacity, d], DType::BF16)?,
                k_state: z(vec![k - 1, nkv * d], DType::F32)?,
                v_state: z(vec![k - 1, nkv * d], DType::F32)?,
                attn_state: z(vec![k - 1, h], DType::F32)?,
                ffn_state: z(vec![k - 1, h], DType::F32)?,
            });
        }
        Ok(ModelCache { slots, seq_len: 0, capacity })
    }

    /// Прогон токенов `ids` с позиции `cache.seq_len`. Возвращает логиты
    /// `[1, V]` последнего токена (или всех `[T, V]` при `all_logits`), F32,
    /// словарь без паддинга.
    pub fn forward(&self, ids: &[u32], cache: &mut ModelCache, all_logits: bool) -> R<Tensor> {
        let t = &self.cfg.text;
        let n = ids.len();
        let past = cache.seq_len;
        if past + n > cache.capacity {
            return Err(ModelError::Forward(format!("inkling2: кэш {} < {}", cache.capacity, past + n)));
        }
        let dev = self.device;
        let idt = Tensor::from_vec(ids.to_vec(), vec![n], dev).map_err(fe)?;
        let x = synaptix_ops::embed::token_embedding(&idt, &self.embed).map_err(fe)?;
        let x = f32c(&x.reshape(vec![n, t.hidden_size]).map_err(fe)?)?;
        let mut cur = kern::rms(&x, Some(&self.embed_norm), t.rms_norm_eps, false)?;

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
        let logits = bf16c(&self.unembed.0.forward(&xn).map_err(fe)?)?;
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
        cache.seq_len = past + n;
        Ok(res)
    }

    fn control(&self, x: &Tensor, m: &Tensor, c: &Ctrl) -> R<Tensor> {
        let eps = 1e-6;
        let xr = kern::rms(x, None, eps, false)?;
        let mr = kern::rms(m, None, eps, false)?;
        let cat = Tensor::cat(&[&xr, &mr], 1).map_err(fe)?;
        let d = cat.matmul(&c.down_t).map_err(fe)?;
        let gain = d.matmul(&c.gain_t).map_err(fe)?.tanh().map_err(fe)?;
        let forget = d.matmul(&c.forget_t).map_err(fe)?.tanh().map_err(fe)?;
        let keep = x.sub(&forget.mul(x).map_err(fe)?).map_err(fe)?;
        f32c(&keep.add(&gain.mul(m).map_err(fe)?).map_err(fe)?)
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
            let mixed = kern::mix(&states, q)?;
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = f32c(&kern::pick(&mixed, if l.swa { 0 } else { j })?)?;
            }
        }
        if let Some(m) = messages {
            let lp = l.lp.as_ref().expect("контроллеры есть, раз есть сообщения");
            for (j, inp) in inputs.iter_mut().enumerate() {
                let msg = f32c(&kern::pick(m, j)?)?;
                *inp = self.control(inp, &msg, &lp.ctrl[j])?;
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
        let q = l.wq.f32(&normed[0])?;
        let kraw = l.wk.f32(&normed[1])?;
        let vraw = l.wv.f32(&normed[2])?;
        let (kc, vc) = match (&l.k_conv_proj, &l.v_conv_proj) {
            (Some(kp), Some(vp)) => {
                let kdyn = kp.f32(&normed[1])?;
                let vdyn = vp.f32(&normed[2])?;
                (
                    kern::conv(&kraw, &l.k_conv, &sc.k_state, Some((&kdyn, nkv)), kd_scale)?,
                    kern::conv(&vraw, &l.v_conv, &sc.v_state, Some((&vdyn, nkv)), kd_scale)?,
                )
            }
            _ => (kern::conv(&kraw, &l.k_conv, &sc.k_state, None, k_scale)?, kern::conv(&vraw, &l.v_conv, &sc.v_state, None, k_scale)?),
        };
        let r = l.wr.f32(&normed[0])?;
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
        a.u[5] = cache.capacity as u32;
        a.u[6] = past as u32;
        a.u[7] = (!l.swa) as u32;
        a.u[8] = t.log_scaling_n_floor as u32;
        a.f[0] = t.qk_l2_clip_threshold;
        a.f[1] = t.rel_l2_clip_threshold;
        a.f[2] = eps;
        a.f[3] = t.log_scaling_alpha;
        launch(dev, "ink2_qkv", &a, (n as u32, (nh + 2 * nkv) as u32, 1), d as u32)?;

        let sink = l.wsink.f32(&normed[0])?;
        let gate = l.wgate.f32(&normed[0])?;
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
        a.u[5] = cache.capacity as u32;
        a.u[6] = past as u32;
        a.u[7] = if l.swa { t.sliding_window_size as u32 } else { 0 };
        a.u[8] = l.extent as u32;
        a.f[0] = 1.0 / (d as f32).sqrt();
        launch(dev, "ink2_attn", &a, (n as u32, nh as u32, 1), 128)?;
        let o = l.wo.f32(&att)?;
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
        let y = self.moe(l, &xn)?;
        let sc = &mut cache.slots[slot];
        let y = kern::conv(&y, &l.ffn_post_conv, &sc.ffn_state, None, k_scale)?;
        let _ = h;
        f32c(&ffn_inp.add(&y).map_err(fe)?)
    }

    /// MoE: `xn [T, H]` BF16 → `[T, H]` F32.
    fn moe(&self, l: &Layer, xn: &Tensor) -> R<Tensor> {
        let t = &self.cfg.text;
        let n = xn.dims()[0];
        let (e, k, hdim, inter) = (t.n_routed_experts, t.num_experts_per_tok, t.hidden_size, t.intermediate_size);
        let dev = self.device;
        let xf = f32c(xn)?;
        let logits = f32c(&xf.matmul(&l.router_t).map_err(fe)?)?;
        let idx = empty(vec![n, k], DType::U32, dev)?;
        let wts = empty(vec![n, k], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&logits)?;
        a.a[1] = addr(&l.router_bias)?;
        a.a[2] = addr(&idx)?;
        a.a[3] = addr(&wts)?;
        a.u[0] = n as u32;
        a.u[1] = e as u32;
        a.u[2] = k as u32;
        a.f[0] = t.route_scale;
        launch(dev, "ink2_route", &a, (n as u32, 1, 1), 128)?;
        let idx_h: Vec<u32> = idx.reshape(vec![n * k]).map_err(fe)?.to_device(Device::Cpu).map_err(fe)?.to_vec1::<u32>().map_err(fe)?;
        let w_h: Vec<f32> = wts.reshape(vec![n * k]).map_err(fe)?.to_device(Device::Cpu).map_err(fe)?.to_vec1::<f32>().map_err(fe)?;
        let ex = &l.experts;
        let pairs = n * k;
        // Строки выхода экспертов в порядке групп; rowmap[пара] — строка пары.
        let mut rowmap = vec![0u32; pairs];
        let (y, alpha_rows_down): (Tensor, Vec<f32>) = if pairs <= 64 {
            let gu_w: Vec<&QuantWeight> = idx_h.iter().map(|&i| &ex.gate_up[i as usize]).collect();
            let rows: Vec<usize> = (0..pairs).map(|p| p / k).collect();
            let gu = QuantWeight::gemv_batched_dense(&gu_w, xn, &rows).map_err(fe)?;
            let alpha: Vec<f32> = idx_h.iter().map(|&i| ex.alpha_gate_up[i as usize]).collect();
            let hact = self.act(&gu, &alpha, inter)?;
            let dn_w: Vec<&QuantWeight> = idx_h.iter().map(|&i| &ex.down[i as usize]).collect();
            let rows: Vec<usize> = (0..pairs).collect();
            let y = QuantWeight::gemv_batched_dense(&dn_w, &hact, &rows).map_err(fe)?;
            for (p, r) in rowmap.iter_mut().enumerate() {
                *r = p as u32;
            }
            (y, idx_h.iter().map(|&i| ex.alpha_down[i as usize]).collect())
        } else {
            let mut groups: Vec<Vec<usize>> = vec![Vec::new(); e];
            for (p, &i) in idx_h.iter().enumerate() {
                groups[i as usize].push(p);
            }
            let mut parts: Vec<Tensor> = Vec::new();
            let mut alpha_rows = vec![0f32; pairs];
            let mut row = 0usize;
            for (ei, g) in groups.iter().enumerate() {
                if g.is_empty() {
                    continue;
                }
                let toks: Vec<u32> = g.iter().map(|&p| (p / k) as u32).collect();
                let tt = Tensor::from_vec(toks, vec![g.len()], dev).map_err(fe)?;
                let xe = bf16c(&xn.index_select(0, &tt).map_err(fe)?)?;
                let wgu = ex.gate_up[ei].dequantize(DType::BF16).map_err(fe)?;
                let gu = bf16c(&Linear::new(wgu, None).map_err(fe)?.forward(&xe).map_err(fe)?)?;
                let hact = self.act(&gu, &vec![ex.alpha_gate_up[ei]; g.len()], inter)?;
                let wd = ex.down[ei].dequantize(DType::BF16).map_err(fe)?;
                let ye = bf16c(&Linear::new(wd, None).map_err(fe)?.forward(&hact).map_err(fe)?)?;
                for &p in g {
                    rowmap[p] = row as u32;
                    alpha_rows[row] = ex.alpha_down[ei];
                    row += 1;
                }
                parts.push(ye);
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            (bf16c(&Tensor::cat(&refs, 0).map_err(fe)?)?, alpha_rows)
        };
        let wcomb: Vec<f32> = (0..pairs).map(|p| w_h[p] * alpha_rows_down[rowmap[p] as usize]).collect();
        let wt = Tensor::from_vec(wcomb, vec![pairs], dev).map_err(fe)?;
        let rm = Tensor::from_vec(rowmap, vec![pairs], dev).map_err(fe)?;
        let out = empty(vec![n, hdim], DType::F32, dev)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&y)?;
        a.a[1] = addr(&wt)?;
        a.a[2] = addr(&rm)?;
        a.a[3] = addr(&out)?;
        a.u[0] = n as u32;
        a.u[1] = hdim as u32;
        a.u[2] = k as u32;
        launch(dev, "ink2_combine", &a, ((hdim as u32).div_ceil(256), n as u32, 1), 256)?;
        Ok(out)
    }

    /// Активация экспертов: `gu [R, 2I]` BF16, множитель на строку → `[R, I]` BF16.
    fn act(&self, gu: &Tensor, alpha: &[f32], inter: usize) -> R<Tensor> {
        let gu = bf16c(gu)?;
        let rows = gu.dims()[0];
        let at = Tensor::from_vec(alpha.to_vec(), vec![rows], self.device).map_err(fe)?;
        let out = empty(vec![rows, inter], DType::BF16, self.device)?;
        let mut a = Ink2Launch::default();
        a.a[0] = addr(&gu)?;
        a.a[1] = addr(&at)?;
        a.a[2] = addr(&out)?;
        a.u[0] = rows as u32;
        a.u[1] = inter as u32;
        a.f[0] = self.cfg.text.mlp_gate_softcap;
        a.f[1] = self.cfg.text.mlp_up_softcap;
        launch(self.device, "ink2_act", &a, ((inter as u32).div_ceil(256), rows as u32, 1), 256)?;
        Ok(out)
    }
}
