use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_range, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

use crate::config::DecoderConfig;
use crate::loader::WeightSource;
use crate::{err, Result, VibeVoiceError};

const NEG_LARGE: f32 = -1.0e30;

struct Attn {
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    o_w: Tensor,
}

struct Mlp {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

struct Layer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: Attn,
    mlp: Mlp,
}

impl Layer {
    /// Копия слоя на `dev` — только копирование, без ядер (так её можно
    /// звать из потока префетча на loader-стриме).
    fn to_device(&self, dev: Device) -> Result<Self> {
        let t = |x: &Tensor| x.to_device(dev).map_err(err);
        Ok(Self {
            input_ln: t(&self.input_ln)?,
            post_ln: t(&self.post_ln)?,
            attn: Attn {
                q_w: t(&self.attn.q_w)?,
                q_b: t(&self.attn.q_b)?,
                k_w: t(&self.attn.k_w)?,
                k_b: t(&self.attn.k_b)?,
                v_w: t(&self.attn.v_w)?,
                v_b: t(&self.attn.v_b)?,
                o_w: t(&self.attn.o_w)?,
            },
            mlp: Mlp { gate: t(&self.mlp.gate)?, up: t(&self.mlp.up)?, down: t(&self.mlp.down)? },
        })
    }

    fn bytes(&self) -> usize {
        let a = &self.attn;
        [
            &self.input_ln,
            &self.post_ln,
            &a.q_w,
            &a.q_b,
            &a.k_w,
            &a.k_b,
            &a.v_w,
            &a.v_b,
            &a.o_w,
            &self.mlp.gate,
            &self.mlp.up,
            &self.mlp.down,
        ]
        .iter()
        .map(|t| t.dtype().bytes_for_numel(t.numel()))
        .sum()
    }
}

pub struct KvCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    len: usize,
    cap: usize,
}

impl KvCache {
    pub fn new(
        layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cap: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let mut k = Vec::with_capacity(layers);
        let mut v = Vec::with_capacity(layers);
        for _ in 0..layers {
            k.push(
                Tensor::zeros(vec![1usize, num_kv_heads, cap, head_dim], dtype, device)
                    .map_err(err)?,
            );
            v.push(
                Tensor::zeros(vec![1usize, num_kv_heads, cap, head_dim], dtype, device)
                    .map_err(err)?,
            );
        }
        Ok(Self { k, v, len: 0, cap })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor> {
    if group == 1 {
        return x.contiguous().map_err(err);
    }
    let d = x.dims().to_vec();
    let (b, nkv, s, hd) = (d[0], d[1], d[2], d[3]);
    x.unsqueeze(2)
        .and_then(|t| t.broadcast_as(vec![b, nkv, group, s, hd]))
        .and_then(|t| t.contiguous())
        .and_then(|t| t.reshape(vec![b, nkv * group, s, hd]))
        .map_err(err)
}

fn causal_bias(q_len: usize, kv_len: usize, device: Device) -> Result<Tensor> {
    let past = kv_len - q_len;
    let mut data = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            if j > past + i {
                data[i * kv_len + j] = NEG_LARGE;
            }
        }
    }
    Tensor::from_vec(data, vec![1usize, 1, q_len, kv_len], device).map_err(err)
}

pub struct Qwen2Model {
    /// На CUDA — в RAM (как и `lm_head`): обе таблицы по 1.1 ГБ у 7B, а
    /// нужны из них только строки (токены промпта и шага, пара строк
    /// ограниченной головы) — выбираются на хосте и едут на карту.
    embed: Tensor,
    layers: Vec<Layer>,
    /// Слои `host_from..` лежат пиннованной копией на хосте и приезжают на
    /// карту по одному во время forward (следующий — на loader-стриме):
    /// 7B-модель в BF16 (~13 ГБ слоёв) на малой карте.
    host_from: usize,
    final_norm: Tensor,
    lm_head: Tensor,
    rope: RopeCache,
    pub cfg: DecoderConfig,
    pub device: Device,
    pub dtype: DType,
}

impl Qwen2Model {
    /// `reserve` — сколько VRAM оставить свободной после слоёв (KV, активации,
    /// VAE-декод): слои, которые его съели бы, уходят на хост.
    pub fn load(
        src: &dyn WeightSource,
        cfg: &DecoderConfig,
        prefix: &str,
        lm_head_name: &str,
        rope_capacity: usize,
        reserve: usize,
    ) -> Result<Self> {
        let final_norm = src.get(&format!("{prefix}.norm.weight"))?;
        let dev = final_norm.device();
        let table_dev = if dev.is_cuda() { Device::Cpu } else { dev };
        let embed = src.get_on(&format!("{prefix}.embed_tokens.weight"), table_dev)?;
        let lm_head = if src.has(lm_head_name) {
            src.get_on(lm_head_name, table_dev)?
        } else {
            embed.clone()
        };
        let n = cfg.num_hidden_layers;
        let free = || match dev {
            Device::Cuda(o) => synaptix_core::device::cuda::mem_info(o).map(|(f, _)| f).unwrap_or(0),
            _ => usize::MAX,
        };
        let trim = || {
            if let Device::Cuda(o) = dev {
                let _ = synaptix_core::device::cuda::synchronize_all(o);
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
            }
        };
        let mut host_from = n;
        let mut blk = 0usize;
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            // Запас — два стримящихся слоя сверх заказанного.
            if host_from == n && i > 0 && dev.is_cuda() && free() < reserve + 3 * blk {
                trim();
                if free() < reserve + 3 * blk {
                    host_from = i;
                    eprintln!("[vibevoice] VRAM: слои LM {i}..{n} на хост, стримятся в forward");
                }
            }
            let k = |s: &str| format!("{prefix}.layers.{i}.{s}");
            let layer = Layer {
                input_ln: src.get(&k("input_layernorm.weight"))?,
                post_ln: src.get(&k("post_attention_layernorm.weight"))?,
                attn: Attn {
                    q_w: src.get(&k("self_attn.q_proj.weight"))?,
                    q_b: src.get(&k("self_attn.q_proj.bias"))?,
                    k_w: src.get(&k("self_attn.k_proj.weight"))?,
                    k_b: src.get(&k("self_attn.k_proj.bias"))?,
                    v_w: src.get(&k("self_attn.v_proj.weight"))?,
                    v_b: src.get(&k("self_attn.v_proj.bias"))?,
                    o_w: src.get(&k("self_attn.o_proj.weight"))?,
                },
                mlp: Mlp {
                    gate: src.get(&k("mlp.gate_proj.weight"))?,
                    up: src.get(&k("mlp.up_proj.weight"))?,
                    down: src.get(&k("mlp.down_proj.weight"))?,
                },
            };
            blk = blk.max(layer.bytes());
            if i >= host_from {
                synaptix_core::device::cuda::set_offload_pinned(true);
                let host = layer.to_device(Device::Cpu);
                synaptix_core::device::cuda::set_offload_pinned(false);
                drop(layer);
                trim();
                layers.push(host?);
            } else {
                layers.push(layer);
            }
        }
        let device = dev;
        let dtype = final_norm.dtype();
        let rope = RopeCache::new(
            cfg.head_dim(),
            rope_capacity.max(1),
            cfg.rope_theta as f32,
            device,
        )
        .map_err(err)?;
        Ok(Self {
            embed,
            layers,
            host_from,
            final_norm,
            lm_head,
            rope,
            cfg: cfg.clone(),
            device,
            dtype,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Сколько слоёв LM лежит на карте (остальные стримятся с хоста).
    pub fn resident_layers(&self) -> usize {
        self.host_from.min(self.layers.len())
    }

    /// Проход по слоям: резидентные как есть, остальные приезжают с хоста,
    /// следующий — на loader-стриме параллельно счёту текущего (только
    /// копирование, схема `acestep::Dit::for_each_layer`).
    fn for_each_layer<F>(&self, mut body: F) -> Result<()>
    where
        F: FnMut(usize, &Layer) -> Result<()>,
    {
        let n = self.layers.len();
        let first = self.resident_layers();
        for (i, l) in self.layers[..first].iter().enumerate() {
            body(i, l)?;
        }
        let ord = match self.device {
            Device::Cuda(o) if first < n => o,
            _ => {
                for (i, l) in self.layers.iter().enumerate().skip(first) {
                    body(i, l)?;
                }
                return Ok(());
            }
        };
        let dev = self.device;
        let ls = synaptix_core::device::cuda::loader_stream(ord).map_err(err)?;
        synaptix_core::device::cuda::set_offload_pinned(true);
        let mut staged = Some(self.layers[first].to_device(dev));
        let mut result = Ok(());
        for idx in first..n {
            let cur = match staged.take() {
                Some(Ok(l)) => l,
                Some(Err(e)) => {
                    result = Err(e);
                    break;
                }
                None => unreachable!(),
            };
            let lsc = ls.clone();
            let (step, next) = std::thread::scope(|sp| {
                let h = (idx + 1 < n).then(|| {
                    let nl = &self.layers[idx + 1];
                    sp.spawn(move || {
                        synaptix_core::device::cuda::set_alloc_stream(Some(lsc.clone()));
                        synaptix_core::device::cuda::set_offload_pinned(true);
                        let r = nl.to_device(dev);
                        let _ = lsc.synchronize();
                        synaptix_core::device::cuda::set_offload_pinned(false);
                        synaptix_core::device::cuda::set_alloc_stream(None);
                        r
                    })
                });
                let step = body(idx, &cur);
                let next = h.map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err(VibeVoiceError::Inference("поток префетча слоя упал".into())))
                });
                (step, next)
            });
            if let Ok(cs) = synaptix_core::device::cuda::default_stream(ord) {
                let _ = cs.synchronize();
            }
            drop(cur);
            if let Err(e) = step {
                result = Err(e);
                break;
            }
            staged = next;
        }
        synaptix_core::device::cuda::set_offload_pinned(false);
        result
    }

    pub fn new_cache(&self, cap: usize) -> Result<KvCache> {
        KvCache::new(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim(),
            cap,
            self.dtype,
            self.device,
        )
    }

    pub fn embed_tokens(&self, ids: &[i64]) -> Result<Tensor> {
        let n = ids.len();
        let idx = Tensor::from_vec(ids.to_vec(), vec![n], self.embed.device()).map_err(err)?;
        self.embed
            .index_select(0, &idx)
            .and_then(|t| t.to_device(self.device))
            .and_then(|t| t.reshape(vec![1usize, n, self.cfg.hidden_size]))
            .map_err(err)
    }

    /// Полные логиты (голова в RAM приезжает на карту на время умножения —
    /// генерации это не нужно, ей хватает [`Self::lm_head_rows`]).
    pub fn lm_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let head = self.lm_head.to_device(self.device).map_err(err)?;
        hidden.linear(&head).map_err(err)
    }

    pub fn lm_head_rows(&self, ids: &[i64]) -> Result<Tensor> {
        let idx = Tensor::from_vec(ids.to_vec(), vec![ids.len()], self.lm_head.device()).map_err(err)?;
        self.lm_head
            .index_select(0, &idx)
            .and_then(|t| t.to_device(self.device))
            .and_then(|t| t.contiguous())
            .map_err(err)
    }

    fn attention(
        &self,
        layer: &Layer,
        layer_idx: usize,
        h: &Tensor,
        cache: &mut KvCache,
        past: usize,
        s: usize,
    ) -> Result<Tensor> {
        let a = &layer.attn;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim();
        let scale = 1.0f32 / (hd as f32).sqrt();

        let q = h
            .linear_bias_residual(&a.q_w, Some(&a.q_b), None)
            .and_then(|t| t.reshape(vec![1usize, s, nh, hd]))
            .and_then(|t| t.permute(vec![0usize, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let k = h
            .linear_bias_residual(&a.k_w, Some(&a.k_b), None)
            .and_then(|t| t.reshape(vec![1usize, s, nkv, hd]))
            .and_then(|t| t.permute(vec![0usize, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;
        let v = h
            .linear_bias_residual(&a.v_w, Some(&a.v_b), None)
            .and_then(|t| t.reshape(vec![1usize, s, nkv, hd]))
            .and_then(|t| t.permute(vec![0usize, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(err)?;

        let q = apply_rope_range(&q, &self.rope, past, s, RopeLayout::Split).map_err(err)?;
        let k = apply_rope_range(&k, &self.rope, past, s, RopeLayout::Split).map_err(err)?;

        cache.k[layer_idx].kv_append_inplace(&k, past).map_err(err)?;
        cache.v[layer_idx].kv_append_inplace(&v, past).map_err(err)?;

        let total = past + s;
        let k_all = cache.k[layer_idx].narrow(2, 0, total).map_err(err)?;
        let v_all = cache.v[layer_idx].narrow(2, 0, total).map_err(err)?;

        let attn = match q.flash_attention(&k_all, &v_all, scale, true) {
            Ok(o) => o,
            Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                let group = nh / nkv;
                let kr = repeat_kv(&k_all, group)?;
                let vr = repeat_kv(&v_all, group)?;
                let bias = if s == 1 {
                    None
                } else {
                    Some(causal_bias(s, total, self.device)?.to_dtype(q.dtype()).map_err(err)?)
                };
                scaled_dot_attention(&q, &kr, &vr, scale, bias.as_ref()).map_err(err)?
            }
            Err(e) => return Err(err(e)),
        };

        attn.permute(vec![0usize, 2, 1, 3])
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![1usize, s, nh * hd]))
            .and_then(|t| t.linear(&a.o_w))
            .map_err(err)
    }

    fn mlp(&self, layer: &Layer, h: &Tensor) -> Result<Tensor> {
        let m = &layer.mlp;
        let gate = h.linear(&m.gate).map_err(err)?;
        let up = h.linear(&m.up).map_err(err)?;
        let act = match gate.silu_and_mul(&up) {
            Ok(t) => t,
            Err(_) => gate.silu().and_then(|g| g.mul(&up)).map_err(err)?,
        };
        act.linear(&m.down).map_err(err)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_pair(
        &self,
        layer: &Layer,
        layer_idx: usize,
        h: &Tensor,
        cache_a: &mut KvCache,
        cache_b: &mut KvCache,
    ) -> Result<Tensor> {
        let a = &layer.attn;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let hd = self.cfg.head_dim();
        let scale = 1.0f32 / (hd as f32).sqrt();

        let proj = |w: &Tensor, b: &Tensor, heads: usize| -> Result<Tensor> {
            h.linear_bias_residual(w, Some(b), None)
                .and_then(|t| t.reshape(vec![2usize, 1, heads, hd]))
                .and_then(|t| t.permute(vec![0usize, 2, 1, 3]))
                .and_then(|t| t.contiguous())
                .map_err(err)
        };
        let q = proj(&a.q_w, &a.q_b, nh)?;
        let k = proj(&a.k_w, &a.k_b, nkv)?;
        let v = proj(&a.v_w, &a.v_b, nkv)?;

        let mut outs: Vec<Tensor> = Vec::with_capacity(2);
        for slot in 0..2usize {
            let cache = if slot == 0 { &mut *cache_a } else { &mut *cache_b };
            let past = cache.len;
            let qi = q.narrow(0, slot, 1).and_then(|t| t.contiguous()).map_err(err)?;
            let ki = k.narrow(0, slot, 1).and_then(|t| t.contiguous()).map_err(err)?;
            let vi = v.narrow(0, slot, 1).and_then(|t| t.contiguous()).map_err(err)?;
            let qi = apply_rope_range(&qi, &self.rope, past, 1, RopeLayout::Split).map_err(err)?;
            let ki = apply_rope_range(&ki, &self.rope, past, 1, RopeLayout::Split).map_err(err)?;
            cache.k[layer_idx].kv_append_inplace(&ki, past).map_err(err)?;
            cache.v[layer_idx].kv_append_inplace(&vi, past).map_err(err)?;
            let total = past + 1;
            let k_all = cache.k[layer_idx].narrow(2, 0, total).map_err(err)?;
            let v_all = cache.v[layer_idx].narrow(2, 0, total).map_err(err)?;
            let attn = match qi.flash_attention(&k_all, &v_all, scale, true) {
                Ok(o) => o,
                Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                    let group = nh / nkv;
                    let kr = repeat_kv(&k_all, group)?;
                    let vr = repeat_kv(&v_all, group)?;
                    scaled_dot_attention(&qi, &kr, &vr, scale, None).map_err(err)?
                }
                Err(e) => return Err(err(e)),
            };
            outs.push(attn);
        }
        let merged = Tensor::cat(&[&outs[0], &outs[1]], 0).map_err(err)?;
        merged
            .permute(vec![0usize, 2, 1, 3])
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![2usize, 1, nh * hd]))
            .and_then(|t| t.linear(&a.o_w))
            .map_err(err)
    }

    pub fn forward_pair(
        &self,
        x_a: &Tensor,
        x_b: &Tensor,
        cache_a: &mut KvCache,
        cache_b: &mut KvCache,
    ) -> Result<(Tensor, Tensor)> {
        if cache_a.len + 1 > cache_a.cap || cache_b.len + 1 > cache_b.cap {
            return Err(VibeVoiceError::Inference("kv cache overflow (pair)".into()));
        }
        let eps = self.cfg.rms_norm_eps;
        let hidden_size = self.cfg.hidden_size;
        let a = x_a.to_dtype(self.dtype).and_then(|t| t.reshape(vec![1usize, 1, hidden_size])).map_err(err)?;
        let b = x_b.to_dtype(self.dtype).and_then(|t| t.reshape(vec![1usize, 1, hidden_size])).map_err(err)?;
        let mut hidden = Tensor::cat(&[&a, &b], 0).map_err(err)?;

        self.for_each_layer(|i, layer| {
            let residual = hidden.clone();
            let h = rms_norm(&hidden, &layer.input_ln, eps).map_err(err)?;
            let mixed = self.attention_pair(layer, i, &h, cache_a, cache_b)?;
            hidden = residual.add(&mixed).map_err(err)?;

            let residual = hidden.clone();
            let h = rms_norm(&hidden, &layer.post_ln, eps).map_err(err)?;
            let m = self.mlp(layer, &h)?;
            hidden = residual.add(&m).map_err(err)?;
            Ok(())
        })?;
        cache_a.len += 1;
        cache_b.len += 1;
        let hidden = rms_norm(&hidden, &self.final_norm, eps).map_err(err)?;
        let ha = hidden.narrow(0, 0, 1).and_then(|t| t.contiguous()).map_err(err)?;
        let hb = hidden.narrow(0, 1, 1).and_then(|t| t.contiguous()).map_err(err)?;
        Ok((ha, hb))
    }

    pub fn rollback(&self, cache: &mut KvCache, n: usize) {
        cache.len = cache.len.saturating_sub(n);
    }

    pub fn forward(&self, inputs_embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let dims = inputs_embeds.dims().to_vec();
        let s = dims[1];
        let past = cache.len;
        if past + s > cache.cap {
            return Err(VibeVoiceError::Inference(format!(
                "kv cache overflow: {past}+{s} > {}",
                cache.cap
            )));
        }
        let eps = self.cfg.rms_norm_eps;
        let mut hidden = inputs_embeds.to_dtype(self.dtype).map_err(err)?;
        self.for_each_layer(|i, layer| {
            let residual = hidden.clone();
            let h = rms_norm(&hidden, &layer.input_ln, eps).map_err(err)?;
            let mixed = self.attention(layer, i, &h, cache, past, s)?;
            hidden = residual.add(&mixed).map_err(err)?;

            let residual = hidden.clone();
            let h = rms_norm(&hidden, &layer.post_ln, eps).map_err(err)?;
            let m = self.mlp(layer, &h)?;
            hidden = residual.add(&m).map_err(err)?;
            Ok(())
        })?;
        cache.len = past + s;
        rms_norm(&hidden, &self.final_norm, eps).map_err(err)
    }
}
