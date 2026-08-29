use serde::Deserialize;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::{DecoderModel, ModelError};
use synaptix_llm_common::weights::{QLinear, WeightSource};
use synaptix_ops::attention::softmax::scaled_dot_attention;
use synaptix_ops::norm::rms_norm::rms_norm;
use synaptix_ops::pos::rope::{apply_rope_range, RopeLayout};
use synaptix_ops::pos::rope_cache::RopeCache;

pub const DFLASH_COMPONENT: &str = "dflash";
/// Имена тензоров внутри компонента уже несут префикс компонента —
/// `syn-pack --component dflash:<dir>:dflash` пишет их как `dflash.<orig>`.
pub const DFLASH_PREFIX: &str = "dflash";

const RING_SLACK: usize = 2048;

#[derive(Debug, Clone, Deserialize)]
struct RopeParams {
    #[serde(default = "default_theta")]
    rope_theta: f32,
}

fn default_theta() -> f32 {
    500_000.0
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct RawConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    sliding_window: usize,
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    max_position_embeddings: usize,
    layer_types: Vec<String>,
    rope_parameters: RopeParams,
}

impl Default for RawConfig {
    fn default() -> Self {
        Self {
            hidden_size: 6656,
            intermediate_size: 19_968,
            num_hidden_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1.0e-5,
            sliding_window: 2048,
            block_size: 16,
            mask_token_id: 201_818,
            target_layer_ids: vec![1, 13, 25, 37, 49],
            max_position_embeddings: 131_072,
            layer_types: Vec::new(),
            rope_parameters: RopeParams { rope_theta: default_theta() },
        }
    }
}

#[derive(Debug, Clone)]
pub struct DFlashConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
}

impl DFlashConfig {
    pub fn from_hf_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        let raw: RawConfig = serde_json::from_slice(bytes)
            .map_err(|e| ModelError::Load(format!("dflash config: {e}")))?;
        if !raw.layer_types.is_empty()
            && raw.layer_types.iter().any(|t| t != "sliding_attention")
        {
            return Err(ModelError::Load(
                "dflash: поддержаны только sliding_attention-слои".into(),
            ));
        }
        if raw.num_attention_heads % raw.num_key_value_heads != 0 {
            return Err(ModelError::Load("dflash: GQA heads % kv_heads != 0".into()));
        }
        if raw.block_size < 2 {
            return Err(ModelError::Load("dflash: block_size < 2".into()));
        }
        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rms_norm_eps: raw.rms_norm_eps,
            sliding_window: raw.sliding_window,
            block_size: raw.block_size,
            mask_token_id: raw.mask_token_id,
            target_layer_ids: raw.target_layer_ids,
            max_position_embeddings: raw.max_position_embeddings,
            rope_theta: raw.rope_parameters.rope_theta,
        })
    }

    /// Сколько токенов драфтер предлагает за один forward: блок минус anchor.
    pub fn draft_len(&self) -> usize {
        self.block_size - 1
    }
}

struct Layer {
    input_norm: Tensor,
    post_attn_norm: Tensor,
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

pub struct DFlashCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    /// Позиций контекста в буфере (без диффузионного окна).
    len: usize,
    /// Абсолютная позиция первого слота буфера (ring).
    start: usize,
    cap: usize,
}

impl DFlashCache {
    pub fn reset(&mut self) {
        self.len = 0;
        self.start = 0;
    }

    /// Откатить контекст к абсолютной позиции `pos`: точка возврата префикс-KV
    /// стоит на конце промпта, а декод успел дописать в кэш дальше.
    /// `false` — позиция уже вне окна (буфер прокрутился), контекст драфтера
    /// придётся набирать заново.
    pub fn truncate_to(&mut self, pos: usize) -> bool {
        if pos < self.start || pos > self.start + self.len {
            return false;
        }
        self.len = pos - self.start;
        true
    }

    pub fn context_len(&self) -> usize {
        self.len
    }

    pub fn context_start(&self) -> usize {
        self.start
    }

    /// Переселить буфер драфтера в host-RAM (см.
    /// `KvCache::park_to_host`): на время вложенной генерации кэш
    /// префикс-KV не должен держать VRAM.
    pub fn park_to_host(&mut self) -> Result<usize, ModelError> {
        let mut moved = 0;
        for t in self.k.iter_mut().chain(self.v.iter_mut()) {
            if t.device() == Device::Cpu {
                continue;
            }
            moved += t.dtype().bytes_for_numel(t.numel());
            *t = t
                .to_device(Device::Cpu)
                .map_err(|e| ModelError::Forward(e.to_string()))?;
        }
        Ok(moved)
    }

    pub fn unpark_to(&mut self, device: Device) -> Result<usize, ModelError> {
        let mut moved = 0;
        for t in self.k.iter_mut().chain(self.v.iter_mut()) {
            if t.device() == device {
                continue;
            }
            moved += t.dtype().bytes_for_numel(t.numel());
            *t = t
                .to_device(device)
                .map_err(|e| ModelError::Forward(e.to_string()))?;
        }
        Ok(moved)
    }

    /// Сколько VRAM держит буфер драфтера прямо сейчас.
    pub fn device_bytes(&self) -> usize {
        self.k
            .iter()
            .chain(self.v.iter())
            .filter(|t| t.device() != Device::Cpu)
            .map(|t| t.dtype().bytes_for_numel(t.numel()))
            .sum()
    }
}

pub struct DFlashModule {
    pub config: DFlashConfig,
    layers: Vec<Layer>,
    enc_fc: QLinear,
    enc_norm: Tensor,
    final_norm: Tensor,
    rope: RopeCache,
    device: Device,
    compute: DType,
}

/// Веса драфтера лежат отдельным компонентом бандла (`tensors:dflash`) — со
/// своими, не префиксованными именами (`encoder.fc.weight`, `layers.N.*`).
pub struct BundleDFlashWeights {
    loader: synaptix_io::weights::syn_bundle::SynBundleLoader,
}

impl BundleDFlashWeights {
    pub fn open(path: impl AsRef<std::path::Path>, device: Device) -> Result<Self, ModelError> {
        let loader = synaptix_io::weights::syn_bundle::SynBundleLoader::open(path)
            .map_err(|e| ModelError::Load(e.to_string()))?
            .with_component(DFLASH_COMPONENT)
            .with_device(device);
        Ok(Self { loader })
    }
}

impl WeightSource for BundleDFlashWeights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        use synaptix_io::weights::WeightLoader;
        self.loader
            .load_to(key, device, dtype)
            .map_err(|e| ModelError::Load(format!("dflash {key}: {e}")))
    }

    fn contains(&self, key: &str) -> bool {
        use synaptix_io::weights::WeightLoader;
        self.loader.names().iter().any(|n| *n == key)
    }
}

pub fn present(weights: &dyn WeightSource) -> bool {
    weights.contains(&format!("{DFLASH_PREFIX}.encoder.fc.weight"))
        && weights.contains(&format!("{DFLASH_PREFIX}.layers.0.self_attn.q_proj.weight"))
        && weights.contains(&format!("{DFLASH_PREFIX}.norm.weight"))
}

impl DFlashModule {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        config: DFlashConfig,
        weights: &dyn WeightSource,
        device: Device,
        compute: DType,
        attn_w: DType,
        mlp_w: DType,
    ) -> Result<Self, ModelError> {
        let qlin = |key: &str, wq: DType| -> Result<QLinear, ModelError> {
            let quantized = wq.is_quantized() && matches!(device, Device::Cuda(_));
            let load_dt = if quantized { DType::F16 } else { compute };
            let w = weights.tensor(key, device, load_dt)?;
            QLinear::build(w, if quantized { wq } else { compute }, compute)
        };
        let norm = |key: &str| -> Result<Tensor, ModelError> {
            weights.tensor(key, device, compute)
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("{DFLASH_PREFIX}.layers.{i}");
            layers.push(Layer {
                input_norm: norm(&format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: norm(&format!("{p}.post_attention_layernorm.weight"))?,
                q_proj: qlin(&format!("{p}.self_attn.q_proj.weight"), attn_w)?,
                k_proj: qlin(&format!("{p}.self_attn.k_proj.weight"), attn_w)?,
                v_proj: qlin(&format!("{p}.self_attn.v_proj.weight"), attn_w)?,
                o_proj: qlin(&format!("{p}.self_attn.o_proj.weight"), attn_w)?,
                q_norm: norm(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: norm(&format!("{p}.self_attn.k_norm.weight"))?,
                gate_proj: qlin(&format!("{p}.mlp.gate_proj.weight"), mlp_w)?,
                up_proj: qlin(&format!("{p}.mlp.up_proj.weight"), mlp_w)?,
                down_proj: qlin(&format!("{p}.mlp.down_proj.weight"), mlp_w)?,
            });
        }

        let rope_cap = config.max_position_embeddings.min(131_072);
        let rope = RopeCache::new(config.head_dim, rope_cap, config.rope_theta, device)
            .map_err(|e| ModelError::Build(format!("dflash rope: {e}")))?;

        Ok(Self {
            enc_fc: qlin(&format!("{DFLASH_PREFIX}.encoder.fc.weight"), attn_w)?,
            enc_norm: norm(&format!("{DFLASH_PREFIX}.encoder.output_norm_enc.weight"))?,
            final_norm: norm(&format!("{DFLASH_PREFIX}.norm.weight"))?,
            config,
            layers,
            rope,
            device,
            compute,
        })
    }

    pub fn make_cache(&self) -> Result<DFlashCache, ModelError> {
        let c = &self.config;
        let cap = c.sliding_window + RING_SLACK + c.block_size;
        let nkv = c.num_key_value_heads;
        let hd = c.head_dim;
        let mut k = Vec::with_capacity(c.num_hidden_layers);
        let mut v = Vec::with_capacity(c.num_hidden_layers);
        for _ in 0..c.num_hidden_layers {
            k.push(
                Tensor::zeros(vec![1, nkv, cap, hd], self.compute, self.device)
                    .map_err(|e| ModelError::Build(e.to_string()))?,
            );
            v.push(
                Tensor::zeros(vec![1, nkv, cap, hd], self.compute, self.device)
                    .map_err(|e| ModelError::Build(e.to_string()))?,
            );
        }
        Ok(DFlashCache { k, v, len: 0, start: 0, cap })
    }

    /// Освободить место под `need` новых контекстных позиций + диффузионное
    /// окно: при переполнении оставляем хвост длиной `sliding_window`.
    fn roll_if_needed(&self, cache: &mut DFlashCache, need: usize) -> Result<(), ModelError> {
        let want = cache.len + need + self.config.block_size;
        if want <= cache.cap {
            return Ok(());
        }
        let keep = self.config.sliding_window.min(cache.len);
        let src = cache.len - keep;
        for l in 0..self.layers.len() {
            if keep > 0 {
                let tk = cache.k[l]
                    .narrow(2, src, keep)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| ModelError::Forward(e.to_string()))?;
                let tv = cache.v[l]
                    .narrow(2, src, keep)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| ModelError::Forward(e.to_string()))?;
                cache.k[l]
                    .kv_append_inplace(&tk, 0)
                    .map_err(|e| ModelError::Forward(e.to_string()))?;
                cache.v[l]
                    .kv_append_inplace(&tv, 0)
                    .map_err(|e| ModelError::Forward(e.to_string()))?;
            }
        }
        cache.start += src;
        cache.len = keep;
        Ok(())
    }

    fn heads(&self, x: &Tensor, s: usize, n: usize) -> Result<Tensor, ModelError> {
        x.reshape(vec![1, s, n, self.config.head_dim])
            .and_then(|t| t.permute(vec![0, 2, 1, 3]))
            .and_then(|t| t.contiguous())
            .map_err(|e| ModelError::Forward(e.to_string()))
    }

    /// Один draft-проход. `ctx_hidden` — выходы target-слоёв `target_layer_ids`
    /// для `m` принятых токенов, `ctx_pos` — абсолютная позиция первого из них,
    /// `anchor` — последний токен target'а (он же позиция `ctx_pos + m`).
    /// Возвращает логиты кандидатов `[draft_len, vocab]`.
    pub fn draft_logits(
        &self,
        target: &DecoderModel,
        cache: &mut DFlashCache,
        ctx_hidden: &[Tensor],
        ctx_pos: usize,
        anchor: u32,
    ) -> Result<Tensor, ModelError> {
        let c = &self.config;
        let e = |r: Result<Tensor, SynaptixError>| r.map_err(|x| ModelError::Forward(x.to_string()));
        if ctx_hidden.len() != c.target_layer_ids.len() {
            return Err(ModelError::Shape(format!(
                "dflash: ожидалось {} tap-слоёв, получено {}",
                c.target_layer_ids.len(),
                ctx_hidden.len()
            )));
        }
        let m = ctx_hidden[0].dims()[1];
        if cache.start + cache.len != ctx_pos {
            return Err(ModelError::Shape(format!(
                "dflash: разрыв контекста — кэш до {}, новый блок с {ctx_pos}",
                cache.start + cache.len
            )));
        }
        // Блок контекста длиннее окна (первый ход длинного промпта: сюда
        // приходят tap-hidden'ы ВСЕГО промпта) режем по хвосту. Внимание
        // драфтера — band |i-j| < sliding_window, а контекстные K/V каждый слой
        // считает из выхода энкодера, а не из предыдущего слоя, поэтому
        // receptive field по слоям не растёт: всё, что дальше окна, на логиты
        // не влияет. Без обрезки блок физически не влезает в ring-кэш
        // (cap = sliding_window + RING_SLACK + block_size) и `kv_append` падает
        // с «seq_pos + t_new > max_seq».
        let (trimmed, ctx_hidden, ctx_pos, m) = if m > c.sliding_window {
            let keep = c.sliding_window;
            let cut: Vec<Tensor> = ctx_hidden
                .iter()
                .map(|t| e(t.narrow(1, m - keep, keep).and_then(|x| x.contiguous())))
                .collect::<Result<Vec<_>, _>>()?;
            // Отброшенный префикс старше окна — кэш начинается с нового хвоста.
            cache.start = ctx_pos + (m - keep);
            cache.len = 0;
            (Some(cut), ctx_hidden, ctx_pos + (m - keep), keep)
        } else {
            (None, ctx_hidden, ctx_pos, m)
        };
        let ctx_hidden: &[Tensor] = trimmed.as_deref().unwrap_or(ctx_hidden);
        self.roll_if_needed(cache, m)?;

        let refs: Vec<&Tensor> = ctx_hidden.iter().collect();
        let ctx = e(Tensor::cat(&refs, 2))?;
        let ctx = e(ctx.to_dtype(self.compute))?;
        let ctx = self.enc_fc.forward(&ctx)?;
        let ctx = e(rms_norm(&ctx, &self.enc_norm, c.rms_norm_eps))?;

        let bs = c.block_size;
        let mut ids = Vec::with_capacity(bs);
        ids.push(anchor);
        ids.resize(bs, c.mask_token_id);
        let ids_t = e(Tensor::from_vec(ids, vec![bs], self.device))?;
        // Драфтер эмбеддит БЕЗ RMS-нормы эмбеддинга (в отличие от target'а).
        let mut hidden = e(target
            .embed_rows(&ids_t)?
            .reshape(vec![1, bs, c.hidden_size]))?;
        hidden = e(hidden.to_dtype(self.compute))?;

        let (nh, nkv) = (c.num_attention_heads, c.num_key_value_heads);
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let block_pos = ctx_pos + m;

        for (li, layer) in self.layers.iter().enumerate() {
            let h = e(rms_norm(&hidden, &layer.input_norm, c.rms_norm_eps))?;

            let q = self.heads(&layer.q_proj.forward(&h)?, bs, nh)?;
            let q = e(rms_norm(&q, &layer.q_norm, c.rms_norm_eps))?;
            let q = e(apply_rope_range(&q, &self.rope, block_pos, bs, RopeLayout::Split))?;

            if m > 0 {
                let kc = self.heads(&layer.k_proj.forward(&ctx)?, m, nkv)?;
                let kc = e(rms_norm(&kc, &layer.k_norm, c.rms_norm_eps))?;
                let kc = e(apply_rope_range(&kc, &self.rope, ctx_pos, m, RopeLayout::Split))?;
                let vc = self.heads(&layer.v_proj.forward(&ctx)?, m, nkv)?;
                cache.k[li]
                    .kv_append_inplace(&kc, cache.len)
                    .map_err(|x| ModelError::Forward(x.to_string()))?;
                cache.v[li]
                    .kv_append_inplace(&vc, cache.len)
                    .map_err(|x| ModelError::Forward(x.to_string()))?;
            }

            let kb = self.heads(&layer.k_proj.forward(&h)?, bs, nkv)?;
            let kb = e(rms_norm(&kb, &layer.k_norm, c.rms_norm_eps))?;
            let kb = e(apply_rope_range(&kb, &self.rope, block_pos, bs, RopeLayout::Split))?;
            let vb = self.heads(&layer.v_proj.forward(&h)?, bs, nkv)?;
            let ctx_end = cache.len + m;
            cache.k[li]
                .kv_append_inplace(&kb, ctx_end)
                .map_err(|x| ModelError::Forward(x.to_string()))?;
            cache.v[li]
                .kv_append_inplace(&vb, ctx_end)
                .map_err(|x| ModelError::Forward(x.to_string()))?;

            let total = ctx_end + bs;
            let k_all = e(cache.k[li].narrow(2, 0, total))?;
            let v_all = e(cache.v[li].narrow(2, 0, total))?;
            // Диффузионное окно смотрит двунаправленно (внутри блока) и на
            // контекст в пределах sliding-окна: band |i-j| < window, без causal.
            let win = (c.sliding_window - 1) as i32;
            let attn = match q.flash_attention_window(&k_all, &v_all, scale, win, false) {
                Ok(a) => a,
                Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {
                    let group = nh / nkv;
                    let kr = e(repeat_kv(&k_all, group))?;
                    let vr = e(repeat_kv(&v_all, group))?;
                    let mask = band_mask(bs, total, c.sliding_window, self.device, self.compute)?;
                    e(scaled_dot_attention(&q, &kr, &vr, scale, Some(&mask)))?
                }
                Err(x) => return Err(ModelError::Forward(x.to_string())),
            };
            let attn = e(attn
                .permute(vec![0, 2, 1, 3])
                .and_then(|t| t.contiguous())
                .and_then(|t| t.reshape(vec![1, bs, nh * c.head_dim])))?;
            let attn = layer.o_proj.forward(&attn)?;
            hidden = e(hidden.add(&attn))?;

            let h2 = e(rms_norm(&hidden, &layer.post_attn_norm, c.rms_norm_eps))?;
            let gate = e(layer.gate_proj.forward(&h2)?.silu())?;
            let up = layer.up_proj.forward(&h2)?;
            let mlp = layer.down_proj.forward(&e(gate.mul(&up))?)?;
            hidden = e(hidden.add(&mlp))?;
        }
        cache.len += m;

        let hidden = e(rms_norm(&hidden, &self.final_norm, c.rms_norm_eps))?;
        // Позиция 0 блока — anchor (уже известен), кандидаты идут с позиции 1.
        let cand = e(hidden
            .narrow(1, 1, c.draft_len())
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![c.draft_len(), c.hidden_size])))?;
        target.lm_head_forward(&cand)
    }
}

fn repeat_kv(x: &Tensor, group: usize) -> Result<Tensor, SynaptixError> {
    if group == 1 {
        return Ok(x.clone());
    }
    let d = x.dims();
    let (b, nkv, s, hd) = (d[0], d[1], d[2], d[3]);
    let reps = Tensor::zeros(vec![b, nkv, group, s, hd], x.dtype(), x.device())?;
    x.unsqueeze(2)?
        .broadcast_add(&reps)?
        .reshape(vec![b, nkv * group, s, hd])
}

fn band_mask(
    tq: usize,
    tkv: usize,
    window: usize,
    device: Device,
    dtype: DType,
) -> Result<Tensor, ModelError> {
    let base = tkv - tq;
    let mut data = vec![0f32; tq * tkv];
    for i in 0..tq {
        let qi = (base + i) as i64;
        for j in 0..tkv {
            if (qi - j as i64).unsigned_abs() as usize >= window {
                data[i * tkv + j] = -1.0e4;
            }
        }
    }
    let m = Tensor::from_vec(data, vec![tq, tkv], device)
        .map_err(|e| ModelError::Forward(e.to_string()))?;
    m.to_dtype(dtype).map_err(|e| ModelError::Forward(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "architectures": ["MuseGlimmerAssistantModel"],
        "block_size": 16,
        "head_dim": 128,
        "hidden_size": 6656,
        "intermediate_size": 19968,
        "mask_token_id": 201818,
        "max_position_embeddings": 131072,
        "num_attention_heads": 32,
        "num_hidden_layers": 5,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-05,
        "sliding_window": 2048,
        "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
        "target_layer_ids": [1, 13, 25, 37, 49],
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}
    }"#;

    #[test]
    fn parses_config() {
        let c = DFlashConfig::from_hf_bytes(SAMPLE.as_bytes()).unwrap();
        assert_eq!(c.num_hidden_layers, 5);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.block_size, 16);
        assert_eq!(c.draft_len(), 15);
        assert_eq!(c.mask_token_id, 201_818);
        assert_eq!(c.target_layer_ids, vec![1, 13, 25, 37, 49]);
        assert_eq!(c.rope_theta, 500_000.0);
    }

    #[test]
    fn band_mask_is_bidirectional() {
        let m = band_mask(4, 10, 3, Device::Cpu, DType::F32).unwrap();
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // q-строка 0 = абсолютная позиция 6 (base = 10-4)
        assert_eq!(v[6], 0.0, "своя позиция видна");
        assert_eq!(v[8], 0.0, "позиция впереди в пределах окна видна (bidirectional)");
        assert_eq!(v[3], -1.0e4, "за пределами окна — маска");
        assert_eq!(v[4], 0.0, "|6-4| = 2 < 3 — видно");
    }
}
