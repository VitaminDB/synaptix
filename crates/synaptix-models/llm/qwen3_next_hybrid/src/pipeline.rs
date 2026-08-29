use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::HybridConfig;
use crate::mrope;
use synaptix_llm_common::model::RopePositions;
use crate::loader::HybridWeights;
use crate::model::{DecoderModel, ModelError};

pub use synaptix_llm_common::generate::{GenerationConfig, GenerationStats, StreamSink};

/// M-RoPE на медиа-промпте включён по умолчанию; `SYN_HYBRID_MROPE=0` —
/// выключатель для A/B-сравнения с 1D-позициями.
fn mrope_enabled() -> bool {
    std::env::var("SYN_HYBRID_MROPE").map(|v| v != "0").unwrap_or(true)
}

/// Дефолтный chunk префилла гибрида. Ограничивает пик VRAM активаций
/// (single-shot M>=512 не влезает в 24 ГБ поверх ~18 ГБ весов) при скорости
/// ~1.6k ток/с (M=256 уже насыщает NVFP4-GEMM'ы).
pub const DEFAULT_PREFILL_CHUNK: usize = 256;

/// Границы чанков ОБЯЗАНЫ быть кратны CS=64 GDN-скана: не-кратный разрез
/// mid-stream оставляет состояние частичного внутреннего чанка, расходящееся
/// с single-shot (проверено prefill_chunk_divergence: 64/128/256 bit-exact,
/// 47/73/100 — max_abs_diff ~2.5-3.0). 0 → дефолт; иное — округление к 64.
/// Внутренний чанк GDN-скана (`CS` в `model.rs`): границы префилла и точки
/// возврата префикс-KV обязаны быть кратны ему.
pub const GDN_CHUNK: usize = 64;

fn effective_prefill_chunk(requested: usize) -> usize {
    // `SYN_PREFILL_CHUNK` перебивает и запрос вызывающего — knob для замеров
    // «пик VRAM / скорость» без пересборки хоста.
    static ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("SYN_PREFILL_CHUNK").ok().and_then(|v| v.trim().parse().ok())
    });
    let c = match (env, requested) {
        (Some(e), _) if e > 0 => e,
        (_, 0) => DEFAULT_PREFILL_CHUNK,
        (_, r) => r,
    };
    (c.max(64) / 64) * 64
}

pub struct HybridPipeline {
    pub model: DecoderModel,
    pub mtp: Option<synaptix_llm_common::mtp::MtpModule>,
    pub vision: Option<synaptix_vlm_qwen3::VisionTower>,
    pub tokenizer: HfTokenizer,
    pub config: HybridConfig,
    pub add_bos: bool,
}

/// Разметка видео в промпте: сколько групп кадров, сколько токенов на
/// группу и таймкод каждой.
#[derive(Debug, Clone)]
pub struct VideoPromptInfo {
    pub groups: usize,
    pub tokens_per_group: usize,
    pub timestamps: Vec<f32>,
    /// Merged-сетка `(h, w)` одной группы кадров — для M-RoPE.
    pub grid_hw: (usize, usize),
}

/// Вложения одной модальности для [`HybridPipeline::generate_with_media`]:
/// id токена-заполнителя, эмбеддинги всех блоков подряд (в порядке
/// появления в промпте) и merged-сетка каждого блока. Пустой `grids` —
/// сетки неизвестны, промпт идёт по 1D-позициям без M-RoPE.
pub struct MediaInput {
    pub pad: u32,
    pub embeds: synaptix_core::tensor::Tensor,
    pub grids: Vec<(usize, usize)>,
}

impl VideoPromptInfo {
    /// Всего vision-токенов видео.
    pub fn tokens(&self) -> usize {
        self.groups * self.tokens_per_group
    }

    /// Блок промпта как у HF-процессора Qwen3-VL: на каждую группу кадров
    /// `<{t:.1} seconds><|vision_start|><|video_pad|>…<|vision_end|>`.
    /// Таймкод — обычный текст, так модель видит временну́ю шкалу и без
    /// M-RoPE по оси времени.
    pub fn prompt_block(&self) -> String {
        let mut s = String::new();
        for g in 0..self.groups {
            let ts = self.timestamps.get(g).copied().unwrap_or(0.0);
            s.push_str(&format!("<{ts:.1} seconds><|vision_start|>"));
            s.push_str(&"<|video_pad|>".repeat(self.tokens_per_group));
            s.push_str("<|vision_end|>");
        }
        s
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MtpStats {
    pub steps: usize,
    pub drafted: usize,
    pub accepted: usize,
}

impl MtpStats {
    pub fn acceptance(&self) -> f32 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f32 / self.drafted as f32
        }
    }
}

/// Трасса VRAM по чанкам префилла (`SYN_TRACE_PREFILL_MEM=1`). Показывает, что
/// именно растёт на длинном промпте: живые аллокации (утечка/кэш) или
/// зарезервированное пулом (churn скретчей). Без переменной — ноль накладных.
fn trace_prefill_mem(device: Device, off: usize, total: usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("SYN_TRACE_PREFILL_MEM").is_ok()) {
        return;
    }
    let Device::Cuda(ord) = device else { return };
    let mb = |b: u64| b / (1024 * 1024);
    let (free, _) = synaptix_core::device::cuda::mem_info(ord).unwrap_or((0, 0));
    let (rsv, used) = synaptix_core::memory::cuda_pool::cuda_mempool_stats(ord).unwrap_or((0, 0));
    let (arsv, aused) = synaptix_core::device::cuda::activations_pool_stats(ord).unwrap_or((0, 0));
    let (wrsv, wused) = synaptix_core::device::cuda::weights_pool_stats(ord).unwrap_or((0, 0));
    eprintln!(
        "[PREFILL_MEM] {off}/{total} ток: free={} MB, default rsv={}/{} MB, активации rsv={}/{} MB, staging rsv={}/{} MB, наш учёт={:.0} MB",
        (free / (1024 * 1024)) as u64,
        mb(rsv),
        mb(used),
        mb(arsv),
        mb(aused),
        mb(wrsv),
        mb(wused),
        synaptix_core::memory::cuda_pool::cuda_allocated_mb()
    );
    if std::env::var("SYN_TRACE_PREFILL_MEM").as_deref() == Ok("trim") {
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(ord);
        let (free2, _) = synaptix_core::device::cuda::mem_info(ord).unwrap_or((0, 0));
        let (rsv2, used2) = synaptix_core::memory::cuda_pool::cuda_mempool_stats(ord).unwrap_or((0, 0));
        eprintln!(
            "[PREFILL_MEM]   после sync+trim: free={} MB, пул rsv={} MB used={} MB",
            (free2 / (1024 * 1024)) as u64,
            mb(rsv2),
            mb(used2)
        );
    }
}

impl HybridPipeline {
    pub fn load(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let weights =
            HybridWeights::load(path, device, dtype).map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tokenizer = HfTokenizer::from_bytes(&weights.tokenizer_json)
            .map_err(|e| PipelineError::Load(format!("tokenizer: {e}")))?;
        let cap = config.max_position_embeddings.min(4096);
        let dcfg = config.to_decoder_config();
        let model = DecoderModel::build_auto(&dcfg, &weights, device, dtype, dtype, dtype, dtype, dtype, cap)
            .map_err(|e| PipelineError::Model(e.to_string()))?;
        Ok(Self { model, mtp: None, vision: None, tokenizer, config, add_bos: false })
    }

    pub fn load_with_precision(
        path: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let weights = HybridWeights::load(path, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tokenizer = HfTokenizer::from_bytes(&weights.tokenizer_json)
            .map_err(|e| PipelineError::Load(format!("tokenizer: {e}")))?;
        let cap = max_seq.unwrap_or_else(|| config.max_position_embeddings.min(4096));
        let dcfg = config.to_decoder_config();
        let model = DecoderModel::build_auto(
            &dcfg, &weights, device, precision.compute, precision.attn_w, precision.mlp_w, precision.lm_head, precision.embed, cap,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?
        // БАГ-ФИКС: пробросить kv-dtype (--kv-dtype mxfp8) в модель. Без этого
        // model.kv_dtype оставался compute(F16) → MXFP8-KV игнорировался, KV
        // аллоцировался F16 и decode шёл по f16-flash (qwen3-pipeline это делал).
        .with_kv_cache_dtype(precision.kv);
        Ok(Self { model, mtp: None, vision: None, tokenizer, config, add_bos: false })
    }

    pub fn load_with_precision_mtp(
        path: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
        enable_mtp: bool,
    ) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let path = path.as_ref();
        let mut me = Self::load_with_precision(path, device, precision, max_seq)?;
        if !enable_mtp || me.config.mtp_num_hidden_layers == 0 {
            return Ok(me);
        }
        if precision.compute != DType::F16 {
            eprintln!(
                "[hybrid] MTP пропущен: спекулятивный путь требует compute=F16, получено {:?}",
                precision.compute
            );
            return Ok(me);
        }
        let weights = HybridWeights::load(path, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        if !synaptix_llm_common::mtp::present(&weights) {
            return Ok(me);
        }
        let cap = max_seq.unwrap_or_else(|| me.config.max_position_embeddings.min(4096));
        let dcfg = me.config.to_decoder_config();
        let module = synaptix_llm_common::mtp::MtpModule::build(
            &dcfg,
            me.config.mtp_num_hidden_layers,
            &weights,
            device,
            precision.compute,
            precision.attn_w,
            precision.mlp_w,
            precision.lm_head,
            precision.embed,
            cap,
        )
        .map_err(|e| PipelineError::Model(format!("mtp: {e}")))?;
        me.mtp = Some(module);
        Ok(me)
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn graph_decode_supported(&self) -> bool {
        matches!(self.model.device, Device::Cuda(_))
            && self.model.dtype == DType::F16
            && self.model.kv_dtype != DType::MXFP8
            && !self.model.has_mxfp8_head_or_embed()
    }


    fn build_mtp_graph(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
    ) -> Result<MtpGraph, PipelineError> {
        use synaptix_core::grad::no_grad;
        use synaptix_infer::graph_capture::GraphCapturer;
        use synaptix_infer::InferError;

        let ord = match self.model.device {
            Device::Cuda(o) => o,
            _ => return Err(PipelineError::Forward("MTP-граф требует CUDA".into())),
        };
        let mut state = self
            .model
            .make_prefill_state(2)
            .map_err(|e| PipelineError::Forward(format!("prefill state: {e}")))?;
        self.model
            .sync_decode_host_state(kv)
            .map_err(|e| PipelineError::Forward(format!("mtp graph sync host: {e}")))?;
        self.model
            .sync_decode_dev_state(kv)
            .map_err(|e| PipelineError::Forward(format!("mtp graph sync dev: {e}")))?;
        let seq0 = kv.seq_len;
        let snap = kv
            .snapshot_linear()
            .map_err(|e| PipelineError::Forward(format!("mtp graph snapshot: {e}")))?;

        state
            .update(&[0u32, 0u32], seq0 as u32)
            .map_err(|e| PipelineError::Forward(format!("mtp graph warmup update: {e}")))?;
        let stream = synaptix_core::device::cuda::default_stream(ord)
            .map_err(|e| PipelineError::Forward(format!("stream: {e}")))?;
        let mut capturer = GraphCapturer::new(3);
        let graph = {
            let model = &self.model;
            let state_ref = &mut state;
            let kv_ref = &mut *kv;
            no_grad(|| {
                capturer.capture_with(&stream, |_s| {
                    model
                        .forward_prefill_dev(state_ref, kv_ref)
                        .map_err(|e| InferError::Other(e.to_string()))
                })
            })
        }
        .map_err(|e| PipelineError::Forward(format!("mtp graph capture: {e}")))?;
        let _ = graph.upload();

        kv.restore_linear(&snap)
            .map_err(|e| PipelineError::Forward(format!("mtp graph restore: {e}")))?;
        kv.seq_len = seq0;
        Ok(MtpGraph {
            state,
            graph,
            stream,
            hidden_size: self.config.hidden_size,
        })
    }


    pub fn load_vision(
        &mut self,
        path: impl AsRef<Path>,
        dtype: DType,
    ) -> Result<bool, PipelineError> {
        let path = path.as_ref();
        if !synaptix_vlm_qwen3::bundle_has_vision(path) {
            return Ok(false);
        }
        let tower = synaptix_vlm_qwen3::load_from_bundle(path, self.model.device, dtype)
            .map_err(|e| PipelineError::Load(format!("vision: {e}")))?;
        self.vision = Some(tower);
        Ok(true)
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Есть ли в бандле компонент vision-башни — без её загрузки.
    pub fn bundle_has_vision(path: impl AsRef<Path>) -> bool {
        synaptix_vlm_qwen3::bundle_has_vision(path)
    }

    /// Выгружает башню и возвращает её память пулу устройства.
    pub fn release_vision(&mut self) {
        if self.vision.take().is_some() {
            if let Device::Cuda(o) = self.model.device {
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
            }
        }
    }

    /// Лимиты препроцессинга под потолок vision-токенов: один токен после
    /// merge накрывает `size_factor²` пикселей, так что `max_tokens`
    /// пересчитывается в `max_pixels` для `smart_resize`. `None`/0 — дефолт.
    pub fn image_limits(
        &self,
        max_tokens: Option<usize>,
    ) -> Result<synaptix_vlm_qwen3::PreprocessLimits, PipelineError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let mut limits = synaptix_vlm_qwen3::PreprocessLimits::default();
        if let Some(n) = max_tokens.filter(|n| *n > 0) {
            let f = tower.config.size_factor();
            limits.max_pixels = limits.max_pixels.min(n * f * f);
            limits.min_pixels = limits.min_pixels.min(limits.max_pixels);
        }
        Ok(limits)
    }

    /// Как [`Self::encode_image`], но с потолком на число vision-токенов
    /// (см. [`Self::image_limits`]).
    pub fn encode_image_limited(
        &self,
        path: impl AsRef<Path>,
        max_tokens: Option<usize>,
    ) -> Result<synaptix_core::tensor::Tensor, PipelineError> {
        let limits = self.image_limits(max_tokens)?;
        self.encode_image(path, limits)
    }

    pub fn encode_image(
        &self,
        path: impl AsRef<Path>,
        limits: synaptix_vlm_qwen3::PreprocessLimits,
    ) -> Result<synaptix_core::tensor::Tensor, PipelineError> {
        self.encode_image_with_grid(path, limits).map(|(t, _)| t)
    }

    /// Как [`Self::encode_image`], плюс merged-сетка `(h, w)` картинки —
    /// нужна M-RoPE, чтобы разложить токены блока по строкам и столбцам.
    pub fn encode_image_with_grid(
        &self,
        path: impl AsRef<Path>,
        limits: synaptix_vlm_qwen3::PreprocessLimits,
    ) -> Result<(synaptix_core::tensor::Tensor, (usize, usize)), PipelineError> {
        use synaptix_core::grad::no_grad;
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let prepared = synaptix_vlm_qwen3::prepare_image(
            path,
            &tower.config,
            limits,
            self.model.device,
        )
        .map_err(|e| PipelineError::Load(format!("image: {e}")))?;
        let m = tower.config.spatial_merge_size.max(1);
        let grid_hw = (prepared.grid.h / m, prepared.grid.w / m);
        let feats = no_grad(|| tower.forward(&prepared.patches, prepared.grid))
            .map_err(|e| PipelineError::Forward(format!("vision forward: {e}")))?;
        Ok((feats, grid_hw))
    }

    /// Как [`Self::encode_image_limited`], плюс merged-сетка картинки.
    pub fn encode_image_limited_with_grid(
        &self,
        path: impl AsRef<Path>,
        max_tokens: Option<usize>,
    ) -> Result<(synaptix_core::tensor::Tensor, (usize, usize)), PipelineError> {
        let limits = self.image_limits(max_tokens)?;
        self.encode_image_with_grid(path, limits)
    }

    pub fn image_token_count(
        &self,
        path: impl AsRef<Path>,
        limits: synaptix_vlm_qwen3::PreprocessLimits,
    ) -> Result<usize, PipelineError> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let img = synaptix_io::image::png::load_image(path, Device::Cpu)
            .map_err(|e| PipelineError::Load(format!("image: {e}")))?;
        let dims = img.dims();
        let (nh, nw) = synaptix_vlm_qwen3::preprocess::smart_resize(
            dims[1],
            dims[2],
            tower.config.size_factor(),
            limits,
        );
        let p = tower.config.patch_size;
        Ok((nh / p) * (nw / p) / tower.config.merge_unit())
    }

    /// Кодирует видео: сэмплинг кадров (ffprobe/ffmpeg) → башня по группам
    /// кадров. Блок промпта собирает [`VideoPromptInfo::prompt_block`] —
    /// с таймкодом каждой группы, как у HF-процессора Qwen3-VL.
    pub fn encode_video(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(synaptix_core::tensor::Tensor, VideoPromptInfo), PipelineError> {
        use synaptix_core::grad::no_grad;
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| PipelineError::Model("vision-башня не загружена".into()))?;
        let synaptix_vlm_qwen3::PreparedVideo { patches, grid, group_timestamps } =
            synaptix_vlm_qwen3::prepare_video(
                path,
                &tower.config,
                synaptix_vlm_qwen3::VideoLimits::default(),
                self.model.device,
            )
            .map_err(|e| PipelineError::Load(format!("video: {e}")))?;
        let feats = no_grad(|| tower.forward(&patches, grid))
            .map_err(|e| PipelineError::Forward(format!("vision forward: {e}")))?;
        let m = tower.config.spatial_merge_size.max(1);
        let info = VideoPromptInfo {
            groups: grid.t,
            tokens_per_group: (grid.h * grid.w) / tower.config.merge_unit(),
            timestamps: group_timestamps,
            grid_hw: (grid.h / m, grid.w / m),
        };
        Ok((feats, info))
    }

    /// Эмбеддинги промпта, где прогоны токенов-заполнителей заменены
    /// строками vision-эмбеддингов. `media` — `(pad_id, тензор)` по
    /// модальностям; тензор модальности — конкатенация всех её вложений в
    /// порядке появления в промпте, каждый прогон заполнителя забирает
    /// следующие `run` строк своим курсором (у видео таких прогонов —
    /// по одному на группу кадров).
    fn embed_with_media(
        &self,
        ids: &[u32],
        media: &[(u32, &synaptix_core::tensor::Tensor)],
    ) -> Result<synaptix_core::tensor::Tensor, PipelineError> {
        use synaptix_core::tensor::Tensor;
        let device = self.model.device;
        let hidden = self.config.hidden_size;
        let mut cursors = vec![0usize; media.len()];
        let is_pad = |id: u32| media.iter().position(|(pad, _)| *pad == id);
        let mut segments: Vec<Tensor> = Vec::new();
        let mut i = 0usize;
        while i < ids.len() {
            if let Some(mi) = is_pad(ids[i]) {
                let pad = ids[i];
                let start = i;
                while i < ids.len() && ids[i] == pad {
                    i += 1;
                }
                let run = i - start;
                let (_, emb) = media[mi];
                let avail = emb.dims()[0];
                let off = cursors[mi];
                if off + run > avail {
                    return Err(PipelineError::Forward(format!(
                        "блок заполнителя {pad}: нужно строк {}..{}, а vision дал {avail}",
                        off,
                        off + run
                    )));
                }
                cursors[mi] += run;
                let e = emb
                    .narrow(0, off, run)
                    .and_then(|t| t.contiguous())
                    .and_then(|t| t.to_dtype(self.model.dtype))
                    .and_then(|t| t.reshape(vec![1usize, run, hidden]))
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                segments.push(e);
            } else {
                let start = i;
                while i < ids.len() && is_pad(ids[i]).is_none() {
                    i += 1;
                }
                let chunk = Tensor::from_vec(ids[start..i].to_vec(), vec![1usize, i - start], device)
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                let e = self
                    .model
                    .embed_ids(&chunk)
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                segments.push(e);
            }
        }
        for (mi, (pad, emb)) in media.iter().enumerate() {
            if cursors[mi] != emb.dims()[0] {
                return Err(PipelineError::Forward(format!(
                    "заполнитель {pad}: в промпте {} строк эмбеддингов, а vision дал {}",
                    cursors[mi],
                    emb.dims()[0]
                )));
            }
        }
        let refs: Vec<&Tensor> = segments.iter().collect();
        Tensor::cat(&refs, 1).map_err(|e| PipelineError::Forward(e.to_string()))
    }

    /// Генерация по промпту с картинками: `image_embeds` — по тензору на
    /// картинку в порядке появления в промпте. Сетки неизвестны, поэтому
    /// без M-RoPE (см. [`Self::generate_with_media`]).
    pub fn generate_with_images(
        &self,
        prompt_ids: &[u32],
        image_embeds: &[synaptix_core::tensor::Tensor],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        use synaptix_core::tensor::Tensor;
        let refs: Vec<&Tensor> = image_embeds.iter().collect();
        let feats = Tensor::cat(&refs, 0).map_err(|e| PipelineError::Forward(e.to_string()))?;
        let pad = self
            .config
            .image_token_id
            .ok_or_else(|| PipelineError::Model("config.json без image_token_id".into()))?;
        let input = MediaInput { pad, embeds: feats, grids: Vec::new() };
        self.generate_with_media(prompt_ids, std::slice::from_ref(&input), gen_cfg, sink)
    }

    /// Генерация по промпту с картинками и/или видео. Эмбеддинги каждой
    /// модальности — конкатенация по порядку появления в промпте (см.
    /// [`Self::embed_with_media`]). Если конфиг несёт `mrope_section`, а у
    /// всех вложений известны сетки, позиции RoPE — мультимодальные
    /// (`crate::mrope`): префилл по таблицам на весь промпт, декод — по
    /// 1D со сдвигом. Спекулятивные пути (MTP / CUDA-graph) здесь не
    /// применяются — префилл идёт по готовым эмбеддингам.
    pub fn generate_with_media(
        &self,
        prompt_ids: &[u32],
        inputs: &[MediaInput],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        use synaptix_core::grad::no_grad;
        use synaptix_core::tensor::Tensor;

        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        if inputs.is_empty() {
            return Err(PipelineError::Model("generate_with_media без медиа".into()));
        }
        let media: Vec<(u32, &Tensor)> = inputs.iter().map(|m| (m.pad, &m.embeds)).collect();

        // M-RoPE: таблицы cos/sin на весь промпт + сдвиг позиций декода.
        let mrope_on = mrope_enabled() && inputs.iter().all(|m| !m.grids.is_empty());
        let mrope_tables = match (&self.config.mrope, mrope_on) {
            (Some(spec), true) => {
                let grids: Vec<Vec<mrope::Grid3>> = inputs
                    .iter()
                    .map(|m| m.grids.iter().map(|(h, w)| mrope::Grid3::image(*h, *w)).collect())
                    .collect();
                let runs: Vec<mrope::MediaRuns> = inputs
                    .iter()
                    .zip(&grids)
                    .map(|(m, g)| mrope::MediaRuns { pad: m.pad, grids: g })
                    .collect();
                let positions = mrope::positions_3d(prompt_ids, &runs).map_err(PipelineError::Forward)?;
                let inv = self.config.rope_inv_freqs();
                let (cos, sin) = mrope::rope_tables(&positions.pos, &inv, &spec.section, spec.interleaved);
                let half = inv.len();
                let l = prompt_ids.len();
                let cos = Tensor::from_vec(cos, vec![l, half], self.model.device)
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                let sin = Tensor::from_vec(sin, vec![l, half], self.model.device)
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                Some((cos, sin, positions.decode_delta()))
            }
            _ => None,
        };
        let prefill_pos = match &mrope_tables {
            Some((cos, sin, _)) => RopePositions::Tables { cos, sin },
            None => RopePositions::Sequential,
        };
        let decode_pos = match &mrope_tables {
            Some((_, _, delta)) => RopePositions::Shifted(*delta),
            None => RopePositions::Sequential,
        };
        let cfg = self.prepare_cfg(gen_cfg);
        let device = self.model.device;
        let l = prompt_ids.len();
        let kv_max = cfg.max_seq.unwrap_or(l + cfg.max_new_tokens + 1);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let eos = synaptix_llm_common::generate::eos_set(&cfg);
        let mut sampler = synaptix_llm_common::generate::TokenSampler::new(&cfg, prompt_ids);

        let t0 = std::time::Instant::now();
        let emb = self.embed_with_media(prompt_ids, &media)?;
        // Префилл чанками, как в текстовом пути: пик активаций гибрида
        // растёт с длиной чанка, и single-shot по длинной истории с
        // картинкой упирался в OOM. `prepare_cfg` уже выровнял чанк по
        // границе GDN-скана (кратные 64 bit-exact к single-shot).
        let chunk = match cfg.prefill_batch {
            0 => l,
            n => n.max(1),
        };
        let mut off = 0usize;
        let mut last_hidden = None;
        while off < l {
            let step = chunk.min(l - off);
            let part = emb
                .narrow(1, off, step)
                .and_then(|t| t.contiguous())
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let h = no_grad(|| self.model.forward_from_hidden_pos(&part, &mut kv, prefill_pos))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            last_hidden = Some(h);
            off += step;
        }
        let hidden =
            last_hidden.ok_or_else(|| PipelineError::Forward("empty prefill".into()))?;
        let mut logits = self
            .model
            .head_at(&hidden, hidden.dims()[1] - 1)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let mut out: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let dec_t0 = std::time::Instant::now();
        loop {
            let tok = sampler.sample(&logits).map_err(PipelineError::from)?;
            out.push(tok);
            if !sink.on_token(tok) || out.len() >= cfg.max_new_tokens || eos.contains(&tok) {
                break;
            }
            if kv.seq_len >= kv.max_seq {
                break;
            }
            let step = Tensor::from_vec(vec![tok], vec![1usize, 1], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            logits = no_grad(|| self.model.forward_pos(&step, &mut kv, decode_pos))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
        }
        let decode_ms = dec_t0.elapsed().as_millis();
        let new_tokens = out.len();
        Ok((
            out,
            GenerationStats {
                prompt_tokens: l,
                new_tokens,
                prefill_ms,
                decode_ms,
            },
        ))
    }

    /// Размеры кэшей MTP-декода под контекст `ctx_tokens` и `max_new` токенов
    /// ответа: `(kv_max, mtp_cap)`. Вынесено, чтобы вызывающий мог аллоцировать
    /// кэши один раз на диалог и переиспользовать префикс между ходами
    /// ([`Self::generate_mtp_resume`]).
    pub fn mtp_cache_caps(&self, ctx_tokens: usize, max_new: usize) -> (usize, usize) {
        let kv_max = ctx_tokens.max(2);
        // mtp_kv растёт быстрее основного kv: +1 на draft и +1 на advance за шаг,
        // а rollback отклонённого драфта основного kv его не откатывает — до
        // 2×max_new_tokens сверх префилла. Выше RoPE-ёмкости MTP-модуля
        // аллоцировать нельзя — при исчерпании кэша decode доработает остаток
        // обычным путём.
        let cap = self.mtp.as_ref().map(|m| m.rope_capacity()).unwrap_or(kv_max);
        (kv_max, (kv_max + max_new + 2).min(cap))
    }

    /// Создать пару кэшей под MTP-декод (основной + кэш MTP-головы).
    pub fn make_mtp_caches(
        &self,
        ctx_tokens: usize,
        max_new: usize,
    ) -> Result<(synaptix_llm_common::KvCache, synaptix_llm_common::KvCache), PipelineError> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| PipelineError::Model("MTP-голова не загружена".into()))?;
        let (kv_max, mtp_cap) = self.mtp_cache_caps(ctx_tokens, max_new);
        let kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let mtp_kv = mtp
            .make_kv_cache(1, mtp_cap)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        Ok((kv, mtp_kv))
    }

    pub fn generate_mtp(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, MtpStats), PipelineError> {
        let (mut kv, mut mtp_kv) = self.mtp_caches_for(prompt_ids, &gen_cfg)?;
        self.generate_mtp_inner(&mut kv, &mut mtp_kv, prompt_ids, gen_cfg, sink, false, None)
    }

    pub fn generate_mtp_with_graph(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats, MtpStats), PipelineError> {
        let use_graph = !self.model.has_mxfp8_head_or_embed();
        let (mut kv, mut mtp_kv) = self.mtp_caches_for(prompt_ids, &gen_cfg)?;
        self.generate_mtp_inner(&mut kv, &mut mtp_kv, prompt_ids, gen_cfg, sink, use_graph, None)
    }

    /// MTP-декод по ГОТОВЫМ кэшам: префилл стартует с `kv.seq_len`, то есть
    /// история, уже посчитанная на прошлом ходу, не считается заново
    /// (префикс-KV). Вызывающий отвечает за то, что `prompt_ids[..kv.seq_len]`
    /// — ровно те токены, что лежат в кэше, а linear-состояние GDN
    /// восстановлено на эту же границу (см. `KvCache::snapshot_linear_full`).
    ///
    /// `on_prefill` зовётся сразу после префилла — там вызывающий снимает новую
    /// точку возврата (снапшот GDN + позицию `mtp_kv`), пока декод её не
    /// продвинул.
    pub fn generate_mtp_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        mtp_kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
        on_prefill: &mut dyn FnMut(
            usize,
            &synaptix_llm_common::KvCache,
            &synaptix_llm_common::KvCache,
        ) -> Result<(), PipelineError>,
    ) -> Result<(Vec<u32>, GenerationStats, MtpStats), PipelineError> {
        let use_graph = !self.model.has_mxfp8_head_or_embed();
        self.generate_mtp_inner(kv, mtp_kv, prompt_ids, gen_cfg, sink, use_graph, Some(on_prefill))
    }

    fn mtp_caches_for(
        &self,
        prompt_ids: &[u32],
        cfg: &GenerationConfig,
    ) -> Result<(synaptix_llm_common::KvCache, synaptix_llm_common::KvCache), PipelineError> {
        let ctx = cfg
            .max_seq
            .unwrap_or(prompt_ids.len() + cfg.max_new_tokens + 2);
        self.make_mtp_caches(ctx, cfg.max_new_tokens)
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_mtp_inner(
        &self,
        mut kv: &mut synaptix_llm_common::KvCache,
        mut mtp_kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
        use_graph: bool,
        mut on_prefill: Option<
            &mut dyn FnMut(
                usize,
                &synaptix_llm_common::KvCache,
                &synaptix_llm_common::KvCache,
            ) -> Result<(), PipelineError>,
        >,
    ) -> Result<(Vec<u32>, GenerationStats, MtpStats), PipelineError> {
        use synaptix_core::grad::no_grad;
        use synaptix_core::tensor::Tensor;

        let Some(mtp) = self.mtp.as_ref() else {
            return Err(PipelineError::Model("MTP-голова не загружена".into()));
        };
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.prepare_cfg(gen_cfg);
        let stochastic = cfg.temperature > 0.0;
        let device = self.model.device;
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let l = prompt.len();
        let eos = synaptix_llm_common::generate::eos_set(&cfg);
        if l > kv.max_seq {
            return Err(PipelineError::Forward(format!(
                "промпт {l} ток не влезает в KV-кэш на {} ток",
                kv.max_seq
            )));
        }

        let argmax = |t: &Tensor| -> Result<u32, PipelineError> {
            let v = t
                .to_dtype(synaptix_core::dtype::DType::F32)
                .and_then(|x| x.flatten_all())
                .and_then(|x| x.to_vec1::<f32>())
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let mut best = 0usize;
            for (i, x) in v.iter().enumerate() {
                if *x > v[best] {
                    best = i;
                }
            }
            Ok(best as u32)
        };

        let t0 = std::time::Instant::now();
        let chunk = match cfg.prefill_batch {
            0 => l,
            n => n.max(1),
        };
        let mut hidden = None;
        // Префикс-KV: всё до `kv.seq_len` уже посчитано на прошлом ходу.
        // Минус один токен — логиты нужно получить хотя бы из одного forward'а.
        let prefix = kv.seq_len.min(l.saturating_sub(1));
        kv.seq_len = prefix;
        let mut off = prefix;
        // Точку возврата для следующего хода ставим на границу, КРАТНУЮ CS=64
        // GDN-скана: разрез посередине его внутреннего чанка оставляет
        // состояние частичного чанка и расходится с непрерывным префиллом
        // (проверено `prefill_chunk_divergence`: 64/128/256 — bit-exact,
        // 47/73/100 — нет). Поэтому хвост промпта после последней кратной
        // границы в кэш-точку не входит и на следующем ходу считается заново.
        let snap_at = if on_prefill.is_some() {
            (l / GDN_CHUNK) * GDN_CHUNK
        } else {
            0
        };
        trace_prefill_mem(device, off, l);
        while off < l {
            let mut end = (off + chunk).min(l);
            if snap_at > off && end > snap_at {
                end = snap_at;
            }
            let part = Tensor::from_vec(
                prompt[off..end].to_vec(),
                vec![1usize, end - off],
                device,
            )
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let h = no_grad(|| self.model.forward_trunk(&part, &mut kv))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let n = end - off;
            let adv = if end < l { n } else { n - 1 };
            if adv > 0 {
                let hs = h
                    .narrow(1, 0, adv)
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                let shifted = Tensor::from_vec(
                    prompt[off + 1..off + 1 + adv].to_vec(),
                    vec![1usize, adv],
                    device,
                )
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
                no_grad(|| mtp.advance(&self.model, &hs, &shifted, &mut mtp_kv))
                    .map_err(|e| PipelineError::Forward(format!("mtp prefill: {e}")))?;
            }
            hidden = Some(h);
            off = end;
            trace_prefill_mem(device, off, l);
            // Точка возврата снимается ЗДЕСЬ, до декода: он продвинет и kv, и
            // linear-состояние, и mtp_kv. Перед снимком выравниваем host и dev
            // половины GDN-состояния: префилл живёт в dev-зеркалах, а
            // `sync_decode_dev_state` перед декодом пересеивает их из host —
            // снимок обязан застать обе половины согласованными, иначе
            // восстановление на следующем ходу подсунет декоду устаревший host.
            // Round-trip f16→f32→f16 точный, поэтому на сам префилл не влияет.
            if off == snap_at && snap_at > 0 {
                if let Some(cb) = on_prefill.as_mut() {
                    self.model
                        .sync_decode_host_state(kv)
                        .map_err(|e| PipelineError::Forward(format!("prefix sync host: {e}")))?;
                    self.model
                        .sync_decode_dev_state(kv)
                        .map_err(|e| PipelineError::Forward(format!("prefix sync dev: {e}")))?;
                    cb(off, kv, mtp_kv)?;
                }
            }
        }
        let hidden = hidden.ok_or_else(|| PipelineError::Forward("prefill: пустой промпт".into()))?;
        let last = hidden.dims()[1] - 1;
        let mut h_prev = hidden
            .narrow(1, last, 1)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let mut sampler = synaptix_llm_common::generate::TokenSampler::new(&cfg, &prompt);
        let head0 = self
            .model
            .head_at(&hidden, last)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let first = if stochastic {
            sampler.sample(&head0).map_err(PipelineError::from)?
        } else {
            argmax(&head0)?
        };
        let prefill_ms = t0.elapsed().as_millis();

        let mut out: Vec<u32> = vec![first];
        let mut cancelled = !sink.on_token(first);
        let mut cur = first;
        let mut forced: Option<u32> = None;
        let mut draft_q: Option<Vec<f32>> = None;
        let mut stats = MtpStats::default();

        let mut graph_ctx = if use_graph {
            Some(self.build_mtp_graph(&mut kv)?)
        } else {
            None
        };

        let mut snap_buf = kv
            .alloc_linear_snapshot()
            .map_err(|e| PipelineError::Forward(e.to_string()))?;

        let mut mprof = MtpProf::new(device);
        let dec_t0 = std::time::Instant::now();
        while !cancelled && out.len() < cfg.max_new_tokens && !eos.contains(&cur) {
            let pos = kv.seq_len;
            if pos + 2 > kv.max_seq {
                break;
            }
            // MTP-кэш исчерпан (draft+advance добавляют 1-2 слота за шаг и не
            // откатываются при reject) — дорабатываем остаток обычным
            // autoregressive decode на основном kv, MTP больше не трогаем.
            let mtp_need = if forced.is_some() { 1 } else { 2 };
            if mtp_kv.seq_len + mtp_need > mtp_kv.max_seq {
                // Отложенный forced-токен уже эмитирован (push при reject), но в
                // kv ещё не проведён — прогоняем [cur, forced] без повторного
                // emit и продолжаем с него.
                if let Some(second) = forced.take() {
                    if pos + 2 > kv.max_seq {
                        break;
                    }
                    let chunk = Tensor::from_vec(vec![cur, second], vec![1usize, 2], device)
                        .map_err(|e| PipelineError::Forward(e.to_string()))?;
                    let _ = no_grad(|| self.model.forward(&chunk, &mut kv))
                        .map_err(|e| PipelineError::Forward(e.to_string()))?;
                    cur = second;
                }
                while !cancelled && out.len() < cfg.max_new_tokens && !eos.contains(&cur) {
                    if kv.seq_len >= kv.max_seq {
                        break;
                    }
                    let t = Tensor::from_vec(vec![cur], vec![1usize, 1], device)
                        .map_err(|e| PipelineError::Forward(e.to_string()))?;
                    let logits = no_grad(|| self.model.forward(&t, &mut kv))
                        .map_err(|e| PipelineError::Forward(e.to_string()))?;
                    let tok = if stochastic {
                        sampler.sample(&logits).map_err(PipelineError::from)?
                    } else {
                        argmax(&logits)?
                    };
                    out.push(tok);
                    cancelled = !sink.on_token(tok);
                    cur = tok;
                }
                break;
            }
            let (second, was_draft) = match forced.take() {
                Some(t) => (t, false),
                None => {
                    mprof.begin();
                    let next = Tensor::from_vec(vec![cur], vec![1usize, 1], device)
                        .map_err(|e| PipelineError::Forward(e.to_string()))?;
                    let dl = no_grad(|| mtp.draft_logits(&self.model, &h_prev, &next, &mut mtp_kv))
                        .map_err(|e| PipelineError::Forward(format!("mtp draft: {e}")))?;
                    stats.drafted += 1;
                    mprof.end("draft");
                    mprof.begin();
                    let tok = if stochastic {
                        let q = sampler.probs(&dl).map_err(PipelineError::from)?;
                        let t = sampler.sample_from_probs(&q);
                        draft_q = Some(q);
                        t
                    } else {
                        argmax(&dl)?
                    };
                    mprof.end("draft_sample");
                    (tok, true)
                }
            };
            stats.steps += 1;

            if was_draft {
                mprof.begin();
                kv.save_linear_into(&mut snap_buf)
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                mprof.end("snapshot");
            }
            let seq_before = kv.seq_len;

            mprof.begin();
            let step = {
                {
                    match graph_ctx.as_mut() {
                        Some(g) => g.run(&mut kv, cur, second, pos as u32)?,
                        None => self.verify_plain(&mut kv, cur, second, device)?,
                    }
                }
            };
            mprof.end("verify");
            mprof.begin();
            let (a, draft_accepted) = if !was_draft {
                (second, false)
            } else if stochastic {
                let p = sampler.probs(&step.logits0).map_err(PipelineError::from)?;
                match draft_q.take() {
                    Some(q) => {
                        let idx = second as usize;
                        let qi = q.get(idx).copied().unwrap_or(0.0);
                        let pi = p.get(idx).copied().unwrap_or(0.0);
                        let ratio = if qi > 0.0 { (pi / qi).min(1.0) } else { 0.0 };
                        if sampler.uniform() < ratio {
                            (second, true)
                        } else {
                            let mut resid: Vec<f32> = p
                                .iter()
                                .zip(q.iter())
                                .map(|(pv, qv)| (pv - qv).max(0.0))
                                .collect();
                            let sum: f32 = resid.iter().sum();
                            if sum > 0.0 {
                                for x in resid.iter_mut() {
                                    *x /= sum;
                                }
                                (sampler.sample_from_probs(&resid), false)
                            } else {
                                (sampler.sample_from_probs(&p), false)
                            }
                        }
                    }
                    None => (sampler.sample_from_probs(&p), false),
                }
            } else {
                let a = argmax(&step.logits0)?;
                let acc = a == second;
                (a, acc)
            };
            mprof.end("accept_sample");

            mprof.begin();
            let fill = Tensor::from_vec(vec![a], vec![1usize, 1], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            no_grad(|| mtp.advance(&self.model, &step.hidden0, &fill, &mut mtp_kv))
                .map_err(|e| PipelineError::Forward(format!("mtp advance: {e}")))?;
            mprof.end("advance");

            if !was_draft {
                mprof.begin();
                let b = if stochastic {
                    sampler.sample(&step.logits1).map_err(PipelineError::from)?
                } else {
                    argmax(&step.logits1)?
                };
                mprof.end("emit_sample");
                out.push(b);
                cancelled = !sink.on_token(b);
                h_prev = step.hidden1;
                cur = b;
                continue;
            }

            out.push(a);
            cancelled = !sink.on_token(a);
            sampler.commit(a);
            if draft_accepted {
                stats.accepted += 1;
                if eos.contains(&a) || out.len() >= cfg.max_new_tokens {
                    cur = a;
                    continue;
                }
                mprof.begin();
                let b = if stochastic {
                    sampler.sample(&step.logits1).map_err(PipelineError::from)?
                } else {
                    argmax(&step.logits1)?
                };
                mprof.end("emit_sample");
                out.push(b);
                cancelled = !sink.on_token(b);
                h_prev = step.hidden1;
                cur = b;
            } else {
                mprof.begin();
                kv.restore_linear(&snap_buf)
                    .map_err(|e| PipelineError::Forward(e.to_string()))?;
                kv.seq_len = seq_before;
                mprof.end("restore");
                forced = Some(a);
            }
        }
        let decode_ms = dec_t0.elapsed().as_millis();
        mprof.report();

        let stats_gen = GenerationStats {
            prompt_tokens: l,
            new_tokens: out.len(),
            prefill_ms,
            decode_ms,
        };
        Ok((out, stats_gen, stats))
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))?;
        Ok(enc.ids.clone())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    /// Промпт в том виде, в каком его видит движок (с BOS, если модель его
    /// добавляет) — вызывающему нужен именно он, чтобы сравнивать префиксы.
    pub fn maybe_prepend_bos(&self, prompt_ids: &[u32]) -> Vec<u32> {
        match self.config.bos_token_id {
            Some(bos) if self.add_bos && prompt_ids.first() != Some(&bos) => {
                let mut v = Vec::with_capacity(prompt_ids.len() + 1);
                v.push(bos);
                v.extend_from_slice(prompt_ids);
                v
            }
            _ => prompt_ids.to_vec(),
        }
    }

    fn prepare_cfg(&self, mut cfg: GenerationConfig) -> GenerationConfig {
        if cfg.eos_token_id.is_none() && cfg.eos_token_ids.is_empty() {
            cfg.eos_token_ids = self.config.eos_ids();
        }
        cfg.prefill_batch = effective_prefill_chunk(cfg.prefill_batch);
        cfg
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let cfg = self.prepare_cfg(gen_cfg);
        synaptix_llm_common::generate::generate(&self.model, &prompt, &cfg)
            .map_err(PipelineError::from)
    }

    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let cfg = self.prepare_cfg(gen_cfg);
        synaptix_llm_common::generate::generate_streaming(&self.model, &prompt, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_streaming_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.prepare_cfg(gen_cfg);
        synaptix_llm_common::generate::generate_streaming_resume(&self.model, kv, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        gen_cfg: GenerationConfig,
    ) -> Result<(String, GenerationStats), PipelineError> {
        let ids = self.encode(prompt)?;
        let (new_ids, stats) = self.generate(&ids, gen_cfg)?;
        let text = self.decode(&new_ids)?;
        Ok((text, stats))
    }

    /// CUDA-graph decode для гибрида (linear + full слои). Prefill — обычный
    /// батч-forward (host-scan для linear строит host-состояние + device KV).
    /// Затем device-зеркала linear-state сеются из host (S0); warmup+capture
    /// одного `forward_decode_dev` (продвигает dev-state, т.к. рекуррентность НЕ
    /// идемпотентна — но host-векторы не тронуты); dev-state восстанавливается в
    /// S0 пере-сеянием; replay-loop обрабатывает токены начиная с tok0@L (по
    /// одному advance state за launch). Greedy совпадает с [`Self::generate`] с
    /// точностью до F16-compute. Требует CUDA-устройство, **compute=F16** (ядра
    /// linear-decode F16-нативные) и не-FP8 KV.
    pub fn generate_with_graph(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let mut noop = |_: u32| true;
        self.generate_with_graph_streaming(prompt_ids, gen_cfg, &mut noop)
    }

    pub fn generate_with_graph_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let prompt = self.maybe_prepend_bos(prompt_ids);
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        self.generate_with_graph_resume(&mut kv, &prompt, gen_cfg, sink)
    }

    /// Как [`Self::generate_with_graph_streaming`], но prefill стартует с `kv.seq_len`
    /// (prefix-KV-кэш). После decode синкает device→host linear-состояние, чтобы
    /// следующий ход продолжил host-scan корректно. `prompt_ids` — уже с BOS (если
    /// нужен): caller отвечает за совпадение с кэшированным префиксом.
    pub fn generate_with_graph_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        use synaptix_core::grad::no_grad;
        use synaptix_infer::graph_capture::GraphCapturer;
        use synaptix_infer::InferError;

        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        if self.model.dtype != DType::F16 {
            return Err(PipelineError::Forward(format!(
                "generate_with_graph: linear-decode ядра требуют compute=F16, получено {:?}",
                self.model.dtype
            )));
        }
        let device = self.model.device;
        let ord = match device {
            Device::Cuda(o) => o,
            _ => return Err(PipelineError::Forward("generate_with_graph requires CUDA device".into())),
        };
        let l = prompt_ids.len();
        let gen_cfg = self.prepare_cfg(gen_cfg);
        let eos = synaptix_llm_common::generate::eos_set(&gen_cfg);
        let mut sampler = synaptix_llm_common::generate::TokenSampler::new(&gen_cfg, prompt_ids);
        let prefix = kv.seq_len.min(l.saturating_sub(1));
        kv.seq_len = prefix;

        // Prefill хвоста prompt_ids[prefix..] чанками: device chunked-scan для linear
        // + device KV для full. Чанкуем, чтобы ограничить пик памяти (буферы scan-
        // оркестратора и активации растут с длиной чанка, а не всего промпта) —
        // иначе длинный первый prefill упирается в VRAM. Стейт linear/conv и KV
        // переносятся между чанками внутри одного `kv`. Берём логиты последнего чанка.
        let suffix = &prompt_ids[prefix..];
        // prepare_cfg гарантирует prefill_batch > 0 и кратность 64 (границы
        // чанков на не-кратных CS=64 позициях ломают состояние GDN-скана; на
        // кратных чанкование bit-exact к single-shot — prefill_chunk_divergence).
        let chunk = if gen_cfg.prefill_batch > 0 {
            gen_cfg.prefill_batch
        } else {
            suffix.len().max(1)
        };
        let t0 = std::time::Instant::now();
        let mut logits_opt = None;
        let mut off = 0usize;
        while off < suffix.len() {
            let end = (off + chunk).min(suffix.len());
            let part = Tensor::from_vec(suffix[off..end].to_vec(), vec![1usize, end - off], device)
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            let lg = no_grad(|| self.model.forward(&part, kv))
                .map_err(|e| PipelineError::Forward(e.to_string()))?;
            logits_opt = Some(lg);
            off = end;
        }
        let logits = logits_opt.ok_or_else(|| PipelineError::Forward("empty prefill suffix".into()))?;
        let prefill_ms = t0.elapsed().as_millis();

        let mut out: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
        let tok0 = sampler.sample(&logits).map_err(PipelineError::from)?;
        out.push(tok0);
        let mut cancelled = !sink.on_token(tok0);

        // Prefill держит linear-state device-резидентным (без per-chunk host
        // round-trip). Обновляем host из dev ОДИН раз — для decode-handoff и
        // continuation. No-op, если prefill не шёл (dev=None) → dev засеется
        // из host ниже.
        self.model
            .sync_decode_host_state(&mut *kv)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        // Засеять device-зеркала linear-state из host (post-prefill S0).
        self.model
            .sync_decode_dev_state(&mut *kv)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;

        let mut state = self
            .model
            .make_decode_state()
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        state.update(tok0, l as u32).map_err(|e| PipelineError::Forward(e.to_string()))?;
        let stream = synaptix_core::device::cuda::default_stream(ord)
            .map_err(|e| PipelineError::Forward(format!("stream: {e}")))?;

        let mut capturer = GraphCapturer::new(3);

        let graph = {
            let model = &self.model;
            let state_ref = &mut state;
            let kv_ref = &mut *kv;
            no_grad(|| {
                capturer.capture_with(&stream, |_s| {
                    model
                        .forward_decode_dev(state_ref, kv_ref)
                        .map_err(|e| InferError::Other(e.to_string()))
                })
            })
        }
        .map_err(|e| PipelineError::Forward(format!("graph capture: {e}")))?;
        let _ = graph.upload();

        // Восстановить S0: warmup+capture продвинули dev linear-state (не
        // идемпотентно), но host-векторы нетронуты → пере-сеять. KV full-слоёв
        // идемпотентен (slot L переписан tok0), восстановления не требует.
        self.model
            .sync_decode_dev_state(&mut *kv)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;

        // decode_ms измеряет ЧИСТЫЙ replay (steady-state throughput); capture —
        // одноразовая стоимость (warmup×3 + capture + instantiate), не в метрике.
        let dec_t0 = std::time::Instant::now();
        // Replay-loop: обрабатываем out[len-1] на позиции L+len-1 (старт tok0@L),
        // каждый launch продвигает linear-state на один шаг.
        while !cancelled && out.len() < gen_cfg.max_new_tokens {
            let last = *out.last().unwrap();
            if eos.contains(&last) {
                break;
            }
            let pos = (l + out.len() - 1) as u32;
            if (pos as usize) >= kv.max_seq {
                break;
            }
            state.update(last, pos).map_err(|e| PipelineError::Forward(e.to_string()))?;
            graph
                .launch()
                .map_err(|e| PipelineError::Forward(format!("graph launch: {e:?}")))?;
            stream
                .synchronize()
                .map_err(|e| PipelineError::Forward(format!("sync post-launch: {e:?}")))?;
            let tok = sampler.sample(&state.logits).map_err(PipelineError::from)?;
            out.push(tok);
            cancelled = !sink.on_token(tok);
        }
        let decode_ms = dec_t0.elapsed().as_millis();
        kv.seq_len = (l + out.len() - 1).min(kv.max_seq);
        // graph-decode продвинул только device linear-state → вернуть в host, чтобы
        // следующий ход (prefix-KV-кэш) продолжил host-scan с верного состояния.
        self.model
            .sync_decode_host_state(kv)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;

        let stats = GenerationStats {
            prompt_tokens: l,
            new_tokens: out.len(),
            prefill_ms,
            decode_ms,
        };
        Ok((out, stats))
    }
}


struct VerifyStep {
    logits0: synaptix_core::tensor::Tensor,
    logits1: synaptix_core::tensor::Tensor,
    hidden0: synaptix_core::tensor::Tensor,
    hidden1: synaptix_core::tensor::Tensor,
}

/// Пофазный профиль MTP-цикла (SYN_MTP_PROF=1): draft / snapshot / verify /
/// sample / advance / restore, с device-sync на границах фаз. Перф-инструмент,
/// в проде выключен (env не выставлен → нулевые накладные).
struct MtpProf {
    on: bool,
    ord: Option<usize>,
    t0: std::time::Instant,
    acc: std::collections::BTreeMap<&'static str, (f64, u64)>,
}

impl MtpProf {
    fn new(device: Device) -> Self {
        let on = std::env::var("SYN_MTP_PROF").as_deref() == Ok("1");
        let ord = match device {
            Device::Cuda(o) if on => Some(o),
            _ => None,
        };
        Self { on, ord, t0: std::time::Instant::now(), acc: Default::default() }
    }

    fn sync(&self) {
        if let Some(o) = self.ord {
            let _ = synaptix_core::device::cuda::synchronize(o);
        }
    }

    fn begin(&mut self) {
        if !self.on {
            return;
        }
        self.sync();
        self.t0 = std::time::Instant::now();
    }

    fn end(&mut self, name: &'static str) {
        if !self.on {
            return;
        }
        self.sync();
        let dt = self.t0.elapsed().as_secs_f64() * 1000.0;
        let e = self.acc.entry(name).or_insert((0.0, 0));
        e.0 += dt;
        e.1 += 1;
    }

    fn report(&self) {
        if !self.on {
            return;
        }
        let total: f64 = self.acc.values().map(|(t, _)| t).sum();
        let mut lines: Vec<_> = self.acc.iter().collect();
        lines.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        eprintln!("=== MTP loop breakdown (synced, {total:.1} ms) ===");
        for (k, (t, c)) in lines {
            eprintln!(
                "  {k:12} {t:9.2} ms  {:5.1}%  ({c} шагов, {:.3} ms/шаг)",
                100.0 * t / total.max(1e-9),
                t / *c as f64
            );
        }
    }
}

pub struct MtpGraph {
    state: synaptix_llm_common::model::PrefillState,
    graph: std::sync::Arc<synaptix_core::device::cuda::CudaGraph>,
    stream: std::sync::Arc<synaptix_core::device::cuda::Stream>,
    hidden_size: usize,
}

impl HybridPipeline {
    fn verify_plain(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        cur: u32,
        second: u32,
        device: Device,
    ) -> Result<VerifyStep, PipelineError> {
        use synaptix_core::grad::no_grad;
        use synaptix_core::tensor::Tensor;
        let chunk = Tensor::from_vec(vec![cur, second], vec![1usize, 2], device)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let hh = no_grad(|| self.model.forward_trunk(&chunk, kv))
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let logits0 = self
            .model
            .head_at(&hh, 0)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let logits1 = self
            .model
            .head_at(&hh, 1)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        let hidden0 = hh.narrow(1, 0, 1).map_err(|e| PipelineError::Forward(e.to_string()))?;
        let hidden1 = hh.narrow(1, 1, 1).map_err(|e| PipelineError::Forward(e.to_string()))?;
        Ok(VerifyStep { logits0, logits1, hidden0, hidden1 })
    }
}

impl MtpGraph {
    fn run(
        &mut self,
        kv: &mut synaptix_llm_common::KvCache,
        cur: u32,
        second: u32,
        pos: u32,
    ) -> Result<VerifyStep, PipelineError> {
        self.state
            .update(&[cur, second], pos)
            .map_err(|e| PipelineError::Forward(format!("mtp graph update: {e}")))?;
        self.graph
            .launch()
            .map_err(|e| PipelineError::Forward(format!("mtp graph launch: {e:?}")))?;
        self.stream
            .synchronize()
            .map_err(|e| PipelineError::Forward(format!("sync post-launch: {e:?}")))?;
        kv.seq_len = pos as usize + 2;
        let row = |t: &synaptix_core::tensor::Tensor, i: usize| {
            t.narrow(0, i, 1)
                .and_then(|x| x.contiguous())
                .map_err(|e| PipelineError::Forward(format!("mtp graph row {i}: {e}")))
        };
        let h0 = row(&self.state.hidden, 0)?
            .reshape(vec![1usize, 1, self.hidden_size])
            .map_err(|e| PipelineError::Forward(format!("mtp graph hidden0: {e}")))?;
        let h1 = row(&self.state.hidden, 1)?
            .reshape(vec![1usize, 1, self.hidden_size])
            .map_err(|e| PipelineError::Forward(format!("mtp graph hidden1: {e}")))?;
        Ok(VerifyStep {
            logits0: row(&self.state.logits, 0)?,
            logits1: row(&self.state.logits, 1)?,
            hidden0: h0,
            hidden1: h1,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("load: {0}")]
    Load(String),
    #[error("model: {0}")]
    Model(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
    #[error("forward: {0}")]
    Forward(String),
}

impl From<ModelError> for PipelineError {
    fn from(e: ModelError) -> Self {
        Self::Model(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn bundle_path() -> Option<PathBuf> {
        let p = PathBuf::from("models/qwen3.6 27B.syn");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    fn nvfp4_precision() -> PrecisionConfig {
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::NVFP4,
            mlp_w: DType::NVFP4,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        }
    }

    #[test]
    fn cuda_nvfp4_generates() {
        if std::env::var("SYN_QWEN_NEXT_CUDA").is_err() {
            return;
        }
        let Some(path) = bundle_path() else { return };
        synaptix_kernels_cpu::ensure_registered();
        synaptix_kernels_cuda::ensure_registered();
        let p = HybridPipeline::load_with_precision(&path, Device::Cuda(0), nvfp4_precision(), Some(2048))
            .expect("load nvfp4 cuda");
        let prompt = std::env::var("SYN_QWEN_NEXT_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
        let ids = p.encode(&prompt).unwrap();
        let (new_ids, stats) = p
            .generate(
                &ids,
                GenerationConfig {
                    max_new_tokens: 32,
                    temperature: 0.0,
                    max_seq: Some(2048),
                    ..Default::default()
                },
            )
            .unwrap();
        let txt = p.decode(&new_ids).unwrap();
        eprintln!(
            "[qwen3-next cuda nvfp4] ids={new_ids:?}\n  '{txt}'\n  prefill_ms={} decode_ms={}",
            stats.prefill_ms, stats.decode_ms
        );
        assert!(!new_ids.is_empty());
    }

    fn nvfp4_f16_precision() -> PrecisionConfig {
        PrecisionConfig {
            compute: DType::F16,
            attn_w: DType::NVFP4,
            mlp_w: DType::NVFP4,
            lm_head: DType::F16,
            embed: DType::F16,
            kv: DType::F16,
        }
    }

    /// Сверка CUDA-graph decode против host-reference (та же модель, greedy):
    /// токены должны совпасть (с точностью до F16-compute). Также печатает
    /// tok/s обоих путей. Gated на `SYN_QWEN_NEXT_GRAPH` + наличие бандла.
    #[test]
    fn cuda_graph_matches_host() {
        if std::env::var("SYN_QWEN_NEXT_GRAPH").is_err() {
            return;
        }
        let Some(path) = bundle_path() else { return };
        synaptix_kernels_cpu::ensure_registered();
        synaptix_kernels_cuda::ensure_registered();
        let p = HybridPipeline::load_with_precision(&path, Device::Cuda(0), nvfp4_f16_precision(), Some(2048))
            .expect("load nvfp4 f16 cuda");
        let prompt = std::env::var("SYN_QWEN_NEXT_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
        let ids = p.encode(&prompt).unwrap();
        let cfg = GenerationConfig {
            max_new_tokens: 96,
            temperature: 0.0,
            max_seq: Some(2048),
            ..Default::default()
        };

        let (host_ids, hstats) = p.generate(&ids, cfg.clone()).expect("host generate");
        let host_tps = host_ids.len() as f64 / (hstats.decode_ms.max(1) as f64 / 1000.0);
        let (graph_ids, gstats) = p.generate_with_graph(&ids, cfg).expect("graph generate");
        // decode_ms графа — чистый replay (capture не входит).
        let graph_tps = graph_ids.len() as f64 / (gstats.decode_ms.max(1) as f64 / 1000.0);

        eprintln!("[host ] decode_ms={} ({host_tps:.1} tok/s)\n  '{}'",
            hstats.decode_ms, p.decode(&host_ids).unwrap());
        eprintln!("[graph] decode_ms={} ({graph_tps:.1} tok/s replay)\n  '{}'",
            gstats.decode_ms, p.decode(&graph_ids).unwrap());

        // F16-graph vs F32-host greedy: префикс совпадает, дальше дрейфует
        // (как в qwen3-графе). Корректность = разумный префикс + связный вывод.
        let common = host_ids.iter().zip(&graph_ids).take_while(|(a, b)| a == b).count();
        eprintln!("[match] {common}/{} токенов префикса совпали; graph speedup ×{:.2}",
            host_ids.len(), graph_tps / host_tps.max(1e-6));
        assert!(!graph_ids.is_empty());
        assert!(common >= 8, "graph разошёлся слишком рано (токен {common}) — вероятен баг, не дрейф");
    }

    #[test]
    fn pipeline_generates_greedy() {
        if std::env::var("SYN_QWEN_NEXT_GENERATE").is_err() {
            return;
        }
        let Some(path) = bundle_path() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = HybridPipeline::load(&path, Device::Cpu, DType::BF16).expect("load");
        let prompt = std::env::var("SYN_QWEN_NEXT_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
        let ids = p.encode(&prompt).unwrap();
        let (new_ids, stats) = p
            .generate(
                &ids,
                GenerationConfig {
                    max_new_tokens: 16,
                    temperature: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
        let txt = p.decode(&new_ids).unwrap();
        eprintln!(
            "[qwen3-next cpu bf16] ids={new_ids:?}\n  '{txt}'\n  prefill_ms={} decode_ms={}",
            stats.prefill_ms, stats.decode_ms
        );
        assert!(!new_ids.is_empty());
    }
}
