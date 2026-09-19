//! [`SdxlModel`] — держатель четырёх нейрокомпонентов SDXL и двух CLIP-
//! токенайзеров, плюс кодирование промпта в conditioning для UNet.
//!
//! Источник — HF-каталог или `.syn`-бандл ([`crate::source`]). Раскладка
//! каталога `stable-diffusion-xl-base-1.0/`:
//!   text_encoder/   model.fp16.safetensors           CLIP-L  (768, quick_gelu)
//!   text_encoder_2/ model.fp16.safetensors           bigG    (1280, gelu, +proj)
//!   unet/           diffusion_pytorch_model.fp16...   UNet2DConditionModel
//!   vae/            diffusion_pytorch_model.fp16...   AutoencoderKL
//!   tokenizer/ tokenizer_2/  vocab.json + merges.txt  (CLIP BPE)

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_nn::linear::Linear;
use synaptix_nn::text::{ClipTextConfig, ClipTextEncoder};
use synaptix_nn::unet::{UNet2DConditionConfig, UNet2DConditionModel};
use synaptix_nn::vae::{AutoencoderKlConfig, KlVae};

use crate::source::{self, SdxlSource};
use crate::tokenizer::ClipTokenizer;
use crate::SdxlError;

pub const MAX_TOKENS: usize = 77;

pub struct SdxlModel {
    pub clip_l: ClipTextEncoder,
    pub clip_g: ClipTextEncoder,
    pub unet: UNet2DConditionModel,
    pub vae: KlVae,
    pub tok_l: ClipTokenizer,
    pub tok_g: ClipTokenizer,
    pub device: Device,
    pub dtype: DType,
    /// dtype VAE-декода. = `dtype`, но F16→BF16 (F16 переполняется в VAE; BF16
    /// имеет F32-диапазон экспоненты → безопасен и быстрее F32).
    pub vae_dtype: DType,
}

/// Conditioning для одного шага UNet с включённым CFG (батч = 2: uncond, cond).
pub struct PromptEmbeds {
    /// `[2, 77, 2048]` — конкатенация penultimate CLIP-L (768) и bigG (1280).
    pub encoder_hidden_states: Tensor,
    /// `[2, 1280]` — спроецированный pooled bigG (`add_text_embeds`).
    pub pooled: Tensor,
}

impl SdxlModel {
    /// Грузит все компоненты из HF-директории SDXL. UNet/CLIP — в `dtype`,
    /// VAE — в `vae_dtype` (= dtype, но F16→BF16: F16 переполняется).
    pub fn from_pretrained(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, SdxlError> {
        Self::from_pretrained_quant(dir, device, dtype, DType::BF16)
    }

    /// Как [`from_pretrained`], но с квантованием весов UNet (`quant` = NVFP4/MXFP8;
    /// BF16/F16/F32 = dense). Квант режет VRAM UNet и считается в F16-активации
    /// (`dtype` должен быть F16 при quant). CLIP/VAE остаются dense.
    pub fn from_pretrained_quant(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        quant: DType,
    ) -> Result<Self, SdxlError> {
        let src = SdxlSource::open(dir)?;
        let (tok_l, tok_g) = tokenizers(&src)?;

        let clip_l = {
            let w = src.weights(source::TEXT_ENCODER)?;
            ClipTextEncoder::load(&ClipTextConfig::clip_l(), "text_model", &|n| w.get(n, device, dtype))?
        };

        let clip_g = {
            let w = src.weights(source::TEXT_ENCODER_2)?;
            let get = |n: &str| w.get(n, device, dtype);
            let enc = ClipTextEncoder::load(&ClipTextConfig::clip_bigg(), "text_model", &get)?;
            let proj = Linear::new(get("text_projection.weight")?, None)?;
            enc.with_projection(proj)
        };

        let unet = {
            let w = src.weights(source::UNET)?;
            // Точность весов UNet: quant квантует attn/GEGLU-линейки (resnet/conv dense).
            synaptix_nn::unet::unet_2d_condition::set_unet_precision(quant, dtype);
            let unet = UNet2DConditionModel::load(&UNet2DConditionConfig::sdxl(), &|n| w.get(n, device, dtype));
            synaptix_nn::unet::unet_2d_condition::set_unet_precision(DType::BF16, DType::BF16);
            unet?
        };

        // VAE: F16 переполняется (5-бит экспонента) → BF16 (F32-range, быстрее F32).
        let vae_dtype = if dtype == DType::F16 { DType::BF16 } else { dtype };
        let vae = {
            let w = src.weights(source::VAE)?;
            KlVae::load(&AutoencoderKlConfig::sdxl(), &|n| w.get(n, device, vae_dtype))?
        };

        Ok(Self { clip_l, clip_g, unet, vae, tok_l, tok_g, device, dtype, vae_dtype })
    }

    /// Токенизирует один текст обоими токенайзерами в `[1, 77]` u32-тензоры.
    fn token_ids(&self, text: &str) -> Result<(Tensor, Tensor), SdxlError> {
        let ids_l = self.tok_l.encode(text, MAX_TOKENS);
        let ids_g = self.tok_g.encode(text, MAX_TOKENS);
        let t_l = Tensor::from_vec(ids_l, (1, MAX_TOKENS), self.device)?;
        let t_g = Tensor::from_vec(ids_g, (1, MAX_TOKENS), self.device)?;
        Ok((t_l, t_g))
    }

    /// Кодирует (negative, prompt) → батч-2 conditioning. Порядок строк
    /// (uncond, cond) совпадает с раскладкой CFG (`split_batch_2`).
    pub fn encode_prompt(
        &self,
        prompt: &str,
        negative: &str,
    ) -> Result<PromptEmbeds, SdxlError> {
        let (uncond_l, uncond_g) = self.token_ids(negative)?;
        let (cond_l, cond_g) = self.token_ids(prompt)?;

        let ids_l = Tensor::cat(&[&uncond_l, &cond_l], 0)?;
        let ids_g = Tensor::cat(&[&uncond_g, &cond_g], 0)?;

        let out_l = self.clip_l.forward(&ids_l)?;
        let out_g = self.clip_g.forward(&ids_g)?;

        let ehs = Tensor::cat(
            &[out_l.penultimate_hidden_state(), out_g.penultimate_hidden_state()],
            2,
        )?
        .contiguous()?;
        let pooled = out_g.pooled_output.clone();

        Ok(PromptEmbeds { encoder_hidden_states: ehs, pooled })
    }
}

/// Оба CLIP-токенайзера из источника (`tokenizer/`, `tokenizer_2/`).
pub fn tokenizers(src: &SdxlSource) -> Result<(ClipTokenizer, ClipTokenizer), SdxlError> {
    let one = |d: &str| -> Result<ClipTokenizer, SdxlError> {
        ClipTokenizer::from_parts(
            &src.read(&format!("{d}/vocab.json"))?,
            &src.read(&format!("{d}/merges.txt"))?,
            src.read_opt(&format!("{d}/tokenizer_config.json")).as_deref(),
        )
    };
    Ok((one("tokenizer")?, one("tokenizer_2")?))
}
