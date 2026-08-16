
use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_llm_common::{
    Activation, DecoderConfig, DecoderModel, LayerKind, NormGain, RopeSpec,
};

use crate::lm::BundleWeightSource;
use crate::loader::CompLoader;
use crate::AceError;

// text2music DiT text-encoder instruction (Python DEFAULT_DIT_INSTRUCTION,
// constants.py:119 / TASK_INSTRUCTIONS["text2music"]). NOT the LM instruction
// "Generate audio semantic tokens…" (that one drives the 5Hz LM, tokenizer.rs).
pub const TASK_INSTRUCTION: &str = "Fill the audio semantic mask based on the given conditions:";

pub fn build_text_prompt(
    caption: &str,
    duration_s: u32,
    bpm: Option<u32>,
    timesig: Option<&str>,
    keyscale: Option<&str>,
) -> String {
    let bpm_s = bpm.map(|b| b.to_string()).unwrap_or_else(|| "N/A".into());
    let timesig_s = timesig.unwrap_or("N/A");
    let keyscale_s = keyscale.unwrap_or("N/A");
    let metas = format!(
        "- bpm: {bpm_s}\n- timesignature: {timesig_s}\n- keyscale: {keyscale_s}\n- duration: {duration_s} seconds\n"
    );
    format!(
        "# Instruction\n{TASK_INSTRUCTION}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n<|endoftext|>"
    )
}

pub fn build_lyric_prompt(lyrics: &str, language: &str) -> String {
    format!("# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|><|endoftext|>")
}

fn qwen3_embedding_config() -> DecoderConfig {
    let head_dim = 128usize;
    DecoderConfig {
        vocab_size: 151669,
        hidden_size: 1024,
        intermediate_size: 3072,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim,
        max_position_embeddings: 32768,
        rms_norm_eps: 1e-6,
        norm_gain: NormGain::Plain,
        activation: Activation::Silu,
        sandwich_norms: false,
        post_norm_eps: None,
        qk_norm: true,
        attn_output_gate: false,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        embed_scale: None,
        embed_rms_norm: false,
        logit_scale: None,
        logit_softcap: None,
        rope_global: RopeSpec { theta: 1_000_000.0, rotary_dim: head_dim, scaled_freqs: None },
        rope_local: None,
        sliding_window: None,
        sliding_window_pattern: 0,
        layer_kinds: vec![LayerKind::Full; 28],
        linear: None,
        tie_word_embeddings: true,
        bos_token_id: Some(151643),
        eos_token_ids: vec![151643],
    }
}

pub struct TextEncoder {
    model: DecoderModel,
    embed: Tensor,
    hidden: usize,
    device: Device,
}

impl TextEncoder {
    pub fn open(
        path: impl AsRef<Path>,
        device: Device,
        compute: DType,
        quant_w: DType,
        rope_capacity: usize,
    ) -> Result<Self, AceError> {
        let path = path.as_ref();
        let embed = CompLoader::open(path, None, device)?.get("embed_tokens.weight", compute)?;
        let hidden = embed.dims()[1];
        let src = BundleWeightSource::new(CompLoader::open(path, None, device)?);
        let dcfg = qwen3_embedding_config();
        let model = DecoderModel::build(
            &dcfg, &src, device, compute, quant_w, quant_w, compute, compute, rope_capacity,
        )
        .map_err(|e| AceError::Load(e.to_string()))?;
        Ok(Self { model, embed, hidden, device })
    }

    pub fn caption_hidden(&self, ids: &Tensor) -> Result<Tensor, AceError> {
        let hs = self
            .model
            .forward_hidden_states(ids, None)
            .map_err(|e| AceError::Other(e.to_string()))?;
        hs.into_iter()
            .last()
            .ok_or_else(|| AceError::Other("forward_hidden_states empty".into()))
    }

    pub fn lyric_embed(&self, ids: &Tensor) -> Result<Tensor, AceError> {
        let d = ids.dims().to_vec();
        let (b, s) = (d[0], d[1]);
        let flat = ids.reshape(vec![b * s])?;
        let e = self.embed.index_select(0, &flat)?;
        Ok(e.reshape(vec![b, s, self.hidden])?)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}
