use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "model_type", rename_all = "snake_case")]
pub enum ModelConfig {
    Llama(super::configs::llama::LlamaConfig),
    Qwen(super::configs::qwen::QwenConfig),
    Qwen36(super::configs::qwen36::Qwen36Config),
    Mistral(super::configs::mistral::MistralConfig),
    Gemma(super::configs::gemma::GemmaConfig),
    Deepseek(super::configs::deepseek::DeepseekConfig),
    Phi(super::configs::phi::PhiConfig),
    Falcon(super::configs::falcon::FalconConfig),
    Baichuan(super::configs::baichuan::BaichuanConfig),
    ChatGlm(super::configs::glm::ChatGlmConfig),
    InternLm(super::configs::internlm::InternLmConfig),
    Yi(super::configs::yi::YiConfig),
    Granite(super::configs::granite::GraniteConfig),
    Olmo(super::configs::olmo::OlmoConfig),
    Mamba(super::configs::mamba::MambaConfig),
    Rwkv(super::configs::rwkv::RwkvConfig),
    Jamba(super::configs::jamba::JambaConfig),
    Hymba(super::configs::hymba::HymbaConfig),
    MiniCpm(super::configs::minicpm::MiniCpmConfig),
    CommandR(super::configs::command_r::CommandRConfig),
}

impl ModelConfig {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
