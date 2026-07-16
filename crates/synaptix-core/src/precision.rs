use crate::dtype::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionConfig {
    pub compute: DType,
    pub attn_w: DType,
    pub mlp_w: DType,
    pub lm_head: DType,
    pub embed: DType,
    pub kv: DType,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self::dense(DType::BF16)
    }
}

impl PrecisionConfig {
    pub fn dense(compute: DType) -> Self {
        Self {
            compute,
            attn_w: compute,
            mlp_w: compute,
            lm_head: compute,
            embed: compute,
            kv: compute,
        }
    }

    pub fn nvfp4() -> Self {
        Self {
            compute: DType::F16,
            attn_w: DType::NVFP4,
            mlp_w: DType::NVFP4,
            // lm_head NVFP4: на 27B экономит чтение 2.5GB→0.7GB/токен (decode +8%) и
            // 1.8GB VRAM (важно для длинного контекста). Качество держится (greedy
            // связно). Override: --lm-head-dtype f16, если нужна максимальная точность.
            lm_head: DType::NVFP4,
            embed: DType::F16,
            kv: DType::F16,
        }
    }

    /// MXFP8 (Blackwell-нативный block-scale FP8): attn/mlp веса MXFP8, compute
    /// F16. Заменил legacy per-tensor FP8 E4M3 (preset `fp8` теперь = MXFP8).
    pub fn mxfp8() -> Self {
        Self {
            compute: DType::F16,
            attn_w: DType::MXFP8,
            mlp_w: DType::MXFP8,
            lm_head: DType::F16,
            embed: DType::F16,
            kv: DType::F16,
        }
    }

    pub fn from_preset(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "none" | "" => Some(Self::default()),
            "nvfp4" => Some(Self::nvfp4()),
            "fp8" | "mxfp8" => Some(Self::mxfp8()),
            _ => None,
        }
    }

    pub fn any_quantized(&self) -> bool {
        self.attn_w.is_quantized()
            || self.mlp_w.is_quantized()
            || self.lm_head.is_quantized()
            || self.embed.is_quantized()
    }

    /// Квантованные веса требуют F16-активаций (`linear_quant` принимает только
    /// F16). Возвращает понятную ошибку, если compute не F16 при наличии кванта.
    pub fn validate(&self) -> Result<(), String> {
        if self.any_quantized() && self.compute != DType::F16 {
            return Err(format!(
                "quantized weights требуют compute=f16 (сейчас {:?}); используйте --quant nvfp4 или --compute-dtype f16",
                self.compute
            ));
        }
        Ok(())
    }
}

pub fn parse_dtype(s: &str) -> Option<DType> {
    match s.to_ascii_lowercase().as_str() {
        "f32" => Some(DType::F32),
        "bf16" => Some(DType::BF16),
        "f16" => Some(DType::F16),
        "fp8" | "mxfp8" => Some(DType::MXFP8),
        "nvfp4" => Some(DType::NVFP4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_dense_bf16_no_quant() {
        let p = PrecisionConfig::default();
        assert_eq!(p.compute, DType::BF16);
        assert!(!p.any_quantized());
        assert!(p.validate().is_ok());
    }

    #[test]
    fn nvfp4_preset_is_f16_compute_with_quant_weights() {
        let p = PrecisionConfig::nvfp4();
        assert_eq!(p.compute, DType::F16);
        assert_eq!(p.attn_w, DType::NVFP4);
        assert_eq!(p.mlp_w, DType::NVFP4);
        assert!(p.any_quantized());
        assert!(p.validate().is_ok());
    }

    #[test]
    fn quant_with_non_f16_compute_rejected() {
        let mut p = PrecisionConfig::nvfp4();
        p.compute = DType::BF16;
        assert!(p.validate().is_err());
    }

    #[test]
    fn preset_parsing() {
        assert!(PrecisionConfig::from_preset("none").is_some());
        assert!(PrecisionConfig::from_preset("nvfp4").is_some());
        assert!(PrecisionConfig::from_preset("bogus").is_none());
        // defp8: fp8 и mxfp8 пресеты оба → MXFP8-веса (legacy E4M3 убран).
        assert_eq!(PrecisionConfig::from_preset("fp8"), Some(PrecisionConfig::mxfp8()));
        assert_eq!(PrecisionConfig::from_preset("mxfp8"), Some(PrecisionConfig::mxfp8()));
        assert_eq!(PrecisionConfig::mxfp8().attn_w, DType::MXFP8);
        assert_eq!(parse_dtype("nvfp4"), Some(DType::NVFP4));
        assert_eq!(parse_dtype("mxfp8"), Some(DType::MXFP8));
        assert_eq!(parse_dtype("fp8"), Some(DType::MXFP8));
        assert_eq!(parse_dtype("xyz"), None);
    }
}
