pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub neon: bool,
    pub sve: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        Self {
            avx2:   is_x86_feature_detected_safe("avx2"),
            avx512f: is_x86_feature_detected_safe("avx512f"),
            fma:    is_x86_feature_detected_safe("fma"),
            neon:   cfg!(target_arch = "aarch64"),
            sve:    false,
        }
    }

    pub fn best_gemm_kernel(&self) -> &'static str {
        if self.avx512f { "avx512" }
        else if self.avx2 { "avx2" }
        else if self.neon { "neon" }
        else { "naive" }
    }
}

fn is_x86_feature_detected_safe(feature: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        match feature {
            "avx2"    => std::arch::is_x86_feature_detected!("avx2"),
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            "fma"     => std::arch::is_x86_feature_detected!("fma"),
            _         => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

static FEATURES: std::sync::OnceLock<CpuFeatures> = std::sync::OnceLock::new();
pub fn cpu_features() -> &'static CpuFeatures {
    FEATURES.get_or_init(CpuFeatures::detect)
}
