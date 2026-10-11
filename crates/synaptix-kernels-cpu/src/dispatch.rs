pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avx_vnni: bool,
    pub avx512_vnni: bool,
    pub neon: bool,
    pub dotprod: bool,
    pub i8mm: bool,
    pub fp16: bool,
    pub sve: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        Self {
            avx2: x86("avx2"),
            avx512f: x86("avx512f"),
            fma: x86("fma"),
            f16c: x86("f16c"),
            avx_vnni: x86("avxvnni"),
            avx512_vnni: x86("avx512vnni"),
            neon: cfg!(target_arch = "aarch64"),
            dotprod: arm("dotprod"),
            i8mm: arm("i8mm"),
            fp16: arm("fp16"),
            sve: arm("sve"),
        }
    }

    pub fn best_gemm_kernel(&self) -> &'static str {
        if self.avx512f {
            "avx512"
        } else if self.avx2 {
            "avx2"
        } else if self.neon {
            "neon"
        } else {
            "naive"
        }
    }

    pub fn int8_dot(&self) -> &'static str {
        if self.avx512_vnni {
            "avx512-vnni"
        } else if self.avx_vnni {
            "avx-vnni"
        } else if self.i8mm {
            "i8mm"
        } else if self.dotprod {
            "dotprod"
        } else if self.avx2 {
            "avx2"
        } else if self.neon {
            "neon"
        } else {
            "scalar"
        }
    }

    pub fn summary(&self) -> String {
        let mut v = Vec::new();
        if self.avx2 {
            v.push("avx2");
        }
        if self.fma {
            v.push("fma");
        }
        if self.f16c {
            v.push("f16c");
        }
        if self.avx_vnni {
            v.push("avx-vnni");
        }
        if self.avx512f {
            v.push("avx512f");
        }
        if self.avx512_vnni {
            v.push("avx512-vnni");
        }
        if self.neon {
            v.push("neon");
        }
        if self.fp16 {
            v.push("fp16");
        }
        if self.dotprod {
            v.push("dotprod");
        }
        if self.i8mm {
            v.push("i8mm");
        }
        if self.sve {
            v.push("sve");
        }
        v.join(" ")
    }
}

fn x86(feature: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        match feature {
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            "fma" => std::arch::is_x86_feature_detected!("fma"),
            "f16c" => std::arch::is_x86_feature_detected!("f16c"),
            "avxvnni" => std::arch::is_x86_feature_detected!("avxvnni"),
            "avx512vnni" => std::arch::is_x86_feature_detected!("avx512vnni"),
            _ => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = feature;
        false
    }
}

fn arm(feature: &str) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        match feature {
            "dotprod" => std::arch::is_aarch64_feature_detected!("dotprod"),
            "i8mm" => std::arch::is_aarch64_feature_detected!("i8mm"),
            "fp16" => std::arch::is_aarch64_feature_detected!("fp16"),
            "sve" => std::arch::is_aarch64_feature_detected!("sve"),
            _ => false,
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = feature;
        false
    }
}

static FEATURES: std::sync::OnceLock<CpuFeatures> = std::sync::OnceLock::new();
pub fn cpu_features() -> &'static CpuFeatures {
    FEATURES.get_or_init(CpuFeatures::detect)
}
