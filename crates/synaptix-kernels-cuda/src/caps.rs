//! Возможности карты и цель NVRTC-компиляции.
//!
//! До 22.09.2026 движок не спрашивал у карты ничего, кроме числа SM: каждое
//! ядро компилировалось под строку-константу (`sm_120a` у GEMM/GEMV, `sm_80`
//! у квант-ядер), и на карте без Blackwell модуль просто не собирался, роняя
//! загрузку модели. Здесь — один запрос атрибутов на контекст и вывод цели
//! компиляции из compute capability:
//!
//! | cc         | цель       | что это даёт                                              |
//! |------------|------------|-----------------------------------------------------------|
//! | 8.0        | `sm_80`    | mma.bf16, cp.async, ldmatrix, smem opt-in 163 КБ         |
//! | 8.6 / 8.7  | `sm_86/87` | то же, smem opt-in 99 КБ                                  |
//! | 8.9        | `sm_89`    | + fp8 mma (`e4m3`, без block_scale) и аппаратный cvt fp8  |
//! | 9.0        | `sm_90a`   | + TMA, mbarrier.try_wait, stmatrix, setmaxnreg            |
//! | 10.x       | `sm_10Xa`  | + block-scale mma (`kind::mxf4nvf4`, `kind::mxf8f6f4`)    |
//! | 12.x       | `sm_12Xa`  | то же для потребительского Blackwell                      |
//!
//! Суффикс `a` — «архитектурно-специфичные» инструкции; PTX под `sm_90a` и
//! выше не переносится на другие карты, поэтому ставится только когда цель
//! совпадает с картой. Без `a` (`SYN_FORCE_ARCH=sm_120`) block-scale MMA и
//! setmaxnreg недоступны — так проверяется «портируемый» путь на Blackwell.
//!
//! **Эмуляция старых карт.** `SYN_FORCE_ARCH=sm_80` (или `sm_86`, `sm_89`,
//! `sm_90`, `compute_80`) компилирует всё под указанную цель: PTX совместим
//! вперёд, драйвер JIT'ит его под реальную карту, а инструкции новее цели
//! NVRTC не пропустит — это и есть гарантия, что ядро пойдёт на sm_80.
//! Скорость при этом не показательна. Цель выше карты игнорируется с
//! предупреждением.
//!
//! Каждому модулю в опции NVRTC добавляются `-DSYN_CC=<major*10+minor>` и
//! `-DSYN_ARCH_A=0|1`, чтобы один `.cu` мог держать и портируемые ядра, и
//! ветку под `#if SYN_ARCH_A && SYN_CC >= 90` (TMA-семейство `gemm_bf16.cu`).

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys::CUdevice_attribute as Attr;
use cudarc::driver::CudaContext;
use synaptix_core::error::{Result, SynaptixError};

/// Возможность, которой требует ядро. Модуль, компилируемый через
/// [`crate::kernels::compile::compile_module_req`], получает `Unsupported`
/// вместо лога NVRTC, если карта (или принудительная цель) её не имеет.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// `mma.sync.kind::mxf4nvf4.block_scale` — NVFP4 на тензорных ядрах.
    Fp4Mma,
    /// `mma.sync.kind::mxf8f6f4.block_scale` — MXFP8 на тензорных ядрах.
    Mxfp8Mma,
    /// `mma.sync … e4m3` без block_scale (Ada/Hopper и выше).
    Fp8Mma,
    /// Аппаратный `cvt` fp8↔f16 (sm_89+); ниже — программная развёртка.
    Fp8Cvt,
    /// TMA (`cp.async.bulk.tensor`), `mbarrier.try_wait`, `stmatrix` — sm_90+.
    Sm90,
    /// `setmaxnreg` и прочее из `sm_90a`/`sm_100a`/`sm_120a`.
    ArchAccel,
}

impl Feature {
    pub fn describe(self) -> &'static str {
        match self {
            Feature::Fp4Mma => "FP4 block-scale MMA (sm_100a/sm_120a)",
            Feature::Mxfp8Mma => "MXFP8 block-scale MMA (sm_100a/sm_120a)",
            Feature::Fp8Mma => "FP8 MMA (sm_89+)",
            Feature::Fp8Cvt => "аппаратный cvt FP8 (sm_89+)",
            Feature::Sm90 => "TMA/mbarrier.try_wait (sm_90+)",
            Feature::ArchAccel => "архитектурно-специфичный PTX (sm_90a+)",
        }
    }
}

/// Что умеет карта — физически и с учётом принудительной цели.
#[derive(Debug, Clone)]
pub struct DeviceCaps {
    pub ordinal: usize,
    pub name: String,
    /// Физическая compute capability.
    pub cc: (u32, u32),
    /// Эффективная: `min(физическая, SYN_FORCE_ARCH)`. По ней считаются
    /// возможности и цель компиляции.
    pub eff_cc: (u32, u32),
    /// Цель включает `a`-суффикс (архитектурно-специфичные инструкции).
    pub accel: bool,
    pub sm_count: u32,
    /// Максимум динамической shared memory на блок (opt-in), байт — с учётом
    /// лимита эмулируемой архитектуры.
    pub smem_optin_bytes: u32,
    pub total_mem_bytes: u64,
    /// Цель NVRTC (`--gpu-architecture=`).
    pub arch: &'static str,
    /// Цель принудительная (`SYN_FORCE_ARCH`).
    pub forced: bool,
}

impl DeviceCaps {
    pub fn fp4_mma(&self) -> bool {
        self.accel && matches!(self.eff_cc.0, 10 | 12)
    }
    pub fn mxfp8_mma(&self) -> bool {
        self.fp4_mma()
    }
    pub fn fp8_mma(&self) -> bool {
        self.eff_cc >= (8, 9)
    }
    pub fn fp8_cvt(&self) -> bool {
        self.eff_cc >= (8, 9)
    }
    pub fn sm90(&self) -> bool {
        self.eff_cc >= (9, 0)
    }
    pub fn bf16_mma(&self) -> bool {
        self.eff_cc >= (8, 0)
    }
    pub fn dp4a(&self) -> bool {
        self.eff_cc >= (6, 1)
    }
    pub fn has(&self, f: Feature) -> bool {
        match f {
            Feature::Fp4Mma => self.fp4_mma(),
            Feature::Mxfp8Mma => self.mxfp8_mma(),
            Feature::Fp8Mma => self.fp8_mma(),
            Feature::Fp8Cvt => self.fp8_cvt(),
            Feature::Sm90 => self.sm90(),
            Feature::ArchAccel => self.accel && self.eff_cc >= (9, 0),
        }
    }

    /// `Unsupported`, если возможности нет. Сообщение статическое (враппер
    /// `SynaptixError::Unsupported` держит `&'static str`), детали — в trace.
    pub fn require(&self, f: Feature) -> Result<()> {
        if self.has(f) {
            return Ok(());
        }
        tracing::debug!(
            "ядро требует {}: карта {} ({}), цель {}",
            f.describe(),
            self.name,
            cc_str(self.cc),
            self.arch
        );
        Err(SynaptixError::Unsupported(match f {
            Feature::Fp4Mma => "ядро требует FP4 block-scale MMA (sm_100a/sm_120a)",
            Feature::Mxfp8Mma => "ядро требует MXFP8 block-scale MMA (sm_100a/sm_120a)",
            Feature::Fp8Mma => "ядро требует FP8 MMA (sm_89+)",
            Feature::Fp8Cvt => "ядро требует аппаратный cvt FP8 (sm_89+)",
            Feature::Sm90 => "ядро требует TMA/mbarrier.try_wait (sm_90+)",
            Feature::ArchAccel => "ядро требует архитектурно-специфичный PTX (sm_90a+)",
        }))
    }

    /// Опции NVRTC, описывающие цель для препроцессора `.cu`.
    pub fn defines(&self) -> [String; 2] {
        [
            format!("-DSYN_CC={}", self.eff_cc.0 * 10 + self.eff_cc.1),
            format!("-DSYN_ARCH_A={}", u8::from(self.accel)),
        ]
    }

    /// Нативно ли исполняется квант-формат весов на этой карте (GEMM/GEMV
    /// тензорными ядрами). Иначе движок идёт путём «деквант → плотный GEMM».
    pub fn quant_native(&self, dtype: synaptix_core::dtype::DType) -> bool {
        use synaptix_core::dtype::DType;
        match dtype {
            DType::NVFP4 => self.fp4_mma(),
            DType::MXFP8 => self.mxfp8_mma(),
            _ => false,
        }
    }

    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} — sm_{}{}, {} SM, smem {} КБ, VRAM {:.1} ГБ, цель {}",
            self.name,
            self.cc.0,
            self.cc.1,
            self.sm_count,
            self.smem_optin_bytes / 1024,
            self.total_mem_bytes as f64 / (1u64 << 30) as f64,
            self.arch
        );
        if self.forced {
            s.push_str(" (SYN_FORCE_ARCH)");
        }
        let flags: Vec<&str> = [
            (self.fp4_mma(), "fp4-mma"),
            (self.mxfp8_mma(), "mxfp8-mma"),
            (self.fp8_mma(), "fp8-mma"),
            (self.fp8_cvt(), "fp8-cvt"),
            (self.sm90(), "tma"),
            (self.bf16_mma(), "bf16-mma"),
            (self.dp4a(), "dp4a"),
        ]
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, n)| *n)
        .collect();
        s.push_str("; ");
        s.push_str(&flags.join(" "));
        s
    }
}

fn cc_str(cc: (u32, u32)) -> String {
    format!("{}.{}", cc.0, cc.1)
}

/// Цель NVRTC по compute capability. Неизвестная минорная версия — ближайшая
/// известная ниже в том же поколении; ниже 8.0 — `sm_75` (движок туда не
/// целится: mma.bf16 требует sm_80).
pub fn arch_for(cc: (u32, u32), accel: bool) -> &'static str {
    match (cc, accel) {
        ((12, 1), true) => "sm_121a",
        ((12, 1), false) => "sm_121",
        ((12, _), true) => "sm_120a",
        ((12, _), false) => "sm_120",
        ((10, 3), true) => "sm_103a",
        ((10, 3), false) => "sm_103",
        ((10, 1), true) => "sm_101a",
        ((10, 1), false) => "sm_101",
        ((10, _), true) => "sm_100a",
        ((10, _), false) => "sm_100",
        ((9, _), true) => "sm_90a",
        ((9, _), false) => "sm_90",
        ((8, 9), _) => "sm_89",
        ((8, 7), _) => "sm_87",
        ((8, 6), _) => "sm_86",
        ((8, _), _) => "sm_80",
        _ => "sm_75",
    }
}

/// Opt-in лимит shared memory на блок по архитектуре (байт).
fn smem_limit_for(cc: (u32, u32)) -> u32 {
    match cc {
        (8, 0) => 163 * 1024,
        (8, 6) | (8, 7) | (8, 9) => 99 * 1024,
        (9, _) | (10, _) => 227 * 1024,
        (12, _) => 99 * 1024,
        _ => 96 * 1024,
    }
}

/// Разбор `SYN_FORCE_ARCH`: `sm_80`, `sm_90a`, `compute_86`, `86`, `8.6`.
/// Возвращает (cc, accel).
pub fn parse_force_arch(s: &str) -> Option<((u32, u32), bool)> {
    let t = s.trim().to_ascii_lowercase();
    let t = t
        .strip_prefix("sm_")
        .or_else(|| t.strip_prefix("compute_"))
        .unwrap_or(&t);
    let (digits, accel) = match t.strip_suffix('a') {
        Some(d) => (d, true),
        None => (t, false),
    };
    if let Some((a, b)) = digits.split_once('.') {
        return Some(((a.parse().ok()?, b.parse().ok()?), accel));
    }
    let n: u32 = digits.parse().ok()?;
    if n < 10 {
        return None;
    }
    Some(((n / 10, n % 10), accel))
}

fn force_arch() -> Option<((u32, u32), bool)> {
    static F: OnceLock<Option<((u32, u32), bool)>> = OnceLock::new();
    *F.get_or_init(|| {
        let v = std::env::var("SYN_FORCE_ARCH").ok()?;
        if v.trim().is_empty() {
            return None;
        }
        match parse_force_arch(&v) {
            Some(x) => Some(x),
            None => {
                eprintln!("[synaptix] SYN_FORCE_ARCH='{v}' не разобран (ожидается sm_80 / sm_90a / compute_86); игнорирую");
                None
            }
        }
    })
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<DeviceCaps>)>>> = OnceLock::new();

/// Число CUDA-карт; 0 — драйвера нет.
pub fn device_count() -> usize {
    CudaContext::device_count().map(|n| n.max(0) as usize).unwrap_or(0)
}

impl DeviceCaps {
    /// Возможности карты контекста; один запрос на контекст, дальше кэш.
    pub fn for_context(ctx: &Arc<CudaContext>) -> Arc<DeviceCaps> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock().unwrap();
            if let Some((_, c)) = g.iter().find(|(k, _)| *k == key) {
                return c.clone();
            }
        }
        let caps = Arc::new(Self::query(ctx));
        let mut g = cache.lock().unwrap();
        if let Some((_, c)) = g.iter().find(|(k, _)| *k == key) {
            return c.clone();
        }
        g.push((key, caps.clone()));
        tracing::info!("CUDA {}: {}", caps.ordinal, caps.summary());
        caps
    }

    /// По порядковому номеру карты (через реестр контекстов ядра).
    pub fn for_ordinal(ordinal: usize) -> Result<Arc<DeviceCaps>> {
        let ctx = synaptix_core::device::cuda::get(ordinal)?;
        Ok(Self::for_context(&ctx))
    }

    fn query(ctx: &Arc<CudaContext>) -> DeviceCaps {
        let attr = |a: Attr| ctx.attribute(a).unwrap_or(0).max(0) as u32;
        let cc = (
            attr(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            attr(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        );
        let sm_count = attr(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        let smem_dev = attr(Attr::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
        let name = ctx.name().unwrap_or_else(|_| format!("cuda:{}", ctx.ordinal()));
        let total_mem_bytes = ctx.total_mem().unwrap_or(0) as u64;
        let (eff_cc, accel, forced) = match force_arch() {
            Some((f, a)) if f <= cc => (f, a, true),
            Some((f, _)) => {
                eprintln!(
                    "[synaptix] SYN_FORCE_ARCH=sm_{}{} выше карты (sm_{}{}); PTX под неё не загрузится — игнорирую",
                    f.0, f.1, cc.0, cc.1
                );
                (cc, true, false)
            }
            None => (cc, true, false),
        };
        // `a`-суффикс есть только у sm_90 и выше; ниже флаг ничего не значит.
        let accel = accel && eff_cc >= (9, 0);
        let smem_optin_bytes = smem_dev.min(smem_limit_for(eff_cc)).max(48 * 1024);
        DeviceCaps {
            ordinal: ctx.ordinal(),
            name,
            cc,
            eff_cc,
            accel,
            sm_count,
            smem_optin_bytes,
            total_mem_bytes,
            arch: arch_for(eff_cc, accel),
            forced,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(eff: (u32, u32), accel: bool) -> DeviceCaps {
        DeviceCaps {
            ordinal: 0,
            name: "t".into(),
            cc: (12, 0),
            eff_cc: eff,
            accel: accel && eff >= (9, 0),
            sm_count: 1,
            smem_optin_bytes: 99 * 1024,
            total_mem_bytes: 0,
            arch: arch_for(eff, accel),
            forced: true,
        }
    }

    #[test]
    fn arch_strings() {
        assert_eq!(arch_for((12, 0), true), "sm_120a");
        assert_eq!(arch_for((12, 0), false), "sm_120");
        assert_eq!(arch_for((9, 0), true), "sm_90a");
        assert_eq!(arch_for((8, 9), true), "sm_89");
        assert_eq!(arch_for((8, 6), false), "sm_86");
        assert_eq!(arch_for((8, 0), true), "sm_80");
        assert_eq!(arch_for((10, 0), true), "sm_100a");
        assert_eq!(arch_for((7, 5), true), "sm_75");
    }

    #[test]
    fn force_arch_parsing() {
        assert_eq!(parse_force_arch("sm_80"), Some(((8, 0), false)));
        assert_eq!(parse_force_arch("SM_90a"), Some(((9, 0), true)));
        assert_eq!(parse_force_arch("compute_86"), Some(((8, 6), false)));
        assert_eq!(parse_force_arch("8.9"), Some(((8, 9), false)));
        assert_eq!(parse_force_arch("120a"), Some(((12, 0), true)));
        assert_eq!(parse_force_arch("x"), None);
        assert_eq!(parse_force_arch("8"), None);
    }

    #[test]
    fn features_by_arch() {
        let b = caps((12, 0), true);
        assert!(b.fp4_mma() && b.mxfp8_mma() && b.fp8_mma() && b.sm90() && b.dp4a());
        let b_plain = caps((12, 0), false);
        assert!(!b_plain.fp4_mma() && b_plain.fp8_mma() && b_plain.sm90());
        let h = caps((9, 0), true);
        assert!(!h.fp4_mma() && h.fp8_mma() && h.sm90() && h.has(Feature::ArchAccel));
        let ada = caps((8, 9), true);
        assert!(!ada.sm90() && ada.fp8_mma() && ada.fp8_cvt() && ada.bf16_mma());
        let amp = caps((8, 0), true);
        assert!(!amp.fp8_mma() && !amp.fp8_cvt() && amp.bf16_mma() && amp.dp4a());
        assert!(amp.require(Feature::Fp4Mma).is_err());
        assert!(amp.require(Feature::Sm90).is_err());
        assert!(b.require(Feature::Fp4Mma).is_ok());
        assert_eq!(amp.defines(), ["-DSYN_CC=80".to_string(), "-DSYN_ARCH_A=0".to_string()]);
        assert_eq!(b.defines(), ["-DSYN_CC=120".to_string(), "-DSYN_ARCH_A=1".to_string()]);
    }

    #[test]
    fn quant_native_follows_mma() {
        use synaptix_core::dtype::DType;
        assert!(caps((12, 0), true).quant_native(DType::NVFP4));
        assert!(!caps((8, 9), true).quant_native(DType::NVFP4));
        assert!(!caps((8, 9), true).quant_native(DType::MXFP8));
        assert!(!caps((12, 0), true).quant_native(DType::BF16));
    }
}
