//! Резолв вычислительного устройства для команд инференса + регистрация
//! backend'ов. CUDA — путь по умолчанию (сборка с feature `cuda`/`cutlass`);
//! при отсутствии GPU или сборке без CUDA — graceful fallback на CPU.

use synaptix_core::device::Device;

/// Разбирает `--device` (`cuda` | `cuda:N` | `gpu` | `auto` | `cpu`) и
/// регистрирует нужные backend'ы. CPU backend регистрируется всегда (нужен как
/// fallback и для host-операций). Возвращает выбранное устройство.
pub fn resolve(pref: &str) -> Device {
    synaptix_kernels_cpu::ensure_registered();

    let p = pref.trim().to_ascii_lowercase();
    if p == "cpu" {
        return Device::Cpu;
    }
    let want_cuda = p.is_empty() || p == "auto" || p == "cuda" || p == "gpu" || p.starts_with("cuda:");

    #[cfg(feature = "cuda")]
    {
        if want_cuda {
            let ord = p
                .split_once(':')
                .and_then(|(_, n)| n.parse::<usize>().ok())
                .unwrap_or(0);
            match synaptix_core::device::cuda::get(ord) {
                Ok(_) => {
                    synaptix_kernels_cuda::ensure_registered();
                    return Device::Cuda(ord);
                }
                Err(e) => {
                    eprintln!("synaptix: CUDA device {ord} недоступен ({e}); fallback на CPU");
                }
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        if want_cuda {
            eprintln!("synaptix: бинарь собран без feature `cuda`; используется CPU");
        }
    }

    Device::Cpu
}

/// Разбирает `--attn` (CLI) → `auto` и устанавливает глобальный
/// attention-backend. Неизвестное значение → предупреждение + auto.
pub fn resolve_attn(cli: Option<&str>) {
    use synaptix_core::backend::attn::{set_mode, AttnMode};
    let mode = match cli {
        Some(s) => match AttnMode::parse(s) {
            Some(m) => m,
            None => {
                eprintln!("synaptix: неизвестный --attn '{s}' (auto|flash-decode|fa2|fa4); используется auto");
                AttnMode::Auto
            }
        },
        None => AttnMode::Auto,
    };
    set_mode(mode);
    if mode != AttnMode::Auto {
        eprintln!("synaptix: attention-backend = {}", mode.as_str());
    }
}
