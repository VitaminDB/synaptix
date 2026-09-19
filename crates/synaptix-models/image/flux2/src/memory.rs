//! Память FLUX.2: замеры VRAM/RAM, возврат пулов драйверу, запасы.
//!
//! Цель — вся линейка (dev 32B + Mistral 24B, klein 4B/9B) на карте от
//! ~7 ГБ: что не влезло в VRAM, стримится — блоки DiT из пиннованной копии
//! на хосте или прямо из mmap-источника, слои энкодера только из источника.

use synaptix_core::device::cuda::WeightsAllocGuard;
use synaptix_core::device::Device;

/// Свободная VRAM устройства (для CPU — `usize::MAX`).
pub fn free_vram(device: Device) -> usize {
    match device {
        Device::Cuda(ord) => synaptix_core::device::cuda::mem_info(ord).map(|(f, _)| f).unwrap_or(0),
        _ => usize::MAX,
    }
}

/// Вернуть драйверу свободное во ВСЕХ пулах устройства (веса без
/// [`WeightsAllocGuard`] и активации живут в пуле активаций, который сам
/// ничего не отдаёт).
pub fn release_pools(device: Device) {
    if let Device::Cuda(ord) = device {
        let _ = synaptix_core::device::cuda::synchronize_all(ord);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(ord);
    }
}

/// Веса — в weights-пул; на выходе staging загрузки отдаётся драйверу.
pub fn weights_guard(device: Device) -> WeightsAllocGuard {
    WeightsAllocGuard::for_device(device)
}

/// `MemAvailable` из `/proc/meminfo` (байты); `None`, если не прочитать.
pub fn host_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    s.lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

/// Запас VRAM под рабочий стол: карта одна на модель и композитор, при
/// сотнях свободных мегабайт KWin начинает ронять atomic commit.
pub const DESKTOP_MARGIN: usize = 1_000_000_000;

/// Оценка пиковых активаций DiT на прогон в `tokens` токенов (текст +
/// латент + референсы): самый широкий тензор — выход `to_qkv_mlp_proj`
/// single-блока `[S, 3·inner + 2·mlp]` плюс его куски и конкатенация.
pub fn dit_activation_bytes(tokens: usize, inner: usize, mlp_hidden: usize) -> usize {
    let row = 3 * inner + 2 * mlp_hidden; // выход fused-проекции
    let per_token = 2 * (2 * row + (inner + mlp_hidden) + 6 * inner);
    tokens * per_token + 256 * (1 << 20)
}
