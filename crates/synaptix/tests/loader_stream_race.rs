//! Гонка «копия на loader-стриме ↔ ядро из потока префетча»: поток префетча
//! весов копирует тензор на карту (loader-стрим, пиннованный staging) и сразу
//! приводит тип. Ядро приведения перенаправляется на compute-стрим и обязано
//! дождаться копии — иначе читает недокопированную память (так 19.09.2026
//! VAE MiniMax-H3 выдал кадр из шума).
//!
//! ```sh
//! cargo test -p synaptix --release --test loader_stream_race -- --nocapture
//! ```

use synaptix_core::device::cuda::{
    default_stream, loader_stream, mem_info, set_alloc_stream, set_offload_pinned,
};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

#[test]
fn prefetch_thread_kernel_waits_for_its_copy() {
    if mem_info(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();

    let n = 64usize << 20; // 256 МБ F32 — копия заметно дольше запуска ядра
    let host: Vec<f32> = (0..n).map(|i| ((i % 997) as f32) * 0.25 - 100.0).collect();
    // Путь загрузчиков весов (`SafetensorsLoader`/`SynBundleLoader::load_to`):
    // сырая H2D-копия из mmap-среза мимо учёта событий cudarc, затем
    // приведение типа на карте.
    let bytes: Vec<u8> = host.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ls = loader_stream(0).expect("loader");

    let mut bad_rounds = 0;
    for round in 0..4 {
        let out = std::thread::scope(|s| {
            s.spawn(|| {
                set_alloc_stream(Some(ls.clone()));
                set_offload_pinned(true);
                let r = Tensor::from_raw_slice(&bytes, vec![n], DType::F32, Device::Cuda(0))
                    .and_then(|t| t.to_dtype(DType::BF16));
                let _ = ls.synchronize();
                set_offload_pinned(false);
                set_alloc_stream(None);
                r
            })
            .join()
            .expect("поток префетча")
        })
        .expect("копия и приведение");
        let _ = default_stream(0).expect("default").synchronize();
        let back = out
            .to_dtype(DType::F32)
            .and_then(|t| t.to_device(Device::Cpu))
            .and_then(|t| t.to_vec1::<f32>())
            .expect("назад");
        // BF16 держит 8 бит мантиссы: |Δ| ≤ |x|·2⁻⁸.
        let wrong = back
            .iter()
            .zip(&host)
            .filter(|(b, h)| (**b - **h).abs() > h.abs() / 256.0 + 1e-3)
            .count();
        eprintln!("[race] прогон {round}: неверных элементов {wrong} из {n}");
        if wrong > 0 {
            bad_rounds += 1;
        }
    }
    assert_eq!(bad_rounds, 0, "ядро из потока префетча читало недокопированный вес");
}
