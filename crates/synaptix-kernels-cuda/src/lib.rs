pub mod attention;
pub mod best_cu;
pub mod comm;
pub mod conv;
pub mod cuda_backend;
pub mod cuda_graph;
pub mod elementwise;
pub mod embed;
pub mod fused;
pub mod gemm;
pub mod kernels;
pub mod nvrtc;
pub mod ptx;
pub mod reduction;
pub mod scan;
pub mod ssm;
pub mod stream_pool;
pub mod tma;
pub mod wsalloc;

pub use cuda_backend::{cuda_backend, ensure_registered};

/// Отдать драйверу device-память, которую CUDA-ядра держат в статиках между
/// моделями: TMA-дескрипторы (ключ — адрес, после выгрузки все записи мертвы)
/// и MXFP8-скретчи. Сами по себе это десятки МБ, но они рассыпаны по сегментам
/// mempool'а: пока хоть одна живая аллокация лежит в сегменте, `cuMemPoolTrimTo`
/// его не вернёт — отсюда «reserved 4.7 ГБ при used 51 МБ» после выгрузки.
/// Вызывать ПЕРЕД тримом пулов. Возвращает (дескрипторов, байт скретчей).
pub fn release_device_caches() -> (usize, usize) {
    let descs = best_cu::gemm::gemm_nvfp4::clear_desc_cache();
    let scratch = gemm::dispatch::clear_mxfp8_scratch()
        + scan::chunk_scan::clear_chunk_scan_ws()
        + attention::linear_prefill::clear_linear_prefill_ws();
    (descs, scratch)
}
