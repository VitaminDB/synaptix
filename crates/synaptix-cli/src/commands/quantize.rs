//! `synaptix quantize` — PTQ конверсия safetensors → нативный квант (NVFP4/MXFP8).
//!
//! Использует `Tensor::quantize_to_nvfp4` / `quantize_to_mxfp8` (block-scale) +
//! `synaptix_bundle::BundleBuilder` для упаковки результата обратно в `.syn`.
//!
//! На MVP подключение ещё не сделано (требует resolve dtype-table в
//! `synaptix-bundle::QUANT_FORMAT_*` и e2e roundtrip-теста на загрузке quantized
//! весов в Qwen3-инференс). Команда возвращает явный exit-код с пояснением.

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "synaptix quantize ещё не подключена в CLI: квантизация в нативные форматы \
         (NVFP4 / MXFP8) есть как Tensor::quantize_to_nvfp4/quantize_to_mxfp8, но e2e-связка \
         с .syn bundle и loader'ом — отдельная задача (см. synaptix-bundle/quantized.rs)"
            .into(),
    )
}
