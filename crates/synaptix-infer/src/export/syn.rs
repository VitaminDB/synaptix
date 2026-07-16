use std::collections::HashMap;
use std::path::Path;

use synaptix_bundle::BundleBuilder;
use synaptix_core::tensor::Tensor;

use crate::error::{InferError, Result};

/// Записать тензоры в `.syn` бандл через [`BundleBuilder`].
///
/// Тензоры сериализуются в safetensors и кладутся как единый `tensors:main`
/// чанк (через `add_tensors_from_safetensors`). Прочитать обратно можно
/// `Bundle::open(path)?.tensors_slice()` → `safetensors::SafeTensors::deserialize`.
pub fn export_syn(tensors: &HashMap<String, Tensor>, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let bytes = crate::export::safetensors::serialize_tensors(tensors)?;

    // Промежуточный safetensors рядом с выходом — детерминированное имя на
    // выходной путь, без гонок между разными экспортами.
    let staged = path.with_extension("export_stage.safetensors");
    std::fs::write(&staged, &bytes)
        .map_err(|e| InferError::Other(format!("write staged safetensors: {e}")))?;

    let result = BundleBuilder::new("synaptix-export", "1.0.0")
        .add_tensors_from_safetensors(&staged)
        .write(path)
        .map_err(|e| InferError::Other(format!("bundle write: {e}")));

    let _ = std::fs::remove_file(&staged);
    result
}
