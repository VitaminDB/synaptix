pub mod qwen35;

use crate::error::{GgufError, Result};
use crate::plan::ConversionPlan;
use crate::reader::GgufFile;

pub fn is_supported(arch: &str) -> bool {
    matches!(arch, "qwen35")
}

pub fn build_plan(
    model: &GgufFile,
    mmproj: Option<&GgufFile>,
    bundle_id: &str,
) -> Result<ConversionPlan> {
    let arch = model.architecture()?.to_string();
    match arch.as_str() {
        "qwen35" => qwen35::build_plan(model, mmproj, bundle_id),
        other => Err(GgufError::UnsupportedArch(other.to_string())),
    }
}
