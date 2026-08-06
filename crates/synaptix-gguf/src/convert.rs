use std::path::{Path, PathBuf};
use std::sync::Arc;

use synaptix_bundle::{BundleBuilder, FileTag, ProgressCallback};

use crate::arch;
use crate::error::{GgufError, Result};
use crate::plan::{ConversionPlan, OutDtype};
use crate::reader::GgufFile;
use crate::tensor_stream::GgufTensorStream;

#[derive(Debug, Clone)]
pub struct ConvertOptions {
    pub dtype: OutDtype,
    pub bundle_id: Option<String>,
    pub mmproj: Option<PathBuf>,
    pub tokenizer_json: Option<PathBuf>,
    pub extra_files: Vec<(String, PathBuf)>,
    pub sha256: bool,
    pub blake3: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            dtype: OutDtype::Auto,
            bundle_id: None,
            mmproj: None,
            tokenizer_json: None,
            extra_files: Vec::new(),
            sha256: false,
            blake3: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConvertReport {
    pub arch: String,
    pub bundle_id: String,
    pub components: Vec<(String, usize)>,
    pub files: Vec<String>,
    pub payload_bytes: u64,
    pub output: PathBuf,
}

pub fn default_bundle_id(model: &Path) -> String {
    model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("gguf-model")
        .to_string()
}

pub fn plan_for(model: &GgufFile, mmproj: Option<&GgufFile>, bundle_id: &str) -> Result<ConversionPlan> {
    arch::build_plan(model, mmproj, bundle_id)
}

pub fn convert_to_syn(
    model_path: &Path,
    out: &Path,
    opts: &ConvertOptions,
    progress: Option<ProgressCallback>,
) -> Result<ConvertReport> {
    let model = Arc::new(GgufFile::open(model_path)?);
    let mmproj = match &opts.mmproj {
        Some(p) => Some(Arc::new(GgufFile::open(p)?)),
        None => None,
    };
    let bundle_id = opts
        .bundle_id
        .clone()
        .unwrap_or_else(|| default_bundle_id(model_path));

    let mut plan = plan_for(&model, mmproj.as_deref(), &bundle_id)?;

    if let Some(p) = &opts.tokenizer_json {
        let bytes = std::fs::read(p)?;
        plan.files.retain(|f| f.path != "tokenizer.json");
        plan.files.push(crate::plan::MappedFile {
            path: "tokenizer.json".into(),
            bytes,
        });
    }

    let mut sources: Vec<Arc<GgufFile>> = vec![model.clone()];
    if let Some(m) = &mmproj {
        sources.push(m.clone());
    }

    let version = model
        .opt_str("general.version")
        .unwrap_or("0.1.0")
        .to_string();
    let mut builder = BundleBuilder::new(plan.bundle_id.clone(), version)
        .arch(plan.arch.clone())
        .purpose("text-generation")
        .with_sha256(opts.sha256)
        .with_blake3(opts.blake3);
    if let Some(cb) = progress {
        builder = builder.with_progress(cb);
    }

    let mut components = Vec::new();
    let mut payload_bytes = 0u64;
    for comp in &plan.components {
        let stream = GgufTensorStream::new(sources.clone(), comp, opts.dtype)?;
        payload_bytes += stream
            .plan()
            .iter()
            .map(|t| t.nbytes())
            .sum::<u64>();
        components.push((comp.name.clone(), comp.tensors.len()));
        builder = builder
            .component(comp.name.clone(), "")
            .add_tensor_stream(&comp.name, Box::new(stream));
    }

    let mut files = Vec::new();
    for f in &plan.files {
        files.push(f.path.clone());
        builder = builder.add_file_bytes(&f.path, f.bytes.clone(), FileTag::Inference)?;
    }
    for (name, path) in &opts.extra_files {
        files.push(name.clone());
        builder = builder.add_file_path(name, path, FileTag::Inference)?;
    }

    let free = synaptix_bundle::available_space(out.parent().unwrap_or(Path::new(".")))?;
    if free < payload_bytes + (payload_bytes / 50) {
        return Err(GgufError::Bundle(format!(
            "недостаточно места: нужно ~{:.1} ГБ, свободно {:.1} ГБ",
            payload_bytes as f64 / 1e9,
            free as f64 / 1e9
        )));
    }

    builder.write(out)?;

    Ok(ConvertReport {
        arch: plan.arch,
        bundle_id: plan.bundle_id,
        components,
        files,
        payload_bytes,
        output: out.to_path_buf(),
    })
}

use synaptix_bundle::TensorStream as _;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_id_from_file_stem() {
        assert_eq!(
            default_bundle_id(Path::new("/x/Qwen3.6-27B-MTP-Q8_0.gguf")),
            "Qwen3.6-27B-MTP-Q8_0"
        );
    }
}
