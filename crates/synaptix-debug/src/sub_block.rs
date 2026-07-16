use std::path::PathBuf;

use synaptix_core::tensor::Tensor;

use crate::dump::dump_to_file;
use crate::error::Result;

pub struct SubBlockDump {
    dir: PathBuf,
    block_label: String,
}

impl SubBlockDump {
    pub fn new(dir: impl Into<PathBuf>, block_label: impl Into<String>) -> Self {
        Self { dir: dir.into(), block_label: block_label.into() }
    }

    pub fn record(&self, sub_label: &str, tensor: &Tensor) -> Result<()> {
        let name = format!("{}-{}", self.block_label, sub_label);
        let path = self.dir.join(format!("{name}.syndump"));
        dump_to_file(tensor, &name, &path)
    }
}
