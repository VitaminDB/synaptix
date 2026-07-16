use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use synaptix_core::tensor::Tensor;

use crate::dump::dump_to_file;

pub fn dump_hook(
    name: impl Into<String>,
    dir: impl AsRef<Path>,
) -> impl Fn(&Tensor) -> Option<Tensor> + Send + Sync + 'static {
    let name = name.into();
    let dir = dir.as_ref().to_path_buf();
    move |t| {
        let path = dir.join(format!("{}.syndump", name));
        if let Err(e) = dump_to_file(t, &name, &path) {
            eprintln!("[dump_hook:{name}] failed: {e}");
        }
        None
    }
}

pub fn register_dump_hook(
    tensor: &Tensor,
    name: impl Into<String>,
    dir: impl AsRef<Path>,
) {
    let hook = dump_hook(name, dir);
    tensor.register_hook(hook);
}

#[derive(Clone)]
pub struct LayerDumpCollector {
    inner: Arc<Mutex<CollectorInner>>,
}

struct CollectorInner {
    dir: PathBuf,
    counter: usize,
}

impl LayerDumpCollector {
    pub fn new(dir: impl AsRef<Path>) -> Self {
        let inner = CollectorInner { dir: dir.as_ref().to_path_buf(), counter: 0 };
        Self { inner: Arc::new(Mutex::new(inner)) }
    }

    pub fn attach(&self, tensor: &Tensor, label: impl Into<String>) {
        let label = label.into();
        let cell = self.inner.clone();
        tensor.register_hook(move |t| {
            let mut g = cell.lock().expect("collector mutex poisoned");
            let idx = g.counter;
            g.counter += 1;
            let dir = g.dir.clone();
            drop(g);
            let path = dir.join(format!("{:04}-{}.syndump", idx, label));
            if let Err(e) = dump_to_file(t, &label, &path) {
                eprintln!("[layer_dump_collector:{label}] failed: {e}");
            }
            None
        });
    }
}
