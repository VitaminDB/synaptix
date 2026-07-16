use once_cell::sync::OnceCell;
use synaptix_core::error::Result;

pub struct LazyModule<M> {
    module: OnceCell<M>,
    init_fn: Box<dyn Fn() -> Result<M> + Send + Sync>,
}

impl<M: Send + Sync> LazyModule<M> {
    pub fn new(f: impl Fn() -> Result<M> + Send + Sync + 'static) -> Self {
        Self { module: OnceCell::new(), init_fn: Box::new(f) }
    }

    pub fn get_or_init(&self) -> Result<&M> {
        if let Some(m) = self.module.get() {
            return Ok(m);
        }
        let m = (self.init_fn)()?;
        Ok(self.module.get_or_init(|| m))
    }

    pub fn is_initialized(&self) -> bool {
        self.module.get().is_some()
    }
}
