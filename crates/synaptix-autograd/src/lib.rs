pub mod backward;
pub mod checkpoint;
pub mod functional;
pub mod grad_fn;
pub mod graph;
pub mod hooks;
pub mod no_grad;
pub mod offload;
pub mod tape;
pub mod variable;

use std::sync::Arc;
use std::sync::Once;

use synaptix_core::error::Result;
use synaptix_core::grad::set_grad_fn_builder;

static INIT: Once = Once::new();

pub fn init() -> Result<()> {
    let mut result = Ok(());
    INIT.call_once(|| {
        let builder = Arc::new(grad_fn::Builder);
        if let Err(e) = set_grad_fn_builder(builder) {
            result = Err(e);
        }
    });
    result
}
