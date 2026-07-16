pub mod full;
pub mod paged;
pub mod prefix;
pub mod quantized;
pub mod radix;
pub mod tiered;

pub use full::FullKvCache;
pub use paged::PagedKvCache;
pub use prefix::PrefixKvCache;

use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub trait KvCache: Send + Sync {
    fn num_layers(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn num_heads(&self) -> usize;
    fn append(&mut self, layer: usize, key: &Tensor, value: &Tensor) -> Result<()>;
    fn get(&self, layer: usize) -> Option<(&Tensor, &Tensor)>;
    fn seq_len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn clear(&mut self);
    fn reset_to(&mut self, len: usize);
    fn is_full(&self) -> bool {
        self.seq_len() >= self.capacity()
    }
}
