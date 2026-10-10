pub mod cross_entropy;
pub mod ctc;
pub mod mse;

pub use cross_entropy::{cross_entropy, nll_loss};
pub use ctc::{ctc_host, ctc_loss, CtcHost, CtcReduction};
pub use mse::{l1_loss, mse_loss, smooth_l1_loss};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}
