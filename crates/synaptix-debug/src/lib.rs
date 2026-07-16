pub mod compare;
pub mod dump;
pub mod error;
pub mod load;
pub mod nan_detector;
pub mod per_layer;
pub mod py_compat;
pub mod shape_assert;
pub mod sub_block;

pub use compare::{cos_sim, l1_distance, l2_distance, max_abs, rel_err, CompareReport};
pub use dump::{TensorDump, dump_to_file, dump_to_writer};
pub use error::{DebugError, Result};
pub use load::{load_from_file, load_from_reader};
pub use nan_detector::{check_finite, nan_inf_hook, FiniteStats};
pub use per_layer::{dump_hook, register_dump_hook, LayerDumpCollector};
