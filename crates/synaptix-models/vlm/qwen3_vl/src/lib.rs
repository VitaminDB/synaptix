pub mod config;
pub mod loader;
pub mod model;
pub mod preprocess;
pub mod pipeline;
pub mod presentation;
pub mod text_model;

pub use config::{ConfigError, VisionConfig};
pub use loader::{bundle_has_vision, load_from_bundle, BundleVisionWeights};
pub use model::{VisionError, VisionTower, VisionWeights};
pub use preprocess::{
    prepare_image, prepare_tensor, prepare_video, ImageGrid, PreparedImage, PreparedVideo,
    PreprocessLimits, VideoLimits,
};
pub use pipeline::{DirWeights, H3Conditioning, H3Encoder, H3_ENCODER_LAYERS};
pub use presentation::{H3Presentation, PresentationItem, TokenTag};
pub use text_model::{build_mrope, rope_positions, MRopeTables, TextConfig, TextEncoder, VisionSpan};
