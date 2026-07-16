#[cfg(feature = "video")]
pub mod ffmpeg;
#[cfg(feature = "video")]
pub mod frame_extractor;
#[cfg(feature = "video")]
pub mod h264_writer;

#[cfg(feature = "video")]
pub use frame_extractor::extract_frames;
#[cfg(feature = "video")]
pub use h264_writer::write_h264;
