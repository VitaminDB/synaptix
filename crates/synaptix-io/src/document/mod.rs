#[cfg(feature = "documents")]
pub mod chunking;
#[cfg(feature = "documents")]
pub mod folder_walker;
#[cfg(feature = "documents")]
pub mod html;
#[cfg(feature = "documents")]
pub mod markdown;
#[cfg(feature = "documents")]
pub mod pdf;

#[cfg(feature = "documents")]
pub use chunking::TextChunk;
#[cfg(feature = "documents")]
pub use folder_walker::{walk_folder, walk_folder_sorted};
#[cfg(feature = "documents")]
pub use markdown::{markdown_to_text, markdown_to_chunks};
#[cfg(feature = "documents")]
pub use html::{html_to_text, html_to_chunks};
#[cfg(feature = "documents")]
pub use pdf::{pdf_to_text, pdf_to_chunks};
