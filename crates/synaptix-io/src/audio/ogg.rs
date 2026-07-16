use std::path::Path;
use crate::error::Result;
use super::AudioBuffer;
use super::mp3::decode_via_symphonia;

pub fn decode_ogg(path: impl AsRef<Path>) -> Result<AudioBuffer> {
    decode_via_symphonia(path.as_ref())
}
