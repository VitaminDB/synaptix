use super::{StreamingDelta, StreamingFinal};
use super::sse_writer::{delta_to_sse, final_to_sse};

pub fn delta_to_ws_text(delta: &StreamingDelta) -> String {
    delta_to_sse(delta)
}

pub fn final_to_ws_text(fin: &StreamingFinal) -> String {
    final_to_sse(fin)
}

pub fn delta_to_ws_bytes(delta: &StreamingDelta) -> Vec<u8> {
    delta_to_ws_text(delta).into_bytes()
}

pub fn final_to_ws_bytes(fin: &StreamingFinal) -> Vec<u8> {
    final_to_ws_text(fin).into_bytes()
}
