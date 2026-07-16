pub struct AudioChunker {
    buffer: Vec<f32>,
    chunk_size: usize,
    hop: usize,
}

impl AudioChunker {
    pub fn new(chunk_size: usize, hop: usize) -> Self {
        Self { buffer: Vec::new(), chunk_size, hop }
    }

    pub fn push(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        self.buffer.extend_from_slice(samples);
        let mut out = Vec::new();
        while self.buffer.len() >= self.chunk_size {
            out.push(self.buffer[..self.chunk_size].to_vec());
            self.buffer.drain(..self.hop.min(self.buffer.len()));
        }
        out
    }

    pub fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.buffer)
    }

    pub fn pending(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_basic() {
        let mut c = AudioChunker::new(4, 2);
        let out = c.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out[1], vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(out[2], vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn chunker_streaming_incremental() {
        let mut c = AudioChunker::new(3, 3);
        assert!(c.push(&[1.0, 2.0]).is_empty());
        let out = c.push(&[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(out, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(c.pending(), 0);
    }
}
