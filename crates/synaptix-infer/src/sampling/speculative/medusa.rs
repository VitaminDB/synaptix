pub struct MedusaHead {
    pub num_heads: usize,
    pub vocab_size: usize,
}

impl MedusaHead {
    pub fn new(num_heads: usize, vocab_size: usize) -> Self { Self { num_heads, vocab_size } }
}
