#[allow(dead_code)]
pub struct ShuffleBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    seed: u64,
    rng_state: u64,
}

impl<T> ShuffleBuffer<T> {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self { buffer: Vec::with_capacity(capacity), capacity, seed, rng_state: seed }
    }

    fn next_rand(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub fn push(&mut self, item: T) -> Option<T> {
        self.buffer.push(item);
        if self.buffer.len() >= self.capacity {
            let len = self.buffer.len();
            let idx = (self.next_rand() as usize) % len;
            let last = len - 1;
            self.buffer.swap(idx, last);
            self.buffer.pop()
        } else {
            None
        }
    }

    pub fn drain(mut self) -> impl Iterator<Item = T> {
        let len = self.buffer.len();
        for i in (1..len).rev() {
            let j = (self.next_rand() as usize) % (i + 1);
            self.buffer.swap(i, j);
        }
        self.buffer.into_iter()
    }
}
