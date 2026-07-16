const PHILOX_M0: u32 = 0xD2511F53;
const PHILOX_M1: u32 = 0xCD9E8D57;
const PHILOX_KEY_BUMP_0: u32 = 0x9E3779B9;
const PHILOX_KEY_BUMP_1: u32 = 0xBB67AE85;
const PHILOX_ROUNDS: usize = 10;

#[derive(Debug, Clone)]
pub struct Philox4x32 {
    key: [u32; 2],
    counter: [u32; 4],
    buffer: [u32; 4],
    buffer_pos: usize,
}

impl Philox4x32 {
    pub fn new(seed: u64) -> Self {
        let key0 = (seed & 0xFFFF_FFFF) as u32;
        let key1 = ((seed >> 32) & 0xFFFF_FFFF) as u32;
        Self {
            key: [key0, key1],
            counter: [0u32; 4],
            buffer: [0u32; 4],
            buffer_pos: 4,
        }
    }

    pub fn from_key_counter(key: [u32; 2], counter: [u32; 4]) -> Self {
        Self { key, counter, buffer: [0u32; 4], buffer_pos: 4 }
    }

    pub fn next_block(&mut self) -> [u32; 4] {
        let out = philox_4x32(self.counter, self.key);
        self.increment_counter();
        out
    }

    pub fn advance(&mut self, n_blocks: u64) {
        self.buffer_pos = 4;
        let mut carry = n_blocks;
        for slot in self.counter.iter_mut() {
            if carry == 0 {
                break;
            }
            let sum = (*slot as u64) + (carry & 0xFFFF_FFFF);
            *slot = (sum & 0xFFFF_FFFF) as u32;
            carry = (carry >> 32) + (sum >> 32);
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.buffer_pos >= 4 {
            self.buffer = self.next_block();
            self.buffer_pos = 0;
        }
        let v = self.buffer[self.buffer_pos];
        self.buffer_pos += 1;
        v
    }

    pub fn next_f32_uniform(&mut self) -> f32 {
        let bits = self.next_u32() >> 8;
        (bits as f32) * (1.0 / (1u32 << 24) as f32)
    }

    fn increment_counter(&mut self) {
        for slot in self.counter.iter_mut() {
            *slot = slot.wrapping_add(1);
            if *slot != 0 {
                return;
            }
        }
    }
}

fn mul_hi_lo(a: u32, b: u32) -> (u32, u32) {
    let product = (a as u64) * (b as u64);
    let hi = (product >> 32) as u32;
    let lo = (product & 0xFFFF_FFFF) as u32;
    (hi, lo)
}

fn philox_4x32(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = counter;
    let mut k = key;
    for _ in 0..PHILOX_ROUNDS {
        let (hi0, lo0) = mul_hi_lo(PHILOX_M0, c[0]);
        let (hi1, lo1) = mul_hi_lo(PHILOX_M1, c[2]);
        let new_c0 = hi1 ^ c[1] ^ k[0];
        let new_c1 = lo1;
        let new_c2 = hi0 ^ c[3] ^ k[1];
        let new_c3 = lo0;
        c = [new_c0, new_c1, new_c2, new_c3];
        k[0] = k[0].wrapping_add(PHILOX_KEY_BUMP_0);
        k[1] = k[1].wrapping_add(PHILOX_KEY_BUMP_1);
    }
    c
}

pub fn fill_uniform_f32(rng: &mut Philox4x32, dst: &mut [f32], low: f32, high: f32) {
    let span = high - low;
    for slot in dst.iter_mut() {
        let u = rng.next_f32_uniform();
        *slot = low + u * span;
    }
}

pub fn fill_uniform_f64(rng: &mut Philox4x32, dst: &mut [f64], low: f64, high: f64) {
    let span = high - low;
    for slot in dst.iter_mut() {
        let hi = (rng.next_u32() >> 6) as u64;
        let lo = (rng.next_u32() >> 5) as u64;
        let bits = (hi << 26) | lo;
        let u = (bits as f64) * (1.0 / ((1u64 << 53) as f64));
        *slot = low + u * span;
    }
}

pub fn fill_normal_f32(rng: &mut Philox4x32, dst: &mut [f32]) {
    let mut i = 0usize;
    while i < dst.len() {
        let u1 = max_floor(rng.next_f32_uniform());
        let u2 = rng.next_f32_uniform();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        dst[i] = z0;
        i += 1;
        if i < dst.len() {
            dst[i] = z1;
            i += 1;
        }
    }
}

fn max_floor(x: f32) -> f32 {
    if x <= f32::MIN_POSITIVE { f32::MIN_POSITIVE } else { x }
}

pub fn bernoulli_mask(rng: &mut Philox4x32, p: f32, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    for slot in out.iter_mut() {
        let u = rng.next_f32_uniform();
        *slot = if u < p { 1.0 } else { 0.0 };
    }
    out
}

pub fn bernoulli_indices(rng: &mut Philox4x32, p: f32, n: usize) -> Vec<bool> {
    let mut out = vec![false; n];
    for slot in out.iter_mut() {
        let u = rng.next_f32_uniform();
        *slot = u < p;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_zero_counter_zero_key() {
        let mut rng = Philox4x32::from_key_counter([0, 0], [0, 0, 0, 0]);
        let block = rng.next_block();
        assert_eq!(block, [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]);
    }

    #[test]
    fn philox_determinism_same_seed() {
        let mut a = Philox4x32::new(42);
        let mut b = Philox4x32::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn philox_advance_skips_blocks() {
        let mut a = Philox4x32::new(7);
        let mut b = Philox4x32::new(7);
        let _ = a.next_block();
        let _ = a.next_block();
        let _ = a.next_block();
        b.advance(3);
        assert_eq!(a.next_block(), b.next_block());
    }

    #[test]
    fn bernoulli_mean_is_close() {
        let mut rng = Philox4x32::new(123);
        let n = 200_000usize;
        let mask = bernoulli_mask(&mut rng, 0.3, n);
        let mean: f64 = mask.iter().map(|&v| v as f64).sum::<f64>() / (n as f64);
        assert!((mean - 0.3).abs() < 0.01, "mean={mean}");
    }

    #[test]
    fn fill_uniform_in_range() {
        let mut rng = Philox4x32::new(0xCAFE_BABE);
        let mut buf = vec![0.0_f32; 1000];
        fill_uniform_f32(&mut rng, &mut buf, -1.0, 2.0);
        for &v in &buf {
            assert!(v >= -1.0 && v < 2.0, "v={v}");
        }
    }

    #[test]
    fn fill_normal_stats_sane() {
        let mut rng = Philox4x32::new(0xC0FFEE);
        let mut buf = vec![0.0_f32; 100_000];
        fill_normal_f32(&mut rng, &mut buf);
        let mean: f64 = buf.iter().map(|&v| v as f64).sum::<f64>() / (buf.len() as f64);
        let var: f64 = buf.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (buf.len() as f64);
        assert!(mean.abs() < 0.02, "mean={mean}");
        assert!((var - 1.0).abs() < 0.05, "var={var}");
    }
}
