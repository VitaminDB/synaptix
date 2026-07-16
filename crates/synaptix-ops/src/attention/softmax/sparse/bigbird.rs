use synaptix_core::device::Device;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;

pub struct BigBirdConfig {
    pub window: usize,
    pub num_global: usize,
    pub random_per_row: usize,
    pub seed: u64,
}

impl Default for BigBirdConfig {
    fn default() -> Self {
        Self { window: 3, num_global: 2, random_per_row: 3, seed: 0 }
    }
}

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

fn bigbird_mask(
    s_q: usize,
    s_kv: usize,
    config: &BigBirdConfig,
    causal: bool,
    device: Device,
) -> Result<Tensor> {
    let mut data = vec![f32::NEG_INFINITY; s_q * s_kv];
    for i in 0..s_q {
        let qi = i + (s_kv - s_q);
        for j in 0..s_kv {
            if causal && j > qi {
                continue;
            }
            let local = (qi as isize - j as isize).abs() as usize <= config.window;
            let global_q = qi < config.num_global;
            let global_k = j < config.num_global;
            if local || global_q || global_k {
                data[i * s_kv + j] = 0.0;
            }
        }
        if config.random_per_row > 0 && s_kv > 0 {
            let mut rng_state = config.seed.wrapping_add(i as u64);
            for _ in 0..config.random_per_row {
                let r = (lcg_next(&mut rng_state) as usize) % s_kv;
                if causal && r > qi {
                    continue;
                }
                data[i * s_kv + r] = 0.0;
            }
        }
    }
    Tensor::from_vec::<_, f32>(data, vec![s_q, s_kv], device)
}

pub fn bigbird_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    config: &BigBirdConfig,
    causal: bool,
) -> Result<Tensor> {
    let s_q = q.dims()[q.rank() - 2];
    let s_kv = k.dims()[k.rank() - 2];
    let mask = bigbird_mask(s_q, s_kv, config, causal, q.device())?;
    scaled_dot_attention(q, k, v, scale, Some(&mask))
}
