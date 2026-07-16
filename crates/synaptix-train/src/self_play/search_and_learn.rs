pub struct SearchAndLearnConfig { pub num_rollouts: usize, pub rollout_length: usize }

impl Default for SearchAndLearnConfig {
    fn default() -> Self { Self { num_rollouts: 16, rollout_length: 64 } }
}
