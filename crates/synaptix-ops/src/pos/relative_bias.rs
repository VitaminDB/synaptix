use synaptix_core::device::Device;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn t5_relative_position_bucket(
    seq_len_q: usize,
    seq_len_k: usize,
    num_buckets: usize,
    max_distance: usize,
    bidirectional: bool,
) -> Vec<u32> {
    let mut buckets = vec![0u32; seq_len_q * seq_len_k];
    let half = if bidirectional { num_buckets / 2 } else { num_buckets };
    let exact = half / 2;
    let log_max = (max_distance as f32 / exact as f32).ln();
    for i in 0..seq_len_q {
        for j in 0..seq_len_k {
            let rel = j as isize - i as isize;
            let n;
            let mut bucket;
            if bidirectional {
                bucket = if rel > 0 { num_buckets / 2 } else { 0 } as u32;
                n = rel.unsigned_abs();
            } else {
                bucket = 0;
                n = (-rel).max(0) as usize;
            }
            let small_inc;
            if n < exact {
                small_inc = n;
            } else {
                let f = ((n as f32 / exact as f32).ln() / log_max) * (half - exact) as f32;
                small_inc = (exact as f32 + f).min((half - 1) as f32) as usize;
            }
            bucket += small_inc as u32;
            buckets[i * seq_len_k + j] = bucket;
        }
    }
    buckets
}

pub fn t5_relative_bias(
    bias_table: &Tensor,
    seq_len_q: usize,
    seq_len_k: usize,
    bidirectional: bool,
    max_distance: usize,
) -> Result<Tensor> {
    if bias_table.rank() != 2 {
        return Err(SynaptixError::Unsupported("t5_relative_bias: table must be (num_buckets, num_heads)"));
    }
    let num_buckets = bias_table.dims()[0];
    let buckets = t5_relative_position_bucket(
        seq_len_q,
        seq_len_k,
        num_buckets,
        max_distance,
        bidirectional,
    );
    let device: Device = bias_table.device();
    let bucket_t = Tensor::from_vec(buckets, (seq_len_q * seq_len_k,), device)?;
    let selected = bias_table.index_select(0, &bucket_t)?;
    selected.reshape((seq_len_q, seq_len_k, bias_table.dims()[1]))
}
