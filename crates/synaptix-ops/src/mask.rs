use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

const NEG_INF: f32 = f32::NEG_INFINITY;

pub fn causal_mask(seq_len: usize, device: Device) -> Result<Tensor> {
    let mut data = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                data[i * seq_len + j] = NEG_INF;
            }
        }
    }
    Tensor::from_vec(data, (seq_len, seq_len), device)
}

pub fn sliding_window_mask(seq_len: usize, window: usize, device: Device) -> Result<Tensor> {
    if window == 0 {
        return Err(SynaptixError::Unsupported("sliding_window_mask: zero window"));
    }
    let mut data = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let in_causal = j <= i;
            let in_window = i.saturating_sub(j) < window;
            if !(in_causal && in_window) {
                data[i * seq_len + j] = NEG_INF;
            }
        }
    }
    Tensor::from_vec(data, (seq_len, seq_len), device)
}

pub fn document_mask(doc_ids: &Tensor) -> Result<Tensor> {
    if doc_ids.rank() != 1 {
        return Err(SynaptixError::RankMismatch { expected: 1, got: doc_ids.rank() });
    }
    if doc_ids.dtype() != DType::U32 && doc_ids.dtype() != DType::I64 && doc_ids.dtype() != DType::I32 {
        return Err(SynaptixError::Unsupported("document_mask: doc_ids dtype (use U32/I32/I64)"));
    }
    let n = doc_ids.dims()[0];
    let ids: Vec<i64> = match doc_ids.dtype() {
        DType::U32 => doc_ids.to_vec1::<u32>()?.into_iter().map(|v| v as i64).collect(),
        DType::I32 => doc_ids.to_vec1::<i32>()?.into_iter().map(|v| v as i64).collect(),
        DType::I64 => doc_ids.to_vec1::<i64>()?,
        _ => unreachable!(),
    };
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if ids[i] != ids[j] || j > i {
                data[i * n + j] = NEG_INF;
            }
        }
    }
    Tensor::from_vec(data, (n, n), doc_ids.device())
}

pub fn sink_mask(seq_len: usize, num_sink: usize, window: usize, device: Device) -> Result<Tensor> {
    if window == 0 {
        return Err(SynaptixError::Unsupported("sink_mask: zero window"));
    }
    let mut data = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let is_sink = j < num_sink;
            let in_window = j <= i && i.saturating_sub(j) < window;
            if !(is_sink && j <= i || in_window) {
                data[i * seq_len + j] = NEG_INF;
            }
        }
    }
    Tensor::from_vec(data, (seq_len, seq_len), device)
}

pub fn combine_masks(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.broadcast_add(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(t: Tensor) -> Vec<f32> {
        t.to_vec2::<f32>().unwrap().into_iter().flatten().collect()
    }

    #[test]
    fn causal_mask_upper_triangle_neg_inf() {
        synaptix_kernels_cpu::ensure_registered();
        let m = causal_mask(4, Device::Cpu).unwrap();
        let v = collect(m);
        for i in 0..4 {
            for j in 0..4 {
                let val = v[i * 4 + j];
                if j > i {
                    assert!(val == NEG_INF, "({i},{j}) should be -inf");
                } else {
                    assert_eq!(val, 0.0, "({i},{j}) should be 0");
                }
            }
        }
    }

    #[test]
    fn sliding_window_mask_respects_window() {
        synaptix_kernels_cpu::ensure_registered();
        let m = sliding_window_mask(5, 2, Device::Cpu).unwrap();
        let v = collect(m);
        for i in 0..5 {
            for j in 0..5 {
                let val = v[i * 5 + j];
                let allowed = j <= i && i - j < 2;
                if allowed {
                    assert_eq!(val, 0.0);
                } else {
                    assert!(val == NEG_INF);
                }
            }
        }
    }

    #[test]
    fn document_mask_blocks_cross_document() {
        synaptix_kernels_cpu::ensure_registered();
        let ids = Tensor::from_vec(vec![0_u32, 0, 1, 1], (4,), Device::Cpu).unwrap();
        let m = document_mask(&ids).unwrap();
        let v = collect(m);
        for i in 0..4 {
            for j in 0..4 {
                let val = v[i * 4 + j];
                let same_doc = (i < 2 && j < 2) || (i >= 2 && j >= 2);
                if same_doc && j <= i {
                    assert_eq!(val, 0.0);
                } else {
                    assert!(val == NEG_INF);
                }
            }
        }
    }

    #[test]
    fn sink_mask_keeps_first_tokens() {
        synaptix_kernels_cpu::ensure_registered();
        let m = sink_mask(6, 2, 2, Device::Cpu).unwrap();
        let v = collect(m);
        for i in 0..6 {
            let val_sink_0 = v[i * 6 + 0];
            let val_sink_1 = v[i * 6 + 1];
            assert_eq!(val_sink_0, 0.0, "sink token 0 visible from i={i}");
            if i >= 1 {
                assert_eq!(val_sink_1, 0.0, "sink token 1 visible from i={i}");
            }
        }
        let val = v[5 * 6 + 2];
        assert!(val == NEG_INF, "token 2 should be out-of-window for i=5");
    }

    #[test]
    fn combine_masks_adds() {
        synaptix_kernels_cpu::ensure_registered();
        let a = causal_mask(3, Device::Cpu).unwrap();
        let b = sliding_window_mask(3, 1, Device::Cpu).unwrap();
        let c = combine_masks(&a, &b).unwrap();
        let v = collect(c);
        for i in 0..3 {
            for j in 0..3 {
                let val = v[i * 3 + j];
                let in_both = j <= i && i - j < 1;
                if in_both {
                    assert_eq!(val, 0.0);
                } else {
                    assert!(val == NEG_INF);
                }
            }
        }
    }
}
