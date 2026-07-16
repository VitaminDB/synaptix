use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Fine-grained sparse MoE forward (DeepSeek-стиль: много мелких экспертов, top-k).
/// `x:[N,D]`, `router_w:[D,E]`, `experts_fc1:[E,H,D]`, `experts_fc2:[E,D,H]`, выбор `k`
/// экспертов на токен. Число мини-экспертов `E` берётся из `experts_fc1`.
///   `logits = x·router_w`; `probs = softmax(logits)`; top-k экспертов, веса
///   перенормированы по top-k; `y_n = Σ_{e∈topk} w·relu(x_n·fc1_eᵀ)·fc2_eᵀ`.
pub fn fine_grained_moe(
    x: &Tensor,
    router_w: &Tensor,
    experts_fc1: &Tensor,
    experts_fc2: &Tensor,
    k: usize,
) -> Result<Tensor> {
    if x.rank() != 2 {
        return Err(SynaptixError::Unsupported("fine_grained_moe: x must be [N,D]"));
    }
    let (n, d) = (x.dims()[0], x.dims()[1]);
    if router_w.rank() != 2 || router_w.dims()[0] != d {
        return Err(SynaptixError::Unsupported("fine_grained_moe: router_w must be [D,E]"));
    }
    let e = router_w.dims()[1];
    if experts_fc1.rank() != 3 || experts_fc1.dims()[0] != e || experts_fc1.dims()[2] != d {
        return Err(SynaptixError::Unsupported("fine_grained_moe: experts_fc1 must be [E,H,D]"));
    }
    let h = experts_fc1.dims()[1];
    if experts_fc2.dims() != [e, d, h] {
        return Err(SynaptixError::Unsupported("fine_grained_moe: experts_fc2 must be [E,D,H]"));
    }
    if k == 0 || k > e {
        return Err(SynaptixError::Unsupported("fine_grained_moe: requires 1 <= k <= E"));
    }
    let xf = f32v(x)?;
    let rw = f32v(router_w)?;
    let f1 = f32v(experts_fc1)?;
    let f2 = f32v(experts_fc2)?;

    let mut out = vec![0.0f32; n * d];
    let mut hid = vec![0.0f32; h];
    for ni in 0..n {
        let xrow = &xf[ni * d..ni * d + d];
        // router-логиты [E]
        let mut logits = vec![0.0f32; e];
        for ei in 0..e {
            let mut acc = 0.0f32;
            for di in 0..d {
                acc += xrow[di] * rw[di * e + ei];
            }
            logits[ei] = acc;
        }
        // top-k экспертов
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_unstable_by(|&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let top = &order[..k];
        // softmax по top-k
        let max = top.iter().map(|&j| logits[j]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|&j| (logits[j] - max).exp()).collect();
        let wsum: f32 = exps.iter().sum();

        for (pos, &ei) in top.iter().enumerate() {
            let w = exps[pos] / wsum;
            // expert_ei: relu(x·fc1ᵀ)·fc2ᵀ
            let f1_base = ei * h * d;
            for hh in 0..h {
                let mut acc = 0.0f32;
                let wrow = f1_base + hh * d;
                for di in 0..d {
                    acc += xrow[di] * f1[wrow + di];
                }
                hid[hh] = acc.max(0.0); // relu
            }
            let f2_base = ei * d * h;
            for di in 0..d {
                let mut acc = 0.0f32;
                let wrow = f2_base + di * h;
                for hh in 0..h {
                    acc += hid[hh] * f2[wrow + hh];
                }
                out[ni * d + di] += w * acc;
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![n, d], x.device())?.to_dtype(x.dtype())
}
