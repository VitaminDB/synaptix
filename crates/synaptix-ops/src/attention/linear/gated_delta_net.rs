use synaptix_core::{
    device::Device,
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::linear::{linear_dims, to_f32_vec};

/// Gated DeltaNet (Qwen3.6). Конвенция bit-exact с CUDA-ядром
/// `synaptix-kernels-cuda/cu/gated_delta_rule.cu`:
///   q,k L2-нормализуются по каналу (eps 1e-6); `q *= q_scale`;
///   `g_t = exp(g[b,h,t])`, `β_t = beta[b,h,t]`.
///   `Sg = g_t·S_{t-1}`;  `kv = Sgᵀ k_t`;  `δ = β_t (v_t − kv)`;
///   `S_t = Sg + k_t δᵀ`;  `o_t = S_tᵀ q_t`.
///
/// `g` и `beta` — формы `[B,H,S]` (g = log-decay).
pub fn gated_delta_net_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    q_scale: f32,
) -> Result<Tensor> {
    let (b, h, s, dk, dv) = linear_dims(q, k, v)?;
    if g.dims() != [b, h, s] {
        return Err(SynaptixError::shape_mismatch(&[b, h, s], g.dims()));
    }
    if beta.dims() != [b, h, s] {
        return Err(SynaptixError::shape_mismatch(&[b, h, s], beta.dims()));
    }
    let dtype_in = q.dtype();
    let qf = to_f32_vec(q)?;
    let kf = to_f32_vec(k)?;
    let vf = to_f32_vec(v)?;
    let gf = to_f32_vec(g)?;
    let bf = to_f32_vec(beta)?;

    let mut out = vec![0.0f32; b * h * s * dv];
    let mut qn = vec![0.0f32; dk];
    let mut kn = vec![0.0f32; dk];
    let mut kv_mem = vec![0.0f32; dv];
    let mut delta = vec![0.0f32; dv];
    for bi in 0..b {
        for hi in 0..h {
            let mut state = vec![0.0f32; dk * dv];
            for t in 0..s {
                let qk_off = ((bi * h + hi) * s + t) * dk;
                let v_off = ((bi * h + hi) * s + t) * dv;
                let sc_idx = (bi * h + hi) * s + t;

                // L2-norm q,k по каналу (eps 1e-6), scale q.
                let mut sq = 0.0f32;
                let mut sk = 0.0f32;
                for r in 0..dk {
                    sq += qf[qk_off + r] * qf[qk_off + r];
                    sk += kf[qk_off + r] * kf[qk_off + r];
                }
                let inv_q = 1.0 / (sq + 1e-6).sqrt();
                let inv_k = 1.0 / (sk + 1e-6).sqrt();
                for r in 0..dk {
                    qn[r] = qf[qk_off + r] * inv_q * q_scale;
                    kn[r] = kf[qk_off + r] * inv_k;
                }

                let g_t = gf[sc_idx].exp();
                let beta_t = bf[sc_idx];

                // decay: Sg = g_t · S
                for x in state.iter_mut() {
                    *x *= g_t;
                }
                // kv[c] = Σ_r Sg[r,c] k_t[r]
                for c in 0..dv {
                    kv_mem[c] = 0.0;
                }
                for r in 0..dk {
                    let kk = kn[r];
                    let row = r * dv;
                    for c in 0..dv {
                        kv_mem[c] += state[row + c] * kk;
                    }
                }
                // delta[c] = beta_t (v_t[c] - kv[c]);  S[r,c] += k_t[r] delta[c]
                for c in 0..dv {
                    delta[c] = beta_t * (vf[v_off + c] - kv_mem[c]);
                }
                for r in 0..dk {
                    let kk = kn[r];
                    let row = r * dv;
                    for c in 0..dv {
                        state[row + c] += kk * delta[c];
                    }
                }
                // o_t[c] = Σ_r S[r,c] q_t[r]
                for c in 0..dv {
                    let mut acc = 0.0f32;
                    for r in 0..dk {
                        acc += state[r * dv + c] * qn[r];
                    }
                    out[v_off + c] = acc;
                }
            }
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![b, h, s, dv], q.device())?.to_dtype(dtype_in)
}

pub struct GatedDeltaNetState {
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    /// Device-резидентные зеркала (для CUDA-graph decode): `conv_state_dev` F16
    /// `[(K-1), conv_dim]`, `ssm_state_dev` F32 `[num_v, dk, dv]`. Создаются
    /// лениво в [`Self::sync_to_device`]; host-векторы остаются источником истины
    /// (prefill пишет их), dev-зеркала пересеиваются перед graph-replay.
    pub conv_state_dev: Option<Tensor>,
    pub ssm_state_dev: Option<Tensor>,
}

impl GatedDeltaNetState {
    pub fn new(conv_dim: usize, conv_k: usize, num_v_heads: usize, dk: usize, dv: usize) -> Self {
        Self {
            conv_state: vec![0.0; conv_k.saturating_sub(1) * conv_dim],
            ssm_state: vec![0.0; num_v_heads * dk * dv],
            conv_state_dev: None,
            ssm_state_dev: None,
        }
    }
    pub fn reset(&mut self) {
        self.conv_state.iter_mut().for_each(|x| *x = 0.0);
        self.ssm_state.iter_mut().for_each(|x| *x = 0.0);
        self.conv_state_dev = None;
        self.ssm_state_dev = None;
    }

    /// Сеет/пересеивает device-зеркала из host-векторов. Первый вызов аллоцирует
    /// dev-тензоры; последующие копируют host→существующий буфер **in-place**
    /// (стабильный указатель — обязателен для graph-replay, который захватил адрес
    /// при capture). `conv_state` → F16, `ssm_state` → F32.
    #[allow(clippy::too_many_arguments)]
    pub fn sync_to_device(
        &mut self,
        device: Device,
        conv_dim: usize,
        conv_k: usize,
        num_v_heads: usize,
        dk: usize,
        dv: usize,
    ) -> Result<()> {
        let rows = conv_k.saturating_sub(1);
        let cs_src = Tensor::from_vec(self.conv_state.clone(), vec![rows, conv_dim], device)?
            .to_dtype(DType::F16)?;
        match self.conv_state_dev.as_mut() {
            Some(d) => d.copy_from(&cs_src)?,
            None => self.conv_state_dev = Some(cs_src),
        }
        let ss_src = Tensor::from_vec(self.ssm_state.clone(), vec![num_v_heads, dk, dv], device)?;
        match self.ssm_state_dev.as_mut() {
            Some(d) => d.copy_from(&ss_src)?,
            None => self.ssm_state_dev = Some(ss_src),
        }
        Ok(())
    }

    /// Обратный синк device→host: считывает device-зеркала обратно в host-векторы.
    /// Нужен после graph-decode (он продвигает только `*_dev`), чтобы продолжение
    /// host-scan на следующем ходе (prefix-KV-кэш) стартовало из верного состояния.
    pub fn sync_to_host(&mut self) -> Result<()> {
        if let Some(d) = self.conv_state_dev.as_ref() {
            self.conv_state = d.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        }
        if let Some(d) = self.ssm_state_dev.as_ref() {
            self.ssm_state = d.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        }
        Ok(())
    }
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn gated_delta_decay_beta(
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    s: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut g = vec![0.0f32; h * s];
    let mut beta = vec![0.0f32; h * s];
    for t in 0..s {
        for hi in 0..h {
            let av = a[t * h + hi];
            let bv = b[t * h + hi];
            beta[hi * s + t] = 1.0 / (1.0 + (-bv).exp());
            g[hi * s + t] = -(a_log[hi].exp()) * softplus(av + dt_bias[hi]);
        }
    }
    (g, beta)
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_net_recurrent(
    ssm_state: &mut [f32],
    qe: &[f32],
    ke: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    h: usize,
    s: usize,
    dk: usize,
    dv: usize,
    q_scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; h * s * dv];
    let mut qn = vec![0.0f32; dk];
    let mut kn = vec![0.0f32; dk];
    let mut kv = vec![0.0f32; dv];
    let mut delta = vec![0.0f32; dv];
    for hi in 0..h {
        let st = &mut ssm_state[hi * dk * dv..(hi + 1) * dk * dv];
        for t in 0..s {
            let qoff = (hi * s + t) * dk;
            let voff = (hi * s + t) * dv;
            let sc = hi * s + t;
            let mut sq = 0.0f32;
            let mut sk = 0.0f32;
            for r in 0..dk {
                sq += qe[qoff + r] * qe[qoff + r];
                sk += ke[qoff + r] * ke[qoff + r];
            }
            let iq = 1.0 / (sq + 1e-6).sqrt();
            let ik = 1.0 / (sk + 1e-6).sqrt();
            for r in 0..dk {
                qn[r] = qe[qoff + r] * iq * q_scale;
                kn[r] = ke[qoff + r] * ik;
            }
            let gt = g[sc].exp();
            let bt = beta[sc];
            for x in st.iter_mut() {
                *x *= gt;
            }
            for c in 0..dv {
                kv[c] = 0.0;
            }
            for r in 0..dk {
                let kk = kn[r];
                let row = r * dv;
                for c in 0..dv {
                    kv[c] += st[row + c] * kk;
                }
            }
            for c in 0..dv {
                delta[c] = bt * (v[voff + c] - kv[c]);
            }
            for r in 0..dk {
                let kk = kn[r];
                let row = r * dv;
                for c in 0..dv {
                    st[row + c] += kk * delta[c];
                }
            }
            for c in 0..dv {
                let mut acc = 0.0f32;
                for r in 0..dk {
                    acc += st[r * dv + c] * qn[r];
                }
                out[voff + c] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod recurrent_tests {
    use super::*;
    use synaptix_core::device::Device;

    fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn recurrent_matches_stateless_op() {
        synaptix_kernels_cpu::ensure_registered();
        let (h, s, dk, dv) = (3usize, 5usize, 4usize, 6usize);
        let q_scale = 1.0 / (dk as f32).sqrt();
        let mut seed = 12345u64;
        let qe = rand_vec(h * s * dk, &mut seed);
        let ke = rand_vec(h * s * dk, &mut seed);
        let v = rand_vec(h * s * dv, &mut seed);
        let g = rand_vec(h * s, &mut seed).iter().map(|x| -x.abs()).collect::<Vec<_>>();
        let beta = rand_vec(h * s, &mut seed).iter().map(|x| 0.5 * (x + 1.0)).collect::<Vec<_>>();

        let mut state = vec![0.0f32; h * dk * dv];
        let mine = gated_delta_net_recurrent(&mut state, &qe, &ke, &v, &g, &beta, h, s, dk, dv, q_scale);

        let qt = Tensor::from_vec(qe.clone(), vec![1, h, s, dk], Device::Cpu).unwrap();
        let kt = Tensor::from_vec(ke.clone(), vec![1, h, s, dk], Device::Cpu).unwrap();
        let vt = Tensor::from_vec(v.clone(), vec![1, h, s, dv], Device::Cpu).unwrap();
        let gt = Tensor::from_vec(g.clone(), vec![1, h, s], Device::Cpu).unwrap();
        let bt = Tensor::from_vec(beta.clone(), vec![1, h, s], Device::Cpu).unwrap();
        let reference = gated_delta_net_attention(&qt, &kt, &vt, &gt, &bt, q_scale).unwrap();
        let ref_v: Vec<f32> = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut maxerr = 0.0f32;
        for (a, b) in mine.iter().zip(ref_v.iter()) {
            maxerr = maxerr.max((a - b).abs());
        }
        assert!(maxerr < 1e-4, "recurrent vs stateless op max abs err {maxerr}");
    }

    #[test]
    fn recurrent_stateful_equals_single_chunk() {
        let (h, s, dk, dv) = (2usize, 6usize, 3usize, 4usize);
        let q_scale = 0.7;
        let mut seed = 42u64;
        let qe = rand_vec(h * s * dk, &mut seed);
        let ke = rand_vec(h * s * dk, &mut seed);
        let v = rand_vec(h * s * dv, &mut seed);
        let g = rand_vec(h * s, &mut seed).iter().map(|x| -x.abs()).collect::<Vec<_>>();
        let beta = rand_vec(h * s, &mut seed).iter().map(|x| 0.5 * (x + 1.0)).collect::<Vec<_>>();

        let mut st_full = vec![0.0f32; h * dk * dv];
        let full = gated_delta_net_recurrent(&mut st_full, &qe, &ke, &v, &g, &beta, h, s, dk, dv, q_scale);

        let take = |src: &[f32], per: usize, t0: usize, t1: usize| -> Vec<f32> {
            let mut out = Vec::new();
            for hi in 0..h {
                for t in t0..t1 {
                    let off = (hi * s + t) * per;
                    out.extend_from_slice(&src[off..off + per]);
                }
            }
            out
        };
        let take_sc = |src: &[f32], t0: usize, t1: usize| -> Vec<f32> {
            let mut out = Vec::new();
            for hi in 0..h {
                for t in t0..t1 {
                    out.push(src[hi * s + t]);
                }
            }
            out
        };
        let place = |chunk: &[f32], t0: usize, t1: usize, got: &mut [f32]| {
            let n = t1 - t0;
            for hi in 0..h {
                for ti in 0..n {
                    let src = (hi * n + ti) * dv;
                    let dst = (hi * s + (t0 + ti)) * dv;
                    got[dst..dst + dv].copy_from_slice(&chunk[src..src + dv]);
                }
            }
        };
        let mut st = vec![0.0f32; h * dk * dv];
        let mut got = vec![0.0f32; h * s * dv];
        for &(t0, t1) in &[(0usize, 4usize), (4, 5), (5, 6)] {
            let n = t1 - t0;
            let oc = gated_delta_net_recurrent(
                &mut st,
                &take(&qe, dk, t0, t1),
                &take(&ke, dk, t0, t1),
                &take(&v, dv, t0, t1),
                &take_sc(&g, t0, t1),
                &take_sc(&beta, t0, t1),
                h,
                n,
                dk,
                dv,
                q_scale,
            );
            place(&oc, t0, t1, &mut got);
        }
        let mut maxerr = 0.0f32;
        for (a, b) in full.iter().zip(got.iter()) {
            maxerr = maxerr.max((a - b).abs());
        }
        assert!(maxerr < 1e-5, "stateful chunked != single chunk, err {maxerr}");
    }
}
