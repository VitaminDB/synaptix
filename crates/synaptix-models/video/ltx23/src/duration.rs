//! Голова длительности LTX-2.5 (`ltx-2.5-duration-head`): по выходам
//! коннекторов текста (видео `[1,T,4096]`, звук `[1,T,2048]`) предсказывает
//! длину ролика в секундах. Эталон: LTX-2 `duration_head.py`,
//! `duration_predictor.py`. Проекции — на устройстве, пулер (один запрос,
//! 4 головы) и MLP — на хосте в f32.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::WeightLoader;

use crate::text_encoder::linear;
use crate::LtxError;

const P: &str = "duration_head";
const DIM: usize = 256;
const HEADS: usize = 4;

pub struct DurationHead {
    v_proj: (Tensor, Tensor),
    a_proj: (Tensor, Tensor),
    v_emb: Vec<f32>,
    a_emb: Vec<f32>,
    query: Vec<f32>,
    in_w: Vec<f32>,
    in_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    hid_w: Vec<f32>,
    hid_b: Vec<f32>,
    fin_w: Vec<f32>,
    fin_b: f32,
    device: Device,
}

fn mat_vec(w: &[f32], b: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
    let cols = x.len();
    (0..rows).map(|r| b[r] + w[r * cols..(r + 1) * cols].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()).collect()
}

impl DurationHead {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Result<Self, LtxError> {
        let ld = crate::loader::open_weights(path.as_ref())?.with_device(device);
        let t = |n: &str, dev: Device| -> Result<Tensor, LtxError> { ld.load_to(&format!("{P}.{n}"), dev, DType::F32).map_err(|e| LtxError::Load(format!("{n}: {e}"))) };
        let v = |n: &str| -> Result<Vec<f32>, LtxError> { Ok(t(n, Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?) };
        Ok(Self {
            v_proj: (t("video_input_proj.weight", device)?, t("video_input_proj.bias", device)?),
            a_proj: (t("audio_input_proj.weight", device)?, t("audio_input_proj.bias", device)?),
            v_emb: v("video_modality_emb")?,
            a_emb: v("audio_modality_emb")?,
            query: v("attention_pooler.query_tokens")?,
            in_w: v("attention_pooler.cross_attn.in_proj_weight")?,
            in_b: v("attention_pooler.cross_attn.in_proj_bias")?,
            out_w: v("attention_pooler.cross_attn.out_proj.weight")?,
            out_b: v("attention_pooler.cross_attn.out_proj.bias")?,
            hid_w: v("mlp_hidden.weight")?,
            hid_b: v("mlp_hidden.bias")?,
            fin_w: v("mlp_out.weight")?,
            fin_b: v("mlp_out.bias")?[0],
            device,
        })
    }

    /// Секунды ролика по контекстам видео и звука (выходы коннекторов).
    pub fn seconds(&self, video: &Tensor, audio: &Tensor) -> Result<f64, LtxError> {
        let host = |x: &Tensor, (w, b): &(Tensor, Tensor), emb: &[f32]| -> Result<Vec<f32>, LtxError> {
            let x = x.to_device(self.device)?.to_dtype(DType::F32)?;
            let y = linear(&x, w, Some(b))?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            Ok(y.chunks(DIM).flat_map(|r| r.iter().zip(emb).map(|(a, e)| a + e).collect::<Vec<_>>()).collect())
        };
        let mut tok = host(video, &self.v_proj, &self.v_emb)?;
        tok.extend(host(audio, &self.a_proj, &self.a_emb)?);
        let n = tok.len() / DIM;
        // nn.MultiheadAttention: in_proj = [Wq; Wk; Wv], один запрос, HEADS голов.
        let (wq, wk, wv) = (&self.in_w[..DIM * DIM], &self.in_w[DIM * DIM..2 * DIM * DIM], &self.in_w[2 * DIM * DIM..]);
        let (bq, bk, bv) = (&self.in_b[..DIM], &self.in_b[DIM..2 * DIM], &self.in_b[2 * DIM..]);
        let q = mat_vec(wq, bq, &self.query, DIM);
        let hd = DIM / HEADS;
        let mut scores = vec![0f32; HEADS * n];
        let mut vals = vec![0f32; n * DIM];
        for (i, row) in tok.chunks(DIM).enumerate() {
            let k = mat_vec(wk, bk, row, DIM);
            vals[i * DIM..(i + 1) * DIM].copy_from_slice(&mat_vec(wv, bv, row, DIM));
            for h in 0..HEADS {
                scores[h * n + i] = q[h * hd..(h + 1) * hd].iter().zip(&k[h * hd..(h + 1) * hd]).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt();
            }
        }
        let mut pooled = vec![0f32; DIM];
        for h in 0..HEADS {
            let s = &scores[h * n..(h + 1) * n];
            let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f32 = e.iter().sum();
            for (i, w) in e.iter().enumerate() {
                let w = w / z;
                for d in 0..hd {
                    pooled[h * hd + d] += w * vals[i * DIM + h * hd + d];
                }
            }
        }
        let pooled = mat_vec(&self.out_w, &self.out_b, &pooled, DIM);
        let hid: Vec<f32> = mat_vec(&self.hid_w, &self.hid_b, &pooled, DIM)
            .into_iter()
            .map(|x| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh()))
            .collect();
        let log_d = self.fin_b + self.fin_w.iter().zip(&hid).map(|(a, b)| a * b).sum::<f32>();
        Ok((log_d as f64).exp())
    }
}

/// Кадры по секундам: `round(s·fps)`, в пределах `[min_s, max_s]`, на сетку `8k+1`.
pub fn frames_for(seconds: f64, fps: f64, min_s: f64, max_s: f64) -> usize {
    let (lo, hi) = ((min_s * fps).round() as usize, (max_s * fps).round() as usize);
    let raw = ((seconds * fps).round() as usize).clamp(lo, hi);
    let snapped = (raw.max(1) - 1) / 8 * 8 + 1;
    if snapped < lo {
        (lo.max(1) - 1).div_ceil(8) * 8 + 1
    } else {
        snapped
    }
}

#[cfg(test)]
mod tests {
    use super::frames_for;

    #[test]
    fn frame_grid() {
        assert_eq!(frames_for(5.04, 24.0, 1.0, 20.0), 121);
        assert_eq!(frames_for(0.2, 24.0, 1.0, 20.0), 25);
        assert_eq!(frames_for(99.0, 24.0, 1.0, 20.0), 473);
        assert_eq!(frames_for(3.0, 24.0, 1.0, 20.0), 65);
    }
}
