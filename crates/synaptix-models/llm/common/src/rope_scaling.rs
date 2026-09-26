//! `rope_scaling` из HF-конфига: частоты RoPE и множитель внимания.
//!
//! Формулы повторяют `transformers.modeling_rope_utils` (`linear`, `llama3`,
//! `yarn`) — HF применяет их статически, на любой длине. `dynamic` на длинах до
//! `original_max_position_embeddings` совпадает с обычным RoPE, поэтому вместо
//! него ёмкость контекста ограничивается этой длиной. Неизвестный тип — то же
//! ограничение и предупреждение в лог: без него модель молча шла за исходный
//! контекст без масштабирования и теряла качество.

use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RopeScaling {
    /// Старые конфиги пишут `type` вместо `rope_type`.
    #[serde(alias = "type", default)]
    pub rope_type: String,
    #[serde(default)]
    pub factor: f32,
    #[serde(default)]
    pub low_freq_factor: Option<f32>,
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub attention_factor: Option<f32>,
    #[serde(default)]
    pub mscale: Option<f32>,
    #[serde(default)]
    pub mscale_all_dim: Option<f32>,
    #[serde(default)]
    pub truncate: Option<bool>,
}

/// Итог разбора `rope_scaling`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaledRope {
    /// Частоты длины `rotary_dim/2`; `None` — обычный RoPE на `theta`.
    pub freqs: Option<Vec<f32>>,
    /// Множитель cos/sin (YaRN `attention_factor`). Поворот q и k умножается
    /// на него, значит скоры — на его квадрат: домножить `attn_scale`.
    pub attn_factor: f32,
    /// Ёмкость контекста после проверки типа.
    pub max_position_embeddings: usize,
}

impl ScaledRope {
    fn plain(max_pos: usize) -> Self {
        Self { freqs: None, attn_factor: 1.0, max_position_embeddings: max_pos }
    }

    /// Множитель для `attn_scale` (скоры q·k).
    pub fn attn_scale_mul(&self) -> f32 {
        self.attn_factor * self.attn_factor
    }
}

fn base_inv_freqs(theta: f64, dim: usize) -> Vec<f64> {
    (0..dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / dim as f64))
        .collect()
}

/// `rs = None` или тип `default` → обычный RoPE. `max_pos` — ёмкость из
/// конфига (`max_position_embeddings`).
pub fn resolve(rs: Option<&RopeScaling>, theta: f32, rotary_dim: usize, max_pos: usize) -> ScaledRope {
    let Some(rs) = rs else {
        return ScaledRope::plain(max_pos);
    };
    let kind = rs.rope_type.to_ascii_lowercase();
    let theta = theta as f64;
    match kind.as_str() {
        "" | "default" => ScaledRope::plain(max_pos),
        "linear" if rs.factor > 0.0 => ScaledRope {
            freqs: Some(
                base_inv_freqs(theta, rotary_dim)
                    .into_iter()
                    .map(|f| (f / rs.factor as f64) as f32)
                    .collect(),
            ),
            attn_factor: 1.0,
            max_position_embeddings: max_pos,
        },
        "llama3" if rs.factor > 0.0 => ScaledRope {
            freqs: Some(llama3_freqs(rs, theta, rotary_dim)),
            attn_factor: 1.0,
            max_position_embeddings: max_pos,
        },
        "yarn" if rs.factor > 0.0 => {
            let (freqs, af) = yarn_freqs(rs, theta, rotary_dim, max_pos);
            ScaledRope { freqs: Some(freqs), attn_factor: af, max_position_embeddings: max_pos }
        }
        _ => {
            let cap = rs.original_max_position_embeddings.unwrap_or(max_pos).min(max_pos);
            if kind != "dynamic" {
                eprintln!(
                    "[synaptix] rope_scaling {:?} (factor {}) не поддержан — обычный RoPE, контекст ограничен {} токенами",
                    rs.rope_type,
                    rs.factor,
                    cap
                );
            }
            ScaledRope::plain(cap)
        }
    }
}

/// `transformers._compute_llama3_parameters`.
fn llama3_freqs(rs: &RopeScaling, theta: f64, dim: usize) -> Vec<f32> {
    let factor = rs.factor as f64;
    let low_ff = rs.low_freq_factor.unwrap_or(1.0) as f64;
    let high_ff = rs.high_freq_factor.unwrap_or(4.0) as f64;
    let orig_ctx = rs.original_max_position_embeddings.unwrap_or(8192) as f64;
    let low_freq_wavelen = orig_ctx / low_ff;
    let high_freq_wavelen = orig_ctx / high_ff;
    let two_pi = 2.0 * std::f64::consts::PI;
    base_inv_freqs(theta, dim)
        .into_iter()
        .map(|inv_freq| {
            let wavelen = two_pi / inv_freq;
            let f = if wavelen < high_freq_wavelen {
                inv_freq
            } else if wavelen > low_freq_wavelen {
                inv_freq / factor
            } else {
                let smooth = (orig_ctx / wavelen - low_ff) / (high_ff - low_ff);
                (1.0 - smooth) * inv_freq / factor + smooth * inv_freq
            };
            f as f32
        })
        .collect()
}

fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// `transformers._compute_yarn_parameters`: частоты и `attention_factor`.
fn yarn_freqs(rs: &RopeScaling, theta: f64, dim: usize, max_pos: usize) -> (Vec<f32>, f32) {
    let factor = rs.factor as f64;
    let attention_factor = match rs.attention_factor {
        Some(a) => a as f64,
        None => match (rs.mscale, rs.mscale_all_dim) {
            (Some(m), Some(ma)) if m != 0.0 && ma != 0.0 => {
                yarn_get_mscale(factor, m as f64) / yarn_get_mscale(factor, ma as f64)
            }
            _ => yarn_get_mscale(factor, 1.0),
        },
    };
    let beta_fast = rs.beta_fast.unwrap_or(32.0) as f64;
    let beta_slow = rs.beta_slow.unwrap_or(1.0) as f64;
    let orig = rs.original_max_position_embeddings.unwrap_or(max_pos) as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let correction_dim =
        |rot: f64| dim as f64 * (orig / (rot * two_pi)).ln() / (2.0 * theta.ln());
    let (mut low, mut high) = (correction_dim(beta_fast), correction_dim(beta_slow));
    if rs.truncate.unwrap_or(true) {
        low = low.floor();
        high = high.ceil();
    }
    let low = low.max(0.0);
    let mut high = high.min(dim as f64 - 1.0);
    if low == high {
        high += 0.001;
    }
    let freqs = base_inv_freqs(theta, dim)
        .into_iter()
        .enumerate()
        .map(|(i, extrap)| {
            let interp = extrap / factor;
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let extrap_factor = 1.0 - ramp;
            (interp * (1.0 - extrap_factor) + extrap * extrap_factor) as f32
        })
        .collect();
    (freqs, attention_factor as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rs(json: &str) -> RopeScaling {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn none_and_default_are_plain() {
        assert_eq!(resolve(None, 1e6, 128, 40960), ScaledRope::plain(40960));
        let d = rs(r#"{"rope_type":"default"}"#);
        assert_eq!(resolve(Some(&d), 1e6, 128, 40960), ScaledRope::plain(40960));
    }

    #[test]
    fn linear_divides_freqs() {
        let r = resolve(Some(&rs(r#"{"type":"linear","factor":4.0}"#)), 10000.0, 8, 4096);
        let f = r.freqs.unwrap();
        assert!((f[0] - 0.25).abs() < 1e-7);
        assert!((f[1] - 0.025).abs() < 1e-7);
        assert_eq!(r.attn_factor, 1.0);
    }

    /// Эталон — `transformers` 5.12 `ROPE_INIT_FUNCTIONS["yarn"]` для Qwen3
    /// (theta 1e6, head_dim 128, factor 4, original 32768).
    #[test]
    fn yarn_matches_transformers() {
        let r = resolve(
            Some(&rs(
                r#"{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}"#,
            )),
            1_000_000.0,
            128,
            131072,
        );
        assert!((r.attn_factor - 1.138_629_4).abs() < 1e-6);
        let f = r.freqs.unwrap();
        assert_eq!(f.len(), 64);
        let want = [
            (0, 1.0f64),
            (18, 0.020_535_251_125_693_32),
            (23, 0.006_978_305_988_013_744),
            (24, 0.005_375_321_488_827_467),
            (30, 0.001_064_360_956_661_403_2),
            (39, 6.490_394_298_452_884e-5),
            (40, 4.445_698_505_151_085_6e-5),
            (63, 3.102_344_408_034_696e-7),
        ];
        for (i, w) in want {
            let rel = ((f[i] as f64) - w).abs() / w;
            assert!(rel < 1e-5, "freq[{i}] = {} vs {w}", f[i]);
        }
    }

    #[test]
    fn unknown_type_caps_context() {
        let r = resolve(
            Some(&rs(
                r#"{"rope_type":"longrope","factor":4.0,"original_max_position_embeddings":4096}"#,
            )),
            10000.0,
            64,
            131072,
        );
        assert_eq!(r, ScaledRope::plain(4096));
        let d = resolve(
            Some(&rs(r#"{"rope_type":"dynamic","factor":2.0,"original_max_position_embeddings":8192}"#)),
            10000.0,
            64,
            32768,
        );
        assert_eq!(d.max_position_embeddings, 8192);
    }
}
