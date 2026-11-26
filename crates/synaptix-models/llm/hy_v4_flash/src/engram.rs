//! Engram: хэшированные 2- и 3-граммы → строки таблицы слоя, нормированные
//! по головам. Хэш и нормировка — на хосте (16 строк по 96 на токен и слой),
//! гейт с потоками iHC — на карте (`hy4_engram_gate`).
//!
//! Семантика — как в vLLM (`common/engram_ops.py`), где она расходится с
//! эталоном transformers: eps норм — eps F32 (у transformers `nn.RMSNorm` без
//! eps берёт eps bf16 = 2^-7); токены-заполнители engram не получают; id вне
//! карты сжатия отображаются сами в себя.

use std::sync::Arc;

use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_llm_common::ModelError;

use crate::config::EngramConfig;

/// Сжатие словаря и хэш одного engram-слоя.
pub struct EngramHash {
    /// `lookup_table` чекпойнта: id токена → id нормализованного словаря.
    token_map: Arc<Vec<u32>>,
    multipliers: [u64; 3],
    /// Модули голов (2-граммы, затем 3-граммы).
    primes: Vec<u64>,
    /// Первая строка головы в общей таблице.
    offsets: Vec<u64>,
    pad_token: u32,
    /// Ещё id без памяти: заполнители картинок и видео (их id остаются в
    /// истории n-грамм следующих токенов, как у эталона).
    skip: Vec<u32>,
    max_ngram: usize,
    heads_per_ngram: usize,
}

impl EngramHash {
    pub fn new(cfg: &EngramConfig, index: usize, token_map: Arc<Vec<u32>>, multipliers: [u64; 3]) -> Self {
        let primes = cfg.head_vocab[index].clone();
        let mut offsets = Vec::with_capacity(primes.len());
        let mut acc = 0u64;
        for p in &primes {
            offsets.push(acc);
            acc += p;
        }
        Self { token_map, multipliers, primes, offsets, pad_token: cfg.pad_id, skip: Vec::new(), max_ngram: cfg.max_ngram, heads_per_ngram: cfg.heads_per_ngram }
    }

    pub fn heads(&self) -> usize {
        self.primes.len()
    }

    /// Пропускать (без памяти) и эти id.
    pub fn with_skip(mut self, ids: Vec<u32>) -> Self {
        self.skip = ids;
        self
    }

    fn map(&self, id: u32) -> u64 {
        self.token_map.get(id as usize).copied().unwrap_or(id) as u64
    }

    /// Строки таблицы для позиций `from..ids.len()` последовательности `ids`
    /// (история — всё до `from`): `heads` строк на позицию, `None` — позиция
    /// без памяти (заполнитель).
    pub fn rows(&self, ids: &[u32], from: usize, out: &mut Vec<Option<Vec<u64>>>) {
        let pad = self.map(self.pad_token);
        for p in from..ids.len() {
            if ids[p] == self.pad_token || self.skip.contains(&ids[p]) {
                out.push(None);
                continue;
            }
            let mut rolling = 0u64;
            let mut blocked = false;
            let mut row = Vec::with_capacity(self.heads());
            for shift in 0..self.max_ngram {
                blocked |= shift > p;
                let src = if blocked { pad } else { self.map(ids[p - shift]) };
                rolling ^= src.wrapping_mul(self.multipliers[shift]);
                if shift > 0 {
                    for h in 0..self.heads_per_ngram {
                        let c = (shift - 1) * self.heads_per_ngram + h;
                        row.push(rolling % self.primes[c] + self.offsets[c]);
                    }
                }
            }
            out.push(Some(row));
        }
    }
}

/// Строки engram-таблицы слоя в F32.
pub trait TableRows: Send + Sync {
    fn dim(&self) -> usize;
    fn gather(&self, row: u64, out: &mut [f32]) -> Result<(), ModelError>;
}

/// Таблица MXFP8 в бандле: коды E4M3 `[rows, dim]` и масштабы E8M0 `[rows, dim/32]`.
pub struct Mxfp8Rows {
    raw: Arc<SafetensorsLoader>,
    packed: String,
    scales: String,
    dim: usize,
    lut: [f32; 256],
}

/// BF16-таблица (каталог HF или бандл без кванта).
pub struct Bf16Rows {
    raw: Arc<SafetensorsLoader>,
    name: String,
    dim: usize,
}

fn e4m3_lut() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        let s = if i & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = ((i >> 3) & 0xF) as i32;
        let m = (i & 7) as f32;
        *v = if e == 0xF && (i & 7) == 7 {
            f32::NAN
        } else if e == 0 {
            s * m / 8.0 * 2f32.powi(-6)
        } else {
            s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
        };
    }
    t
}

impl TableRows for Mxfp8Rows {
    fn dim(&self) -> usize {
        self.dim
    }
    fn gather(&self, row: u64, out: &mut [f32]) -> Result<(), ModelError> {
        let e = |m: &str| ModelError::Forward(format!("engram: {m}"));
        let (p, _, _) = self.raw.raw_bytes(&self.packed).ok_or_else(|| e("нет кодов таблицы"))?;
        let (s, _, _) = self.raw.raw_bytes(&self.scales).ok_or_else(|| e("нет масштабов таблицы"))?;
        let r = row as usize;
        let nb = self.dim / 32;
        let codes = p.get(r * self.dim..(r + 1) * self.dim).ok_or_else(|| e("строка вне таблицы"))?;
        let sc = s.get(r * nb..(r + 1) * nb).ok_or_else(|| e("масштаб вне таблицы"))?;
        for (i, (o, c)) in out.iter_mut().zip(codes).enumerate() {
            let k = sc[i / 32] as i32 - 127;
            *o = self.lut[*c as usize] * 2f32.powi(k);
        }
        Ok(())
    }
}

impl TableRows for Bf16Rows {
    fn dim(&self) -> usize {
        self.dim
    }
    fn gather(&self, row: u64, out: &mut [f32]) -> Result<(), ModelError> {
        let (b, _, _) = self.raw.raw_bytes(&self.name).ok_or_else(|| ModelError::Forward("engram: нет таблицы".into()))?;
        let r = row as usize * self.dim * 2;
        let src = b.get(r..r + self.dim * 2).ok_or_else(|| ModelError::Forward("engram: строка вне таблицы".into()))?;
        for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
            *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
        }
        Ok(())
    }
}

/// Источник строк таблицы слоя `name` (`…multi_head_embedding.embedding.weight`).
pub fn table_rows(raw: &Arc<SafetensorsLoader>, name: &str, dim: usize) -> Result<Box<dyn TableRows>, ModelError> {
    let packed = format!("{name}.qpacked");
    let scales = format!("{name}.qscales");
    if raw.raw_bytes(&packed).is_some() && raw.raw_bytes(&scales).is_some() {
        raw.advise_random(&packed);
        raw.advise_random(&scales);
        return Ok(Box::new(Mxfp8Rows { raw: raw.clone(), packed, scales, dim, lut: e4m3_lut() }));
    }
    match raw.raw_bytes(name) {
        Some((_, dt, shape)) if dt == synaptix_core::dtype::DType::BF16 && shape.len() == 2 && shape[1] == dim => {
            raw.advise_random(name);
            Ok(Box::new(Bf16Rows { raw: raw.clone(), name: name.to_string(), dim }))
        }
        Some((_, dt, shape)) => Err(ModelError::Load(format!("{name}: {dt:?} {shape:?} — ожидались MXFP8-блобы или BF16 [rows, {dim}]"))),
        None => Err(ModelError::Load(format!("{name}: нет в весах"))),
    }
}

/// Хэш, таблица и нормы голов одного слоя: вход проекций `[T, heads·dim]`.
pub struct EngramLookup {
    pub hash: EngramHash,
    pub table: Box<dyn TableRows>,
    /// `lookup_per_head_norms.h.weight` подряд: `[heads, dim]`.
    pub head_norm: Vec<f32>,
}

impl EngramLookup {
    /// Нормированные строки позиций `from..ids.len()` и признак «есть память».
    pub fn embed(&self, ids: &[u32], from: usize) -> Result<(Vec<f32>, Vec<bool>), ModelError> {
        let mut rows = Vec::new();
        self.hash.rows(ids, from, &mut rows);
        let (heads, dim) = (self.hash.heads(), self.table.dim());
        let width = heads * dim;
        let mut out = vec![0f32; rows.len() * width];
        let live: Vec<bool> = rows.iter().map(|r| r.is_some()).collect();
        // Параллельно по (токен, голова): на декоде это 16 случайных чтений
        // строк таблицы с диска разом, а не по очереди.
        use rayon::prelude::*;
        out.par_chunks_mut(dim).enumerate().try_for_each(|(i, x)| -> Result<(), ModelError> {
            let (t, h) = (i / heads, i % heads);
            let Some(r) = &rows[t] else { return Ok(()) };
            self.table.gather(r[h], x)?;
            let ms = x.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let rs = 1.0 / (ms + f32::EPSILON).sqrt();
            for (v, g) in x.iter_mut().zip(&self.head_norm[h * dim..(h + 1) * dim]) {
                *v *= rs * g;
            }
            Ok(())
        })?;
        Ok((out, live))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Хэш на крошечном словаре против прямого счёта по формуле эталона.
    #[test]
    fn hash_matches_reference_formula() {
        let cfg = EngramConfig {
            layers: vec![1],
            layer_ids: vec![2],
            max_ngram: 3,
            heads_per_ngram: 2,
            head_dim: 4,
            head_vocab: vec![vec![7, 11, 13, 17]],
            pad_id: 9,
            seed: 0,
            fp8_embedding: false,
        };
        let map: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 3];
        let mult = [5u64, 7, 9];
        let h = EngramHash::new(&cfg, 0, Arc::new(map.clone()), mult);
        let ids = [4u32, 9, 2, 6];
        let mut out = Vec::new();
        h.rows(&ids, 0, &mut out);
        assert!(out[1].is_none(), "заполнитель без памяти");
        let m = |i: isize| -> u64 { if i < 0 { map[9] as u64 } else { map[ids[i as usize] as usize] as u64 } };
        for p in [0usize, 2, 3] {
            let r = out[p].as_ref().unwrap();
            let (a, b, c) = (m(p as isize) * 5, m(p as isize - 1) * 7, m(p as isize - 2) * 9);
            let two = a ^ b;
            let three = a ^ b ^ c;
            assert_eq!(r, &vec![two % 7, two % 11 + 7, three % 13 + 18, three % 17 + 31], "позиция {p}");
        }
    }
}
