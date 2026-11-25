//! Спекулятивный драфтер DFly (DFly2) Hy4-flash: 5 слоёв MLA без iHC, которые
//! смотрят в латентный кэш и выбор групп индексатора целевых слоёв 25…41.
//!
//! Шаг: блок `[bonus, mask × k]` на позициях `a … a+k` (`a` — позиция за
//! последним принятым токеном) проходит слои драфтера одним проходом с
//! каузальной маской. Строка 0 перед этим смешивается со скрытым состоянием
//! цели (`mtp_fusion`, slot0). Драфты — из строк 1…k через `hidden_correction`
//! с эмбеддингом предыдущего токена цепочки и lm_head цели. Эмбеддинги и
//! голова — цели. Латенты блока пишутся в кэш целевых слоёв на позиции блока;
//! верификация целью их перезаписывает. Спецификация — `docs/hy4_flash.md`.

use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::{QLinear, WeightSource};

use crate::config::{AttnKind, DflyConfig, Hy4Config};
use crate::kern::{self, fe, f32c, R};
use crate::model::{f32t, fwd, lin, load_attn, Attn, ExtKeys, Hy4Model, ModelCache};

const ACT: DType = DType::BF16;

struct DLayer {
    in_norm: Tensor,
    attn: Attn,
    post_norm: Tensor,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
    /// Целевой слой: его латентный кэш и выбор групп.
    target: usize,
}

pub struct Dfly {
    pub cfg: DflyConfig,
    layers: Vec<DLayer>,
    final_norm: Tensor,
    f_enorm: Tensor,
    f_hnorm: Tensor,
    f_eh: QLinear,
    f_gate: Tensor,
    c_hnorm: Tensor,
    c_enorm: Tensor,
    c_gate: QLinear,
    c_up: QLinear,
    c_down: QLinear,
    eps: f32,
    kpool: usize,
}

impl Dfly {
    /// Драфтер из бандла, если в конфиге и весах он есть.
    pub fn load(w: &dyn WeightSource, cfg: &Hy4Config, dev: synaptix_core::device::Device) -> R<Option<Self>> {
        let Some(dc) = cfg.dfly.clone() else { return Ok(None) };
        let p = "model.dfly_layers";
        if !w.contains(&format!("{p}.final_norm.weight")) {
            return Ok(None);
        }
        let mut layers = Vec::with_capacity(dc.layers);
        for (i, &target) in dc.target_layers.iter().enumerate() {
            let l = format!("{p}.{i}");
            layers.push(DLayer {
                in_norm: f32t(w, &format!("{l}.input_layernorm.weight"), dev)?,
                attn: load_attn(w, &format!("{l}.self_attn"), *cfg.dims(target), AttnKind::Indexed, cfg.kv_lora_rank, dev, false)?,
                post_norm: f32t(w, &format!("{l}.post_attention_layernorm.weight"), dev)?,
                gate: lin(w, &format!("{l}.mlp.gate_proj.weight"), dev)?,
                up: lin(w, &format!("{l}.mlp.up_proj.weight"), dev)?,
                down: lin(w, &format!("{l}.mlp.down_proj.weight"), dev)?,
                target,
            });
        }
        Ok(Some(Self {
            layers,
            final_norm: f32t(w, &format!("{p}.final_norm.weight"), dev)?,
            f_enorm: f32t(w, &format!("{p}.mtp_fusion.enorm.weight"), dev)?,
            f_hnorm: f32t(w, &format!("{p}.mtp_fusion.hnorm.weight"), dev)?,
            f_eh: lin(w, &format!("{p}.mtp_fusion.eh_proj.weight"), dev)?,
            f_gate: f32t(w, &format!("{p}.mtp_fusion.gate"), dev)?,
            c_hnorm: f32t(w, &format!("{p}.hidden_correction.hidden_norm.weight"), dev)?,
            c_enorm: f32t(w, &format!("{p}.hidden_correction.embed_norm.weight"), dev)?,
            c_gate: lin(w, &format!("{p}.hidden_correction.gate_proj.weight"), dev)?,
            c_up: lin(w, &format!("{p}.hidden_correction.up_proj.weight"), dev)?,
            c_down: lin(w, &format!("{p}.hidden_correction.down_proj.weight"), dev)?,
            eps: cfg.rms_norm_eps,
            kpool: cfg.index_kpool,
            cfg: dc,
        }))
    }

    fn rms(&self, x: &Tensor, w: &Tensor, dt: DType) -> R<Tensor> {
        kern::rms(&kern::contig(x)?, Some(w), self.eps, dt)
    }

    /// Ключи слоя драфтера по снимку целевого слоя: строка последнего
    /// принятого токена (позиция `a − 1`). `None` — снимка нет.
    fn keys(&self, cache: &ModelCache, target: usize, a: usize) -> R<Option<ExtKeys>> {
        let Some((groups, p0)) = &cache.layers[target].snap else { return Ok(None) };
        let src = a - 1;
        if src < *p0 {
            return Ok(None);
        }
        Ok(Some(match groups {
            None => ExtKeys::Dense,
            Some(g) => {
                let r = src - p0;
                if r >= g.dims()[0] {
                    return Ok(None);
                }
                let row = kern::contig(&g.narrow(0, r, 1).map_err(fe)?)?;
                ExtKeys::Groups { row, lo: src / self.kpool * self.kpool }
            }
        }))
    }

    /// `k` жадных драфтов после `bonus` (позиция `a`). `h_tgt` — финальное
    /// скрытое цели для позиции `a − 1` (`[1, H]`). `None` — драфтить нельзя
    /// (нет снимка выбора групп, кэш мал).
    pub fn draft(&self, m: &Hy4Model, cache: &mut ModelCache, a: usize, bonus: u32, h_tgt: &Tensor, k: usize) -> R<Option<Vec<u32>>> {
        let w = k + 1;
        if k == 0 || a == 0 || a + w > cache.max_seq {
            return Ok(None);
        }
        let mut keys = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            match self.keys(cache, l.target, a)? {
                Some(kk) => keys.push(kk),
                None => return Ok(None),
            }
        }
        let dev = m.device;
        let mut ids = vec![self.cfg.mask_token; w];
        ids[0] = bonus;
        let pos: Vec<i32> = (a..a + w).map(|p| p as i32).collect();
        let pos = Tensor::from_vec(pos, vec![w], dev).map_err(fe)?;
        // Строка 0: e + gate ⊙ eh_proj([rms(e; enorm) | rms(h; hnorm)]).
        let e = m.embed_rows(&ids)?;
        let e0 = kern::contig(&e.narrow(0, 0, 1).map_err(fe)?)?;
        let cat = Tensor::cat(&[&self.rms(&e0, &self.f_enorm, ACT)?, &self.rms(&f32c(h_tgt)?, &self.f_hnorm, ACT)?], 1).map_err(fe)?;
        let f = f32c(&fwd(&self.f_eh, &kern::contig(&cat)?)?)?;
        let e0 = e0.add(&f.broadcast_mul(&self.f_gate).map_err(fe)?).map_err(fe)?;
        let rest = kern::contig(&e.narrow(0, 1, k).map_err(fe)?)?;
        let mut h = kern::contig(&Tensor::cat(&[&kern::contig(&e0)?, &rest], 0).map_err(fe)?)?;
        let mut res: Option<Tensor> = None;
        for (l, kk) in self.layers.iter().zip(&keys) {
            let x = match &res {
                Some(r) => h.add(r).map_err(fe)?,
                None => h.clone(),
            };
            let hn = self.rms(&x, &l.in_norm, ACT)?;
            let y = m.attention(l.target, &l.attn, &hn, &pos, a, &mut cache.layers[l.target], Some(kk))?;
            let y = self.rms(&y, l.attn.post_norm(), DType::F32)?;
            let x = y.add(&x).map_err(fe)?;
            let hn = self.rms(&x, &l.post_norm, ACT)?;
            let g = fwd(&l.gate, &hn)?;
            let u = fwd(&l.up, &hn)?;
            let act = g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|g| g.mul(&u))).map_err(fe)?;
            h = f32c(&fwd(&l.down, &act)?)?;
            res = Some(x);
        }
        let hfin = h.add(res.as_ref().expect("слои есть")).map_err(fe)?;
        let hh = self.rms(&hfin, &self.final_norm, DType::F32)?;
        // Цепочка: c = H[s+1] + down(silu(g)⊙u), z = [rms(H; hidden_norm) | rms(embed(prev); embed_norm)].
        let mut out = Vec::with_capacity(k);
        let mut prev = bonus;
        for s in 0..k {
            let hs = kern::contig(&hh.narrow(0, s + 1, 1).map_err(fe)?)?;
            let ep = m.embed_rows(&[prev])?;
            let z = kern::contig(&Tensor::cat(&[&self.rms(&hs, &self.c_hnorm, ACT)?, &self.rms(&ep, &self.c_enorm, ACT)?], 1).map_err(fe)?)?;
            let g = fwd(&self.c_gate, &z)?;
            let u = fwd(&self.c_up, &z)?;
            let act = g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|g| g.mul(&u))).map_err(fe)?;
            let c = hs.add(&f32c(&fwd(&self.c_down, &act)?)?).map_err(fe)?;
            let logits = m.logits(&kern::to(&c, ACT)?)?;
            prev = argmax(&logits)?;
            out.push(prev);
        }
        Ok(Some(out))
    }
}

/// Индекс максимума строки логитов `[1, V]`.
pub fn argmax(logits: &Tensor) -> R<u32> {
    Ok(argmax_rows(logits)?[0])
}

/// Индексы максимумов строк логитов `[T, V]` (редукция на карте, на хост — T чисел).
pub fn argmax_rows(logits: &Tensor) -> R<Vec<u32>> {
    let idx = logits.argmax(1).map_err(fe)?;
    // Индекс словаря < 2^24 — в F32 точен при любом dtype редукции.
    let v = idx.to_dtype(DType::F32).and_then(|t| t.to_device(synaptix_core::device::Device::Cpu)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(fe)?;
    Ok(v.into_iter().map(|x| x as u32).collect())
}
