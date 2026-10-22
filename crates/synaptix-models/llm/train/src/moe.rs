//! MoE-FFN слоя на обучении (Qwen3.5/3.6-MoE):
//!
//! ```text
//! w, idx = top_k(softmax(x · Rᵀ))           # веса нормированы на сумму top-k
//! y      = Σ_i w_i · down_i(silu(gate_i(x)) · up_i(x))
//! y     += sigmoid(x · gᵀ) · shared(x)
//! ```
//!
//! Роутер и эксперты заморожены, адаптеры — на проекциях shared-эксперта.
//! Градиент по входу идёт через всё: экспертов, веса роутера (top-k
//! нормированный softmax — это softmax по выбранным логитам) и гейт
//! shared-эксперта.
//!
//! Пары «токен — эксперт» сортируются по эксперту, и каждый эксперт получает
//! один GEMM на свои строки. Вес эксперта деквантуется на время его GEMM и
//! не кэшируется: плотная копия всех экспертов слоя — гигабайты.

use synaptix_core::backend::train::TrainOp;
use synaptix_core::device::Device;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::{MoeFfn, TrainExpert};

use crate::block::{proj, proj_bwd, BlockLora};
use crate::error::other;
use crate::linear::{backward_input_with, dense_weight, Weights};
use crate::Result;

/// Что прямой проход MoE оставляет обратному.
pub struct MoeSaved {
    /// Вход `[T, H]` в compute-типе.
    x: Tensor,
    k: usize,
    /// Вес пары в порядке сортировки по эксперту `[P]`.
    w_sorted: Vec<f32>,
    /// `order[j]` — пара `t·k + s`, посчитанная на позиции `j`.
    order: Vec<u32>,
    /// `(эксперт, начало, конец)` в отсортированном порядке.
    segs: Vec<(usize, usize, usize)>,
    /// Строка токена пары `[P]` U32.
    rows: Tensor,
    /// Позиция пары в отсортированном порядке `[P]` U32 (обратная перестановка).
    inverse: Tensor,
    /// `x·GUᵀ` пар `[P, 2I]` (gate | up), compute-тип.
    gu: Tensor,
    /// Выходы экспертов без веса `[P, H]`, compute-тип.
    y: Tensor,
    shared: Option<SharedSaved>,
}

/// Адреса весов экспертов слоя под групповые GEMM обучения: запись `j` —
/// эксперт сегмента `j`.
struct GroupTable {
    w: Tensor,
    s: Tensor,
    dtype: DType,
    shuffled: bool,
    n: usize,
    k: usize,
}

impl GroupTable {
    fn build(ws: &[&QuantWeight], dev: Device) -> Option<Self> {
        let first = ws.first()?;
        let (dtype, n, k) = (first.dtype(), first.n(), first.k());
        if k % 64 != 0 {
            return None;
        }
        let mut w = Vec::with_capacity(ws.len());
        let mut s = Vec::with_capacity(ws.len());
        let mut shuffled = None;
        for q in ws {
            if q.dtype() != dtype || q.n() != n || q.k() != k {
                return None;
            }
            let (wa, sa, sh) = q.blob_addrs()?;
            if *shuffled.get_or_insert(sh) != sh {
                return None;
            }
            w.push(wa as i64);
            s.push(sa as i64);
        }
        let len = w.len();
        Some(Self {
            w: Tensor::from_vec(w, vec![len], dev).ok()?,
            s: Tensor::from_vec(s, vec![len], dev).ok()?,
            dtype,
            shuffled: shuffled.unwrap_or(false),
            n,
            k,
        })
    }

    /// `y[r] = x[src(r)] · W_eᵀ` → `[R, N]` BF16.
    fn nt(&self, x: &Tensor, rows: Option<&Tensor>, tiles: &Tensor, r: usize) -> Result<Tensor> {
        let op = TrainOp::MoeGroupedNt { dtype: self.dtype, shuffled: self.shuffled, has_rows: rows.is_some() };
        let mut ins = vec![&self.w, &self.s, tiles, x];
        if let Some(rw) = rows {
            ins.push(rw);
        }
        Ok(Tensor::train_op(op, &ins, &[(vec![r, self.n], DType::BF16)])?.remove(0))
    }

    /// `dx[r] = dy[r] · W_e` → `[R, K]` F32; `dy` — `[R, N]` BF16.
    fn nn(&self, dy: &Tensor, tiles: &Tensor) -> Result<Tensor> {
        let op = TrainOp::MoeGroupedNn { dtype: self.dtype, shuffled: self.shuffled };
        let r = dy.dims()[0];
        Ok(Tensor::train_op(op, &[&self.w, &self.s, tiles, dy], &[(vec![r, self.k], DType::F32)])?.remove(0))
    }
}

/// Таблицы (gate_up, down). `None` — групповой путь неприменим (плотная
/// база, F32-счёт, формат без декодера) — считается поштучно.
fn grouped_tables(experts: &[TrainExpert<'_>], x: &Tensor) -> Option<(GroupTable, GroupTable)> {
    if x.dtype() != DType::BF16 || std::env::var("SYN_TRAIN_MOE_GROUPED").as_deref() == Ok("0") {
        return None;
    }
    let gu: Vec<_> = experts.iter().map(|e| e.gate_up().quant_weight()).collect::<Option<Vec<_>>>()?;
    let dn: Vec<_> = experts.iter().map(|e| e.down().quant_weight()).collect::<Option<Vec<_>>>()?;
    Some((GroupTable::build(&gu, x.device())?, GroupTable::build(&dn, x.device())?))
}

/// Тайлы строк по 64: `(эксперт таблицы, начало, конец, 0)` U32.
fn tiles(segs: &[(usize, usize, usize)], dev: Device) -> Result<Tensor> {
    let mut t: Vec<u32> = Vec::new();
    for (j, &(_, a, b)) in segs.iter().enumerate() {
        let mut r = a;
        while r < b {
            let e = (r + 64).min(b);
            t.extend_from_slice(&[j as u32, r as u32, e as u32, 0]);
            r = e;
        }
    }
    let n = t.len();
    Ok(Tensor::from_vec(t, vec![n], dev)?)
}

struct SharedSaved {
    zg: Option<Tensor>,
    zu: Option<Tensor>,
    g: Tensor,
    u: Tensor,
    m: Option<Tensor>,
    zd: Option<Tensor>,
    /// Выход shared-эксперта до гейта `[T, H]`.
    d: Tensor,
    /// `sigmoid(x · gᵀ)` `[T]`.
    sg: Vec<f32>,
}

fn host_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

fn take_rows(src: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match src.embed_gather(ids) {
        Ok(t) => Ok(t),
        Err(synaptix_core::error::SynaptixError::Unsupported(_)) => Ok(src.index_select(0, ids)?),
        Err(e) => Err(e.into()),
    }
}

/// Сумма `[P, H]` (пары в порядке `t·k + s`) по `k` слотам → `[T, H]`.
fn sum_slots(x: &Tensor, t: usize, k: usize) -> Result<Tensor> {
    let h = x.dims()[1];
    Ok(x.reshape(vec![t, k, h])?.sum([1usize])?)
}

fn gate_up_split(gu: &Tensor, i: usize) -> Result<(Tensor, Tensor)> {
    Ok((gu.narrow(1, 0, i)?.contiguous()?, gu.narrow(1, i, i)?.contiguous()?))
}

/// Прямой проход. `x` — `[T, H]` (выход нормы перед FFN) в compute-типе.
pub fn forward(moe: &MoeFfn, lora: &BlockLora, x: &Tensor, w: &Weights) -> Result<(Tensor, MoeSaved)> {
    let dev = x.device();
    let cfg = moe.config();
    let (t, h) = (x.dims()[0], x.dims()[1]);
    let (e, k, inter) = (cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size);

    // ── Роутер ──
    let (sel, wts) = crate::prof::prof(dev, "moe_route", || -> Result<(Vec<u32>, Vec<f32>)> {
        let logits = x.to_dtype(DType::F32)?.linear(moe.train_router())?; // [T, E]
        let lg = host_f32(&logits)?;
        let mut sel = vec![0u32; t * k];
        let mut wts = vec![0f32; t * k];
        for r in 0..t {
            let row = &lg[r * e..(r + 1) * e];
            let mut idx: Vec<u32> = (0..e as u32).collect();
            idx.select_nth_unstable_by(k - 1, |a, b| row[*b as usize].total_cmp(&row[*a as usize]));
            let top = &mut idx[..k];
            top.sort_unstable_by(|a, b| row[*b as usize].total_cmp(&row[*a as usize]));
            let max = row[top[0] as usize];
            let ex: Vec<f32> = top.iter().map(|j| (row[*j as usize] - max).exp()).collect();
            let den: f32 = if cfg.norm_topk_prob { ex.iter().sum() } else { row.iter().map(|v| (v - max).exp()).sum() };
            for s in 0..k {
                sel[r * k + s] = top[s];
                wts[r * k + s] = ex[s] / den;
            }
        }
        Ok((sel, wts))
    })?;

    // ── Пары по экспертам ──
    let p = t * k;
    let mut order: Vec<u32> = (0..p as u32).collect();
    order.sort_unstable_by_key(|&q| (sel[q as usize], q));
    let mut segs = Vec::new();
    let mut pos = 0;
    while pos < p {
        let ex = sel[order[pos] as usize];
        let mut end = pos + 1;
        while end < p && sel[order[end] as usize] == ex {
            end += 1;
        }
        segs.push((ex as usize, pos, end));
        pos = end;
    }
    let mut inverse = vec![0u32; p];
    for (j, q) in order.iter().enumerate() {
        inverse[*q as usize] = j as u32;
    }
    let w_sorted: Vec<f32> = order.iter().map(|q| wts[*q as usize]).collect();
    let rows = Tensor::from_vec(order.iter().map(|q| q / k as u32).collect::<Vec<u32>>(), vec![p], dev)?;
    let inverse = Tensor::from_vec(inverse, vec![p], dev)?;

    let ids: Vec<u32> = segs.iter().map(|s| s.0 as u32).collect();
    moe.train_prefetch(&ids);

    // ── Эксперты ──
    let experts: Vec<TrainExpert<'_>> = crate::prof::prof(dev, "moe_take", || segs.iter().map(|s| moe.train_expert(s.0)).collect::<std::result::Result<_, _>>())?;
    let tables = crate::prof::prof(dev, "moe_tables", || grouped_tables(&experts, x));
    let (gu, y) = match tables {
        Some((t_gu, t_dn)) => crate::prof::prof(dev, "moe_experts_fwd", || -> Result<(Tensor, Tensor)> {
            // Строки токенов собирает само ядро — без копии [P, H].
            let tl = tiles(&segs, dev)?;
            let xc = x.contiguous()?;
            let gu = crate::prof::prof(dev, "gg_gu", || t_gu.nt(&xc, Some(&rows), &tl, p))?;
            let (g, u) = gate_up_split(&gu, inter)?;
            let hdn = g.silu_and_mul(&u)?;
            let y = crate::prof::prof(dev, "gg_dn", || t_dn.nt(&hdn, None, &tl, p))?;
            Ok((gu, y))
        })?,
        None => {
            let xg = take_rows(x, &rows)?; // [P, H]
            let mut gus = Vec::with_capacity(segs.len());
            let mut ys = Vec::with_capacity(segs.len());
            crate::prof::prof(dev, "moe_experts_fwd", || -> Result<()> {
                for (expert, &(_, a, b)) in experts.iter().zip(&segs) {
                    let xe = xg.narrow(0, a, b - a)?;
                    let wgu = dense_weight(expert.gate_up())?;
                    let gu = xe.to_dtype(wgu.dtype())?.linear(&wgu)?;
                    drop(wgu);
                    let (g, u) = gate_up_split(&gu, inter)?;
                    let hdn = g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|s| s.mul(&u)))?;
                    let wd = dense_weight(expert.down())?;
                    let y = hdn.linear(&wd)?;
                    gus.push(gu);
                    ys.push(y);
                }
                Ok(())
            })?;
            let cat = |v: &[Tensor]| -> Result<Tensor> {
                if v.len() == 1 {
                    return Ok(v[0].clone());
                }
                let r: Vec<&Tensor> = v.iter().collect();
                Ok(Tensor::cat(&r, 0)?)
            };
            (cat(&gus)?, cat(&ys)?)
        }
    };
    drop(experts);
    let y = y.to_dtype(x.dtype())?;

    let ws = Tensor::from_vec(w_sorted.clone(), vec![p, 1], dev)?.to_dtype(y.dtype())?;
    let mixed = sum_slots(&take_rows(&y.broadcast_mul(&ws)?, &inverse)?, t, k)?;

    // ── Shared-эксперт ──
    let (out, shared) = match moe.train_shared() {
        Some(sh) => {
            let (g, zg) = proj(w, sh.gate, lora.gate.as_ref(), x, None)?;
            let (u, zu) = proj(w, sh.up, lora.up.as_ref(), x, None)?;
            let m = g.silu_and_mul(&u).or_else(|_| g.silu().and_then(|s| s.mul(&u)))?;
            let (d, zd) = proj(w, sh.down, lora.down.as_ref(), &m, None)?;
            let sg = x.to_dtype(DType::F32)?.linear(sh.gate_w)?.sigmoid()?; // [T, 1]
            let out = mixed.add(&d.broadcast_mul(&sg.to_dtype(d.dtype())?)?.to_dtype(mixed.dtype())?)?;
            let saved = SharedSaved {
                zg,
                zu,
                g,
                u,
                m: lora.down.is_some().then_some(m),
                zd,
                d,
                sg: host_f32(&sg)?,
            };
            (out, Some(saved))
        }
        None => (mixed, None),
    };
    let _ = h;
    Ok((out, MoeSaved { x: x.clone(), k, w_sorted, order, segs, rows, inverse, gu, y, shared }))
}

/// Обратный проход: `dy` — `[T, H]` F32, результат — `dx` F32.
pub fn backward(moe: &MoeFfn, lora: &mut BlockLora, s: MoeSaved, dy: &Tensor, w: &Weights) -> Result<Tensor> {
    let dev = dy.device();
    let cfg = moe.config();
    let (t, h) = (dy.dims()[0], dy.dims()[1]);
    let (e, k, inter) = (cfg.num_experts, s.k, cfg.moe_intermediate_size);
    let p = t * k;

    // ── Эксперты ──
    let dyg = take_rows(dy, &s.rows)?; // [P, H] F32, отсортировано
    // dL/dw пары = <dy токена, выход эксперта>.
    let dw_sorted = host_f32(&dyg.mul(&s.y.to_dtype(DType::F32)?)?.sum_keepdim(1)?)?;
    let ws = Tensor::from_vec(s.w_sorted.clone(), vec![p, 1], dev)?;
    let dyw = dyg.broadcast_mul(&ws)?;
    drop(dyg);
    let experts: Vec<TrainExpert<'_>> = crate::prof::prof(dev, "moe_take", || s.segs.iter().map(|g| moe.train_expert(g.0)).collect::<std::result::Result<_, _>>())?;
    let tables = crate::prof::prof(dev, "moe_tables", || grouped_tables(&experts, &s.x));
    let dx_sorted = match tables {
        Some((t_gu, t_dn)) => crate::prof::prof(dev, "moe_experts_bwd", || -> Result<Tensor> {
            let tl = tiles(&s.segs, dev)?;
            let dh = crate::prof::prof(dev, "ggn_dn", || t_dn.nn(&dyw.to_dtype(DType::BF16)?, &tl))?; // [P, I]
            let (g, u) = gate_up_split(&s.gu, inter)?;
            let shp = g.dims().to_vec();
            let gu = Tensor::train_op(TrainOp::SiluMulBwd, &[&g, &u, &dh], &[(shp.clone(), DType::F32), (shp, DType::F32)])?;
            let dgu = Tensor::cat(&[&gu[0], &gu[1]], 1)?.to_dtype(DType::BF16)?;
            crate::prof::prof(dev, "ggn_gu", || t_gu.nn(&dgu, &tl)) // [P, H]
        })?,
        None => {
            let mut dxs = Vec::with_capacity(s.segs.len());
            crate::prof::prof(dev, "moe_experts_bwd", || -> Result<()> {
                for (expert, &(_, a, b)) in experts.iter().zip(&s.segs) {
                    let n = b - a;
                    let wd = dense_weight(expert.down())?;
                    let dh = backward_input_with(&wd, &dyw.narrow(0, a, n)?.contiguous()?)?; // [n, I]
                    drop(wd);
                    let (g, u) = gate_up_split(&s.gu.narrow(0, a, n)?, inter)?;
                    let shp = g.dims().to_vec();
                    let gu = Tensor::train_op(TrainOp::SiluMulBwd, &[&g, &u, &dh], &[(shp.clone(), DType::F32), (shp, DType::F32)])?;
                    let dgu = Tensor::cat(&[&gu[0], &gu[1]], 1)?;
                    let wgu = dense_weight(expert.gate_up())?;
                    dxs.push(backward_input_with(&wgu, &dgu)?); // [n, H]
                }
                Ok(())
            })?;
            if dxs.len() == 1 {
                dxs.pop().unwrap()
            } else {
                let r: Vec<&Tensor> = dxs.iter().collect();
                Tensor::cat(&r, 0)?
            }
        }
    };
    drop(experts);
    let mut dx = sum_slots(&take_rows(&dx_sorted, &s.inverse)?, t, k)?;
    drop(dx_sorted);

    // ── Роутер ──
    // При нормировке top-k вес пары — softmax по выбранным логитам:
    // dl_i = w_i · (dw_i − Σ_j w_j·dw_j); у невыбранных экспертов 0.
    if !cfg.norm_topk_prob {
        return Err(other("обучение: MoE без нормировки top-k пока не поддержано"));
    }
    let mut dl = vec![0f32; t * e];
    let mut wp = vec![0f32; p];
    let mut dwp = vec![0f32; p];
    let mut ep = vec![0u32; p];
    for (j, q) in s.order.iter().enumerate() {
        let q = *q as usize;
        wp[q] = s.w_sorted[j];
        dwp[q] = dw_sorted[j];
    }
    for &(ex, a, b) in &s.segs {
        for j in a..b {
            ep[s.order[j] as usize] = ex as u32;
        }
    }
    for r in 0..t {
        let base = r * k;
        let sum: f32 = (0..k).map(|i| wp[base + i] * dwp[base + i]).sum();
        for i in 0..k {
            dl[r * e + ep[base + i] as usize] += wp[base + i] * (dwp[base + i] - sum);
        }
    }
    let dl = Tensor::from_vec(dl, vec![t, e], dev)?;
    dx = dx.add(&dl.matmul(moe.train_router())?)?;

    // ── Shared-эксперт ──
    if let (Some(sh), Some(ss)) = (moe.train_shared(), s.shared) {
        let sg = Tensor::from_vec(ss.sg.clone(), vec![t, 1], dev)?;
        let dd = dy.broadcast_mul(&sg)?;
        let dsg = host_f32(&dy.mul(&ss.d.to_dtype(DType::F32)?)?.sum_keepdim(1)?)?;
        let dsl: Vec<f32> = dsg.iter().zip(&ss.sg).map(|(d, g)| d * g * (1.0 - g)).collect();
        let dsl = Tensor::from_vec(dsl, vec![t, 1], dev)?;
        let gw = if sh.gate_w.dtype() == DType::F32 { sh.gate_w.clone() } else { sh.gate_w.to_dtype(DType::F32)? };
        dx = dx.add(&dsl.broadcast_mul(&gw.reshape(vec![1, h])?)?)?;
        let dm = proj_bwd(w, sh.down, lora.down.as_mut(), ss.m.as_ref(), ss.zd.as_ref(), &dd)?;
        let shp = ss.g.dims().to_vec();
        let gu = Tensor::train_op(TrainOp::SiluMulBwd, &[&ss.g, &ss.u, &dm], &[(shp.clone(), DType::F32), (shp, DType::F32)])?;
        let x_in = (lora.gate.is_some() || lora.up.is_some()).then_some(&s.x);
        dx = dx
            .add(&proj_bwd(w, sh.gate, lora.gate.as_mut(), x_in, ss.zg.as_ref(), &gu[0])?)?
            .add(&proj_bwd(w, sh.up, lora.up.as_mut(), x_in, ss.zu.as_ref(), &gu[1])?)?;
    }
    Ok(dx)
}
