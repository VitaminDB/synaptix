//! Слитые операции шага декода (T = 1) поверх [`Backend`]: хвосты слоя с
//! нормами и квант-эпилогами, роутер MoE, geglu с квантом, подготовка
//! внимания, групповые и индексные GEMV. Выходы аллоцируются uninit — ядра
//! пишут их целиком (у NVFP4-пар — только строки активации; хвост 128-тайла
//! масштабов GEMV не читает).

use std::sync::Arc;

use crate::backend::{registry, DecIndexedOut, DecNormOut, DecNormSpec, DecTopk};
use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::stream::Stream;
use crate::tensor::layout::Layout;
use crate::tensor::quant::{ExpertTable, QuantWeight};
use crate::tensor::shape::Shape;
use crate::tensor::storage::Storage;
use crate::tensor::Tensor;

/// В каком виде нужна норма от `hidden`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecOutFmt {
    Bf16,
    Nvfp4,
    Mxfp8,
}

/// Выход нормы: bf16-строка либо квант-пара `(packed, scales)`.
pub enum DecOut {
    Bf16(Tensor),
    Quant(Tensor, Tensor),
}

impl DecOut {
    pub fn pair(&self) -> Option<(&Tensor, &Tensor)> {
        match self {
            DecOut::Quant(p, s) => Some((p, s)),
            DecOut::Bf16(_) => None,
        }
    }
}

fn u8_layout(bytes: usize) -> Layout {
    Layout::contiguous(Shape::new(vec![bytes]), DType::U8)
}

/// Размеры NVFP4-пары активации `m×k` (синхронно с `nvfp4_quantize_act`).
pub fn nvfp4_pair_bytes(m: usize, k: usize) -> (usize, usize) {
    (m * k / 2, (k.div_ceil(64) * 4) * (m.div_ceil(128) * 128))
}

fn row_tensor(t: &Tensor, what: &str) -> Result<Tensor> {
    if !t.is_contiguous() || t.layout.byte_offset() != 0 {
        return t.contiguous();
    }
    let _ = what;
    Ok(t.clone())
}

fn alloc(backend: &'static dyn crate::backend::Backend, dev: crate::device::Device, bytes: usize) -> Result<Storage> {
    backend.alloc_uninit(bytes.max(1), dev)
}

impl Tensor {
    /// Хвост после внимания. `self` — выход `o_proj` `[.., H]` (bf16),
    /// `hidden_in` — residual той же формы. Возвращает новый `hidden` и по
    /// выходу на каждую пару `(вес, формат)` из `outs`.
    pub fn dec_attn_tail(
        &self,
        post_w: Option<&Tensor>,
        hidden_in: &Tensor,
        outs: &[(&Tensor, DecOutFmt)],
        eps_post: f32,
        eps: f32,
    ) -> Result<(Tensor, Vec<DecOut>)> {
        let h = *self.dims().last().ok_or(SynaptixError::Unsupported("dec_attn_tail: scalar"))?;
        if self.layout.numel() != h || hidden_in.layout.numel() != h {
            return Err(SynaptixError::Unsupported("dec_attn_tail: одна строка [.., H]"));
        }
        if self.dtype() != DType::BF16 || hidden_in.dtype() != DType::BF16 {
            return Err(SynaptixError::Unsupported("dec_attn_tail: только BF16"));
        }
        let dev = self.device();
        let backend = registry::backend_for(dev)?;
        let x = row_tensor(self, "attn_out")?;
        let r = row_tensor(hidden_in, "hidden")?;
        let hidden_layout = hidden_in.layout.clone();
        let mut hidden_st = alloc(backend, dev, DType::BF16.bytes_for_numel(h))?;

        // Буферы выходов по форматам.
        let mut bufs: Vec<(Storage, Option<Storage>)> = Vec::with_capacity(outs.len());
        for (_, fmt) in outs {
            match fmt {
                DecOutFmt::Bf16 => bufs.push((alloc(backend, dev, DType::BF16.bytes_for_numel(h))?, None)),
                DecOutFmt::Nvfp4 => {
                    let (pb, sb) = nvfp4_pair_bytes(1, h);
                    bufs.push((alloc(backend, dev, pb)?, Some(alloc(backend, dev, sb)?)));
                }
                DecOutFmt::Mxfp8 => bufs.push((alloc(backend, dev, h)?, Some(alloc(backend, dev, h / 32)?))),
            }
        }
        {
            let mut specs: Vec<DecNormSpec<'_>> = Vec::with_capacity(outs.len());
            for ((w, fmt), (a, b)) in outs.iter().zip(bufs.iter_mut()) {
                let out = match (fmt, b) {
                    (DecOutFmt::Bf16, _) => DecNormOut::Bf16(a),
                    (DecOutFmt::Nvfp4, Some(s)) => DecNormOut::Nvfp4 { packed: a, scales: s },
                    (DecOutFmt::Mxfp8, Some(s)) => DecNormOut::Mxfp8 { packed: a, scales: s },
                    _ => unreachable!(),
                };
                specs.push(DecNormSpec { weight: &w.storage, out });
            }
            let stream = Stream::default_for(dev)?;
            backend.dec_attn_tail(
                &x.storage,
                post_w.map(|t| &*t.storage),
                &r.storage,
                &mut hidden_st,
                &mut specs,
                h,
                eps_post,
                eps,
                &stream,
            )?;
        }
        let hidden = Tensor::from_parts(Arc::new(hidden_st), hidden_layout);
        let mut res = Vec::with_capacity(outs.len());
        for ((_, fmt), (a, b)) in outs.iter().zip(bufs) {
            res.push(match (fmt, b) {
                (DecOutFmt::Bf16, _) => DecOut::Bf16(Tensor::from_parts(
                    Arc::new(a),
                    Layout::contiguous(Shape::new(vec![1usize, h]), DType::BF16),
                )),
                (_, Some(s)) => {
                    let (pb, sb) = match fmt {
                        DecOutFmt::Nvfp4 => nvfp4_pair_bytes(1, h),
                        _ => (h, h / 32),
                    };
                    DecOut::Quant(
                        Tensor::from_parts(Arc::new(a), u8_layout(pb)),
                        Tensor::from_parts(Arc::new(s), u8_layout(sb)),
                    )
                }
                _ => unreachable!(),
            });
        }
        Ok((hidden, res))
    }

    /// Хвост FFN-части блока. `self` — выход плотного MLP `[.., H]`;
    /// `moe_acc` — f32 `[H]` взвешенная сумма экспертов (или `None`);
    /// `next` — вес нормы входа следующего слоя и нужные форматы (bf16, MXFP8).
    /// Возвращает `(hidden, next_bf16, next_mxfp8)`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn dec_ffn_tail(
        &self,
        moe_acc: Option<&Tensor>,
        hidden_in: &Tensor,
        w_post_dense: Option<&Tensor>,
        w_post_moe: Option<&Tensor>,
        w_post_mlp: Option<&Tensor>,
        layer_scalar: f32,
        next: Option<(&Tensor, bool, bool)>,
        eps_post: f32,
        eps: f32,
    ) -> Result<(Tensor, Option<Tensor>, Option<(Tensor, Tensor)>)> {
        let h = *self.dims().last().ok_or(SynaptixError::Unsupported("dec_ffn_tail: scalar"))?;
        if self.layout.numel() != h || hidden_in.layout.numel() != h {
            return Err(SynaptixError::Unsupported("dec_ffn_tail: одна строка [.., H]"));
        }
        if self.dtype() != DType::BF16 || hidden_in.dtype() != DType::BF16 {
            return Err(SynaptixError::Unsupported("dec_ffn_tail: только BF16"));
        }
        if let Some(acc) = moe_acc {
            if acc.dtype() != DType::F32 || acc.layout.numel() != h {
                return Err(SynaptixError::Unsupported("dec_ffn_tail: moe_acc — F32[H]"));
            }
            if w_post_dense.is_none() || w_post_moe.is_none() {
                return Err(SynaptixError::Unsupported("dec_ffn_tail: MoE-ветке нужны обе пост-нормы"));
            }
        }
        let dev = self.device();
        let backend = registry::backend_for(dev)?;
        let x = row_tensor(self, "dense")?;
        let r = row_tensor(hidden_in, "hidden")?;
        let acc = match moe_acc {
            Some(a) => Some(row_tensor(a, "moe_acc")?),
            None => None,
        };
        let hidden_layout = hidden_in.layout.clone();
        let mut hidden_st = alloc(backend, dev, DType::BF16.bytes_for_numel(h))?;
        let (next_w, want_bf16, want_mx) = match next {
            Some((w, b, m)) => (Some(w), b, m),
            None => (None, false, false),
        };
        let mut nb = if want_bf16 { Some(alloc(backend, dev, DType::BF16.bytes_for_numel(h))?) } else { None };
        let mut nm = if want_mx { Some((alloc(backend, dev, h)?, alloc(backend, dev, h / 32)?)) } else { None };
        {
            let stream = Stream::default_for(dev)?;
            backend.dec_ffn_tail(
                &x.storage,
                acc.as_ref().map(|t| &*t.storage),
                &r.storage,
                w_post_dense.map(|t| &*t.storage),
                w_post_moe.map(|t| &*t.storage),
                w_post_mlp.map(|t| &*t.storage),
                layer_scalar,
                &mut hidden_st,
                next_w.map(|t| &*t.storage),
                nb.as_mut(),
                nm.as_mut().map(|(p, s)| (p, s)),
                h,
                eps_post,
                eps,
                &stream,
            )?;
        }
        let hidden = Tensor::from_parts(Arc::new(hidden_st), hidden_layout.clone());
        let next_bf16 = nb.map(|st| Tensor::from_parts(Arc::new(st), hidden_layout));
        let next_mx = nm.map(|(p, s)| {
            (
                Tensor::from_parts(Arc::new(p), u8_layout(h)),
                Tensor::from_parts(Arc::new(s), u8_layout(h / 32)),
            )
        });
        Ok((hidden, next_bf16, next_mx))
    }

    /// `gelu_tanh(gate)·up` → NVFP4-пара `rows × inter`. `gate`/`up` —
    /// (тензор, смещение в элементах), `stride` — шаг строки в элементах.
    /// `topk = Some((logits F32[e], pes, k, h))` — тем же запуском top-k
    /// роутера MoE: вернёт `(idx U32[k], w F32[k], acc F32[h] нулевой)`.
    #[allow(clippy::type_complexity)]
    pub fn dec_geglu_quant_nvfp4(
        gate: (&Tensor, usize),
        up: (&Tensor, usize),
        stride: usize,
        rows: usize,
        inter: usize,
        topk: Option<(&Tensor, Option<&Tensor>, usize, usize)>,
    ) -> Result<(Tensor, Tensor, Option<(Tensor, Tensor, Tensor)>)> {
        let (g, go) = gate;
        let (u, uo) = up;
        let dt = g.dtype();
        if u.dtype() != dt || !matches!(dt, DType::F16 | DType::BF16) {
            return Err(SynaptixError::Unsupported("dec_geglu_quant_nvfp4: gate/up F16|BF16 одного типа"));
        }
        if !g.is_contiguous() || !u.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let esz = dt.size_in_bits() / 8;
        let dev = g.device();
        let backend = registry::backend_for(dev)?;
        let (pb, sb) = nvfp4_pair_bytes(rows, inter);
        let mut packed = alloc(backend, dev, pb)?;
        let mut scales = alloc(backend, dev, sb)?;
        let mut tk_bufs = match topk {
            Some((logits, pes, k, h)) => {
                let e = logits.numel();
                if logits.dtype() != DType::F32 || !logits.is_contiguous() {
                    return Err(SynaptixError::Unsupported("dec_geglu_quant_nvfp4: логиты F32[e]"));
                }
                if let Some(p) = pes {
                    if p.numel() != e || p.dtype() != DType::F32 || !p.is_contiguous() {
                        return Err(SynaptixError::Unsupported("dec_geglu_quant_nvfp4: per_expert_scale F32[e]"));
                    }
                }
                Some((alloc(backend, dev, k * 4)?, alloc(backend, dev, k * 4)?, alloc(backend, dev, h * 4)?, e, k, h))
            }
            None => None,
        };
        {
            let stream = Stream::default_for(dev)?;
            let tk = match (&topk, &mut tk_bufs) {
                (Some((logits, pes, _, _)), Some((idx, w, acc, e, k, h))) => Some(DecTopk {
                    logits: &logits.storage,
                    pes: pes.map(|t| &*t.storage),
                    idx,
                    w,
                    acc,
                    e: *e,
                    k: *k,
                    h: *h,
                }),
                _ => None,
            };
            backend.dec_geglu_quant_nvfp4(
                (&g.storage, g.layout.byte_offset() + go * esz),
                (&u.storage, u.layout.byte_offset() + uo * esz),
                stride,
                dt,
                &mut packed,
                &mut scales,
                rows,
                inter,
                tk,
                &stream,
            )?;
        }
        let tk_out = tk_bufs.map(|(idx, w, acc, _, k, h)| {
            (
                Tensor::from_parts(Arc::new(idx), Layout::contiguous(Shape::new(vec![k]), DType::U32)),
                Tensor::from_parts(Arc::new(w), Layout::contiguous(Shape::new(vec![k]), DType::F32)),
                Tensor::from_parts(Arc::new(acc), Layout::contiguous(Shape::new(vec![h]), DType::F32)),
            )
        });
        Ok((
            Tensor::from_parts(Arc::new(packed), u8_layout(pb)),
            Tensor::from_parts(Arc::new(scales), u8_layout(sb)),
            tk_out,
        ))
    }

    /// Нормы голов + RoPE + запись K/V в кэш. `q`/`k`/`v` — (тензор bf16,
    /// смещение в элементах) со строками голов подряд; кэши — `[1, nkv,
    /// max_seq, hd]`. Возвращает `q` после нормы и RoPE `[1, nh, 1, hd]`.
    #[allow(clippy::too_many_arguments)]
    pub fn dec_attn_prep(
        q: (&Tensor, usize),
        k: (&Tensor, usize),
        v: Option<(&Tensor, usize)>,
        q_norm: Option<&Tensor>,
        k_norm: Option<&Tensor>,
        v_norm: bool,
        cos: &Tensor,
        sin: &Tensor,
        pos: &Tensor,
        rotary_dim: usize,
        kv_pos: &Tensor,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        nh: usize,
        nkv: usize,
        hd: usize,
        eps: f32,
    ) -> Result<Tensor> {
        let (qt, qo) = q;
        let (kt, ko) = k;
        if qt.dtype() != DType::BF16 || kt.dtype() != DType::BF16 {
            return Err(SynaptixError::Unsupported("dec_attn_prep: только BF16"));
        }
        if !qt.is_contiguous() || !kt.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let kd = k_cache.dims().to_vec();
        if kd.len() != 4 || kd[0] != 1 || kd[1] != nkv || kd[3] != hd || v_cache.dims() != kd.as_slice() {
            return Err(SynaptixError::Unsupported("dec_attn_prep: кэш [1, nkv, max_seq, hd]"));
        }
        if k_cache.dtype() != DType::BF16 || v_cache.dtype() != DType::BF16 {
            return Err(SynaptixError::Unsupported("dec_attn_prep: кэш BF16"));
        }
        if cos.dtype() != DType::BF16 || sin.dtype() != DType::BF16 || pos.dtype() != DType::U32 || kv_pos.dtype() != DType::U32 {
            return Err(SynaptixError::Unsupported("dec_attn_prep: cos/sin BF16, позиции U32"));
        }
        let max_seq = kd[2];
        let dev = qt.device();
        let backend = registry::backend_for(dev)?;
        let mut q_out = alloc(backend, dev, DType::BF16.bytes_for_numel(nh * hd))?;
        let esz = 2usize;
        {
            let stream = Stream::default_for(dev)?;
            let kc_layout = k_cache.layout.clone();
            let vc_layout = v_cache.layout.clone();
            if kc_layout.byte_offset() != 0 || vc_layout.byte_offset() != 0 {
                return Err(SynaptixError::NonContiguous);
            }
            let kc = Arc::get_mut(&mut k_cache.storage)
                .ok_or_else(|| SynaptixError::Other("dec_attn_prep: K-кэш aliased".into()))?;
            let vc = Arc::get_mut(&mut v_cache.storage)
                .ok_or_else(|| SynaptixError::Other("dec_attn_prep: V-кэш aliased".into()))?;
            backend.dec_attn_prep(
                (&qt.storage, qt.layout.byte_offset() + qo * esz),
                (&kt.storage, kt.layout.byte_offset() + ko * esz),
                v.map(|(t, o)| (&*t.storage, t.layout.byte_offset() + o * esz)),
                q_norm.map(|t| &*t.storage),
                k_norm.map(|t| &*t.storage),
                v_norm,
                &cos.storage,
                &sin.storage,
                &pos.storage,
                rotary_dim,
                &kv_pos.storage,
                &mut q_out,
                kc,
                vc,
                max_seq,
                nh,
                nkv,
                hd,
                eps,
                &stream,
            )?;
        }
        Ok(Tensor::from_parts(
            Arc::new(q_out),
            Layout::contiguous(Shape::new(vec![1usize, nh, 1, hd]), DType::BF16),
        ))
    }

    /// Групповой GEMV от общей квант-пары: выход `[1, ΣN]` в `out_dtype`,
    /// блоки групп подряд. `router = Some((w F32[e, K], x BF16[K]))` — логиты
    /// роутера MoE тем же запуском (второй элемент результата, F32[e]).
    pub fn dec_gemv_grouped(
        groups: &[&QuantWeight],
        x_packed: &Tensor,
        x_scales: &Tensor,
        out_dtype: DType,
        router: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let first = groups.first().ok_or(SynaptixError::Unsupported("dec_gemv_grouped: пустая группа"))?;
        let dev = first.device();
        let esz = out_dtype.size_in_bits() / 8;
        let mut offs = Vec::with_capacity(groups.len());
        let mut total = 0usize;
        for g in groups {
            offs.push(total * esz);
            total += g.n();
        }
        let backend = registry::backend_for(dev)?;
        let mut out = alloc(backend, dev, total * esz)?;
        let specs: Vec<(&QuantWeight, usize)> = groups.iter().copied().zip(offs).collect();
        let stream = Stream::default_for(dev)?;
        let mut logits = match router {
            Some((w, x)) => {
                if w.rank() != 2 || w.dims()[1] != first.k() || w.dtype() != DType::F32 || !w.is_contiguous() {
                    return Err(SynaptixError::Unsupported("dec_gemv_grouped: роутер w — F32 [e, K]"));
                }
                if x.dtype() != DType::BF16 || x.numel() != first.k() || !x.is_contiguous() || x.layout.byte_offset() != 0 {
                    return Err(SynaptixError::Unsupported("dec_gemv_grouped: вход роутера — BF16 [K]"));
                }
                Some((alloc(backend, dev, w.dims()[0] * 4)?, w.dims()[0]))
            }
            None => None,
        };
        let router_arg = match (router, &mut logits) {
            (Some((w, x)), Some((lg, e))) => Some((&*w.storage, &*x.storage, lg, *e)),
            _ => None,
        };
        backend.dec_gemv_grouped(&specs, &x_packed.storage, &x_scales.storage, &mut out, out_dtype, router_arg, &stream)?;
        let logits = logits.map(|(lg, e)| {
            Tensor::from_parts(Arc::new(lg), Layout::contiguous(Shape::new(vec![e]), DType::F32))
        });
        Ok((
            Tensor::from_parts(
                Arc::new(out),
                Layout::contiguous(Shape::new(vec![1usize, total]), out_dtype),
            ),
            logits,
        ))
    }
}

impl ExpertTable {
    /// Индексный GEMV без скретчей указателей: строки `[pairs, N]` в `dtype`.
    pub fn gemv_indexed_rows(
        &self,
        idx: &Tensor,
        packed: &Tensor,
        scales: &Tensor,
        rows_per_pair: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let pairs = idx.numel();
        let dev = self.device();
        let backend = registry::backend_for(dev)?;
        let mut out = alloc(backend, dev, dtype.bytes_for_numel(pairs * self.n()))?;
        let stream = Stream::default_for(dev)?;
        let idx_c = if idx.is_contiguous() { idx.clone() } else { idx.contiguous()? };
        backend.dec_gemv_indexed(
            self.w_table_storage(),
            self.s_table_storage(),
            &idx_c.storage,
            &packed.storage,
            &scales.storage,
            rows_per_pair,
            DecIndexedOut::Rows { out: &mut out, dtype },
            self.n(),
            self.k(),
            self.count(),
            pairs,
            &stream,
        )?;
        Ok(Tensor::from_parts(
            Arc::new(out),
            Layout::contiguous(Shape::new(vec![pairs, self.n()]), dtype),
        ))
    }

    /// Индексный GEMV со взвешенной f32-суммой в `acc[N]` (`acc` уже обнулён
    /// роутером): `acc += wts[p] · W[idx[p]] · x_p`.
    pub fn gemv_indexed_accumulate(
        &self,
        idx: &Tensor,
        packed: &Tensor,
        scales: &Tensor,
        rows_per_pair: bool,
        wts: &Tensor,
        acc: &mut Tensor,
    ) -> Result<()> {
        let pairs = idx.numel();
        if wts.numel() != pairs || wts.dtype() != DType::F32 {
            return Err(SynaptixError::Unsupported("gemv_indexed_accumulate: wts — F32[pairs]"));
        }
        if acc.dtype() != DType::F32 || acc.numel() != self.n() {
            return Err(SynaptixError::Unsupported("gemv_indexed_accumulate: acc — F32[N]"));
        }
        let dev = self.device();
        let backend = registry::backend_for(dev)?;
        let stream = Stream::default_for(dev)?;
        let idx_c = if idx.is_contiguous() { idx.clone() } else { idx.contiguous()? };
        let acc_st = Arc::get_mut(&mut acc.storage)
            .ok_or_else(|| SynaptixError::Other("gemv_indexed_accumulate: acc aliased".into()))?;
        backend.dec_gemv_indexed(
            self.w_table_storage(),
            self.s_table_storage(),
            &idx_c.storage,
            &packed.storage,
            &scales.storage,
            rows_per_pair,
            DecIndexedOut::Accumulate { acc: acc_st, wts: &wts.storage },
            self.n(),
            self.k(),
            self.count(),
            pairs,
            &stream,
        )
    }
}

/// Partials на голову у [`Tensor::dec_flash_gqa`] (как в ядре: внутри блока
/// ключи делятся на `4 / WPH` частей).
pub fn dec_flash_gqa_partials(nrep: usize, splits: usize) -> usize {
    let hpw = nrep.div_ceil(4);
    let wph = nrep.div_ceil(hpw);
    splits * (4 / wph)
}

impl Tensor {
    /// Годится ли `dec_flash_gqa` для такой геометрии на этом устройстве.
    pub fn dec_flash_gqa_supported(dev: crate::device::Device, hd: usize, nrep: usize) -> bool {
        registry::backend_for(dev).map(|b| b.dec_flash_gqa_supported(hd, nrep)).unwrap_or(false)
    }

    /// Flash-decode одного запроса с GQA. `self` — q `[1, nh, 1, hd]` bf16,
    /// `k`/`v` — кэши `[1, nkv, cap, hd]`, `tkv` — U32[1], `window` —
    /// `Some(w)` для sliding-слоя. Возвращает `[1, nh, 1, hd]`.
    #[allow(clippy::type_complexity)]
    pub fn dec_flash_gqa(
        &self,
        k: &Tensor,
        v: &Tensor,
        tkv: &Tensor,
        scale: f32,
        window: Option<usize>,
        splits: usize,
        want_bf16: bool,
        want_mxfp8: bool,
    ) -> Result<(Option<Tensor>, Option<(Tensor, Tensor)>)> {
        if self.rank() != 4 || self.dims()[0] != 1 || self.dims()[2] != 1 || self.dtype() != DType::BF16 {
            return Err(SynaptixError::Unsupported("dec_flash_gqa: q [1, nh, 1, hd] bf16"));
        }
        let (nh, hd) = (self.dims()[1], self.dims()[3]);
        let kd = k.dims();
        if kd.len() != 4 || kd[0] != 1 || kd[3] != hd || v.dims() != kd || k.dtype() != DType::BF16 {
            return Err(SynaptixError::Unsupported("dec_flash_gqa: кэш [1, nkv, cap, hd] bf16"));
        }
        let (nkv, cap) = (kd[1], kd[2]);
        if nkv == 0 || nh % nkv != 0 {
            return Err(SynaptixError::Unsupported("dec_flash_gqa: nh кратно nkv"));
        }
        if !self.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let dev = self.device();
        let backend = registry::backend_for(dev)?;
        let s = dec_flash_gqa_partials(nh / nkv, splits);
        let mut pm = alloc(backend, dev, nh * s * 4)?;
        let mut pl = alloc(backend, dev, nh * s * 4)?;
        let mut pa = alloc(backend, dev, nh * s * hd * 4)?;
        if !want_bf16 && !want_mxfp8 {
            return Err(SynaptixError::Unsupported("dec_flash_gqa: нужен хотя бы один выход"));
        }
        if (nh * hd) % 32 != 0 {
            return Err(SynaptixError::Unsupported("dec_flash_gqa: nh·hd кратно 32"));
        }
        let mut out = if want_bf16 { Some(alloc(backend, dev, DType::BF16.bytes_for_numel(nh * hd))?) } else { None };
        let mut mx = if want_mxfp8 { Some((alloc(backend, dev, nh * hd)?, alloc(backend, dev, nh * hd / 32)?)) } else { None };
        let stream = Stream::default_for(dev)?;
        backend.dec_flash_gqa(
            &self.storage,
            &k.storage,
            &v.storage,
            &tkv.storage,
            scale,
            window.unwrap_or(0),
            cap,
            nh,
            nkv,
            hd,
            splits,
            (&mut pm, &mut pl, &mut pa),
            out.as_mut(),
            mx.as_mut().map(|(p, sc)| (p, sc)),
            &stream,
        )?;
        let out_t = out.map(|o| {
            Tensor::from_parts(Arc::new(o), Layout::contiguous(Shape::new(vec![1usize, nh, 1, hd]), DType::BF16))
        });
        let mx_t = mx.map(|(p, sc)| {
            (
                Tensor::from_parts(Arc::new(p), u8_layout(nh * hd)),
                Tensor::from_parts(Arc::new(sc), u8_layout(nh * hd / 32)),
            )
        });
        Ok((out_t, mx_t))
    }
}
