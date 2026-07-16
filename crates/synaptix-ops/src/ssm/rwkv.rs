use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn rwkv_time_mix(
    x: &Tensor,
    x_prev: &Tensor,
    time_mix_k: &Tensor,
    time_mix_v: &Tensor,
    time_mix_r: &Tensor,
) -> Result<Tensor> {
    let xk = x.mul(time_mix_k)?.add(&x_prev.mul(&time_mix_k.affine(-1.0, 1.0)?)?)?;
    let xv = x.mul(time_mix_v)?.add(&x_prev.mul(&time_mix_v.affine(-1.0, 1.0)?)?)?;
    let xr = x.mul(time_mix_r)?.add(&x_prev.mul(&time_mix_r.affine(-1.0, 1.0)?)?)?;
    Tensor::cat(&[&xk, &xv, &xr], x.rank() - 1)
}

pub fn rwkv_channel_mix(
    x: &Tensor,
    x_prev: &Tensor,
    time_mix_k: &Tensor,
    time_mix_r: &Tensor,
) -> Result<Tensor> {
    let xk = x.mul(time_mix_k)?.add(&x_prev.mul(&time_mix_k.affine(-1.0, 1.0)?)?)?;
    let xr = x.mul(time_mix_r)?.add(&x_prev.mul(&time_mix_r.affine(-1.0, 1.0)?)?)?;
    Tensor::cat(&[&xk, &xr], x.rank() - 1)
}

pub fn rwkv_wkv(
    k: &Tensor,
    v: &Tensor,
    r: &Tensor,
    time_decay: &Tensor,
    time_first: &Tensor,
) -> Result<Tensor> {
    if k.rank() != 3 || v.rank() != 3 || r.rank() != 3 {
        return Err(SynaptixError::Unsupported(
            "rwkv_wkv: requires rank-3 [B,L,D]",
        ));
    }
    let (b, l, d) = (k.dims()[0], k.dims()[1], k.dims()[2]);
    if time_decay.dims() != [d] || time_first.dims() != [d] {
        return Err(SynaptixError::shape_mismatch(&[d], time_decay.dims()));
    }

    let w_neg = time_decay.exp()?.affine(-1.0, 0.0)?;
    let u = time_first;

    let mut aa = Tensor::zeros(vec![b, d], k.dtype(), k.device())?;
    let mut bb = Tensor::zeros(vec![b, d], k.dtype(), k.device())?;
    let neg_large = -1.0e30_f32;
    let mut pp = Tensor::zeros(vec![b, d], k.dtype(), k.device())?.affine(0.0, neg_large)?;

    let mut ys: Vec<Tensor> = Vec::with_capacity(l);
    for t in 0..l {
        let kt = k.narrow(1, t, 1)?.squeeze(1)?;
        let vt = v.narrow(1, t, 1)?.squeeze(1)?;
        let rt = r.narrow(1, t, 1)?.squeeze(1)?;

        let ww_out = u.add(&kt)?;
        let p_out = pp.maximum(&ww_out)?;
        let e1_out = pp.sub(&p_out)?.exp()?;
        let e2_out = ww_out.sub(&p_out)?.exp()?;
        let num = e1_out.mul(&aa)?.add(&e2_out.mul(&vt)?)?;
        let den = e1_out.mul(&bb)?.add(&e2_out)?;
        let wkv = num.div(&den)?;
        ys.push(rt.sigmoid()?.mul(&wkv)?.unsqueeze(1)?);

        let ww_st = pp.add(&w_neg)?;
        let p_st = ww_st.maximum(&kt)?;
        let e1_st = ww_st.sub(&p_st)?.exp()?;
        let e2_st = kt.sub(&p_st)?.exp()?;
        aa = e1_st.mul(&aa)?.add(&e2_st.mul(&vt)?)?;
        bb = e1_st.mul(&bb)?.add(&e2_st)?;
        pp = p_st;
    }
    let refs: Vec<&Tensor> = ys.iter().collect();
    Tensor::cat(&refs, 1)
}
