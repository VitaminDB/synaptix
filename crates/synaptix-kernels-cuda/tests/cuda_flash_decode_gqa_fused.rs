//! `dec_flash_gqa` (split + merge) против прежних flash-ядер декода:
//! плотный кэш bf16/f16 — против `flash_attention_dev`, кэш MXFP8 — против
//! `flash_attention_mxfp8kv_dev` на тех же квантованных байтах и против
//! плотного f16-эталона; выходной гейт — против явного `· sigmoid(gate)`.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn det_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            (u as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn flat(t: &Tensor) -> Vec<f32> {
    let n: usize = t.dims().iter().product();
    t.to_dtype(DType::F32).unwrap().reshape(vec![n]).unwrap().to_vec1::<f32>().unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn amax(a: &[f32]) -> f32 {
    a.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

#[test]
fn flash_decode_gqa_matches_reference_kernels() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    let dev = Device::Cuda(0);
    let (nh, nkv, hd, cap, tkv) = (16usize, 2usize, 256usize, 1024usize, 777usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let tkv_t = Tensor::from_vec(vec![tkv as u32], vec![1], dev).unwrap();
    let q_f = det_f32(1, nh * hd, 1.0);
    let mut k_f = det_f32(2, nkv * cap * hd, 1.0);
    let mut v_f = det_f32(3, nkv * cap * hd, 1.0);
    for h in 0..nkv {
        for t in tkv..cap {
            for d in 0..hd {
                k_f[(h * cap + t) * hd + d] = 0.0;
                v_f[(h * cap + t) * hd + d] = 0.0;
            }
        }
    }
    let gate_f = det_f32(4, nh * 2 * hd, 3.0);

    for dt in [DType::BF16, DType::F16] {
        let q = Tensor::from_vec(q_f.clone(), vec![1, nh, 1, hd], dev).unwrap().to_dtype(dt).unwrap();
        let k = Tensor::from_vec(k_f.clone(), vec![1, nkv, cap, hd], dev).unwrap().to_dtype(dt).unwrap();
        let v = Tensor::from_vec(v_f.clone(), vec![1, nkv, cap, hd], dev).unwrap().to_dtype(dt).unwrap();
        let reference = flat(&q.flash_attention_dev(&k, &v, &tkv_t, scale, true).unwrap());
        let (row, _) = q.dec_flash_gqa(&k, &v, None, &tkv_t, scale, None, 32, None, true, false).unwrap();
        let got = flat(&row.unwrap());
        let d = max_diff(&reference, &got);
        eprintln!("{dt:?} плотный: |ref| ≤ {:.4}, Δ {d:.3e}", amax(&reference));
        assert!(d < 4e-3 * amax(&reference).max(0.1), "{dt:?}: Δ {d}");

        // Выходной гейт: y · sigmoid(gate[h, hd + d]) при шаге 2·hd.
        let gate = Tensor::from_vec(gate_f.clone(), vec![1, nh * 2 * hd], dev).unwrap().to_dtype(dt).unwrap();
        let (row_g, _) = q
            .dec_flash_gqa(&k, &v, None, &tkv_t, scale, None, 32, Some((&gate, hd, 2 * hd)), true, false)
            .unwrap();
        let got_g = flat(&row_g.unwrap());
        let gate_r = flat(&gate);
        let expect: Vec<f32> = (0..nh * hd)
            .map(|i| {
                let (h, d) = (i / hd, i % hd);
                let g = gate_r[h * 2 * hd + hd + d];
                reference[i] / (1.0 + (-g).exp())
            })
            .collect();
        // Эталон округляет дважды (строка в T, затем произведение) — допуск в ulp T.
        let dg = max_diff(&expect, &got_g);
        let ulp = if dt == DType::BF16 { 1e-2 } else { 2e-3 };
        eprintln!("{dt:?} гейт: |ожид.| ≤ {:.4}, Δ {dg:.3e}", amax(&expect));
        assert!(dg < ulp * amax(&expect).max(0.1), "{dt:?} гейт: Δ {dg}");

        if dt != DType::F16 {
            continue;
        }
        // MXFP8-кэш: квант прежним append-ядром, затем новое ядро против прежнего
        // MXFP8-flash и плотного f16-эталона.
        let mut kq = Tensor::zeros(vec![1, nkv, cap, hd], DType::MXFP8, dev).unwrap();
        let mut vq = Tensor::zeros(vec![1, nkv, cap, hd], DType::MXFP8, dev).unwrap();
        let mut ks = Tensor::zeros(vec![1, nkv, cap, hd / 32], DType::U8, dev).unwrap();
        let mut vs = Tensor::zeros(vec![1, nkv, cap, hd / 32], DType::U8, dev).unwrap();
        let k_rows = k.narrow(2, 0, tkv).unwrap().contiguous().unwrap();
        let v_rows = v.narrow(2, 0, tkv).unwrap().contiguous().unwrap();
        kq.kv_append_quant_mxfp8_inplace(&mut ks, &k_rows, 0).unwrap();
        vq.kv_append_quant_mxfp8_inplace(&mut vs, &v_rows, 0).unwrap();
        let old_mx = flat(&q.flash_attention_mxfp8kv_dev(&kq, &vq, &ks, &vs, &tkv_t, scale, true).unwrap());
        let (row_mx, _) = q
            .dec_flash_gqa(&kq, &vq, Some((&ks, &vs)), &tkv_t, scale, None, 32, None, true, false)
            .unwrap();
        let got_mx = flat(&row_mx.unwrap());
        let d_old = max_diff(&old_mx, &got_mx);
        let d_dense = max_diff(&reference, &got_mx);
        let d_old_dense = max_diff(&reference, &old_mx);
        eprintln!("MXFP8: Δ(новое, прежнее MXFP8) {d_old:.3e}, Δ(новое, плотное) {d_dense:.3e}, Δ(прежнее, плотное) {d_old_dense:.3e}");
        assert!(d_old < 1e-2 * amax(&reference).max(0.1), "MXFP8 против прежнего ядра: Δ {d_old}");
        assert!(d_dense < 3.0 * d_old_dense.max(1e-3), "MXFP8 против плотного хуже прежнего ядра: {d_dense} vs {d_old_dense}");
    }
}

/// Квант-запись K/V в MXFP8 из `dec_attn_prep` (без норм и RoPE) совпадает
/// байт в байт с `kv_append_quant_mxfp8_inplace`; q-строка с шагом 2·hd
/// (выходной гейт) берётся из первой половины головы.
#[test]
fn attn_prep_mxfp8_kv_matches_append_kernel() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    let dev = Device::Cuda(0);
    let (nh, nkv, hd, cap, pos) = (16usize, 2usize, 256usize, 64usize, 37usize);
    let q_stride = 2 * hd;
    let qkv_len = nh * q_stride + 2 * nkv * hd;
    let qkv_f = det_f32(11, qkv_len, 2.0);
    let qkv = Tensor::from_vec(qkv_f.clone(), vec![1, qkv_len], dev).unwrap().to_dtype(DType::F16).unwrap();
    let k_off = nh * q_stride;
    let v_off = k_off + nkv * hd;
    let mut kq = Tensor::zeros(vec![1, nkv, cap, hd], DType::MXFP8, dev).unwrap();
    let mut vq = Tensor::zeros(vec![1, nkv, cap, hd], DType::MXFP8, dev).unwrap();
    let mut ks = Tensor::zeros(vec![1, nkv, cap, hd / 32], DType::U8, dev).unwrap();
    let mut vs = Tensor::zeros(vec![1, nkv, cap, hd / 32], DType::U8, dev).unwrap();
    let cos = Tensor::zeros(vec![cap, 2], DType::F16, dev).unwrap();
    let pos_t = Tensor::from_vec(vec![pos as u32], vec![1], dev).unwrap();
    let q = Tensor::dec_attn_prep(
        (&qkv, 0), (&qkv, k_off), Some((&qkv, v_off)), None, None, false, &cos, &cos, &pos_t, 0, &pos_t,
        &mut kq, &mut vq, Some((&mut ks, &mut vs)), nh, nkv, hd, q_stride, 1e-6,
    )
    .unwrap();
    let q_got = flat(&q);
    for h in 0..nh {
        for d in 0..hd {
            let want = half::f16::from_f32(qkv_f[h * q_stride + d]).to_f32();
            assert_eq!(q_got[h * hd + d], want, "q[{h}, {d}]");
        }
    }
    // Эталон: прежнее append-ядро на тех же строках K/V.
    let k_rows = qkv.narrow(1, k_off, nkv * hd).unwrap().contiguous().unwrap().reshape(vec![1, nkv, 1, hd]).unwrap();
    let v_rows = qkv.narrow(1, v_off, nkv * hd).unwrap().contiguous().unwrap().reshape(vec![1, nkv, 1, hd]).unwrap();
    let mut kq2 = Tensor::zeros(vec![1, nkv, cap, hd], DType::MXFP8, dev).unwrap();
    let mut vq2 = Tensor::zeros(vec![1, nkv, cap, hd], DType::MXFP8, dev).unwrap();
    let mut ks2 = Tensor::zeros(vec![1, nkv, cap, hd / 32], DType::U8, dev).unwrap();
    let mut vs2 = Tensor::zeros(vec![1, nkv, cap, hd / 32], DType::U8, dev).unwrap();
    kq2.kv_append_quant_mxfp8_inplace(&mut ks2, &k_rows, pos).unwrap();
    vq2.kv_append_quant_mxfp8_inplace(&mut vs2, &v_rows, pos).unwrap();
    let bytes = |t: &Tensor| -> Vec<u8> { t.to_device(Device::Cpu).unwrap().to_bytes().unwrap() };
    assert_eq!(bytes(&ks), bytes(&ks2), "масштабы K");
    assert_eq!(bytes(&vs), bytes(&vs2), "масштабы V");
    let (kb, kb2) = (bytes(&kq), bytes(&kq2));
    let (vb, vb2) = (bytes(&vq), bytes(&vq2));
    assert_eq!(kb.len(), nkv * cap * hd);
    assert_eq!(kb, kb2, "байты K");
    assert_eq!(vb, vb2, "байты V");
    eprintln!("MXFP8-запись K/V из attn_prep: байты и масштабы совпали ({} байт)", kb.len());
}
