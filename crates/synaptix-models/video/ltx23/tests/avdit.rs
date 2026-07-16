//! Фаза 8: joint A/V DiT — валидация velocity видео+аудио против LTX `LTXModel`
//! (AudioVideo). N-блок f32 CPU. Гейт: наличие весов + reference.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::AvDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const VREF: &str = "tests/reference_data/ltx_gemma/dit_video_ref.safetensors";
const AREF_DIR: &str = "tests/reference_data/ltx_gemma";

fn flat(t: &Tensor) -> Vec<f32> {
    let n: usize = t.dims().iter().product();
    t.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()
}
fn f64v(t: &Tensor) -> Vec<f64> {
    flat(t).into_iter().map(|x| x as f64).collect()
}
fn percol_cos(a: &[f32], b: &[f32], rows: usize, cols: usize) -> (f64, f64) {
    // cos по столбцам? нет — per-row (по токенам): cos каждой строки, берём минимум.
    let mut worst = 2.0f64;
    let mut global = {
        let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
        for i in 0..a.len() { d += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
        d / (na.sqrt() * nb.sqrt())
    };
    for r in 0..rows {
        let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
        for cc in 0..cols {
            let (x, y) = (a[r * cols + cc] as f64, b[r * cols + cc] as f64);
            d += x * y; na += x * x; nb += y * y;
        }
        let c = if na > 0.0 && nb > 0.0 { d / (na.sqrt() * nb.sqrt()) } else { 1.0 };
        worst = worst.min(c);
    }
    global = global.min(1.0);
    (global, worst)
}

#[test]
fn avdit_joint_matches_ltx() {
    let n = std::env::var("SYN_AVDIT_N").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let aref = format!("{AREF_DIR}/avdit_{n}blk.safetensors");
    if !Path::new(CKPT).exists() || !Path::new(&aref).exists() {
        eprintln!("skip avdit_joint: weights/ref absent");
        return;
    }
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(Some(n));
    synaptix_kernels_cpu::ensure_registered();
    // SYN_AVDIT_CUDA=1: прод-путь (CUDA bf16) — ловля расхождений ядер
    let cuda = std::env::var("SYN_AVDIT_CUDA").as_deref() == Ok("1");
    if cuda { synaptix_kernels_cuda::ensure_registered(); }
    let dev = if cuda { Device::Cuda(0) } else { Device::Cpu };
    let dt = if cuda { DType::BF16 } else { DType::F32 };

    let vl = SafetensorsLoader::open(VREF).unwrap().with_device(dev);
    let al = SafetensorsLoader::open(&aref).unwrap().with_device(dev);

    let v_lat = vl.load("latent").unwrap().reshape(vec![1, 64, 128]).unwrap().to_dtype(dt).unwrap();
    let v_ts = flat(&vl.load("timesteps").unwrap());
    let v_sigma = flat(&vl.load("sigma").unwrap())[0];
    let v_pos = f64v(&vl.load("positions").unwrap()); // [3,64,2] → flat
    let v_ctx = vl.load("context").unwrap();
    let (ttv, dv) = (v_ctx.dims()[0], v_ctx.dims()[1]);
    let v_ctx = v_ctx.reshape(vec![1, ttv, dv]).unwrap().to_dtype(dt).unwrap();

    let a_lat_t = al.load("a_latent").unwrap();
    let (ta, _) = (a_lat_t.dims()[0], a_lat_t.dims()[1]);
    let a_lat = a_lat_t.reshape(vec![1, ta, 128]).unwrap().to_dtype(dt).unwrap();
    let a_ts = flat(&al.load("a_timesteps").unwrap());
    let a_sigma = flat(&al.load("a_sigma").unwrap())[0];
    let a_pos = f64v(&al.load("a_positions").unwrap()); // [1,Ta,2]
    let a_ctx = al.load("a_context").unwrap();
    let (tta, da) = (a_ctx.dims()[0], a_ctx.dims()[1]);
    let a_ctx = a_ctx.reshape(vec![1, tta, da]).unwrap().to_dtype(dt).unwrap();

    let want_v = flat(&al.load("v_velocity_joint").unwrap());
    let want_a = flat(&al.load("a_velocity").unwrap());

    let ckpt = LtxCheckpoint::open(CKPT, dev, dt).unwrap();
    let dit = AvDit::load(&ckpt, dev, dt, dt).expect("avdit load");

    let (vv, av) = synaptix_core::grad::no_grad(|| {
        dit.forward(&v_lat, &v_ts, v_sigma, &v_pos, &v_ctx,
                    &a_lat, &a_ts, a_sigma, &a_pos, &a_ctx, None)
    }).expect("forward");
    eprintln!("vv {:?} av {:?}", vv.dims(), av.dims());

    let (vg, vw) = percol_cos(&flat(&vv), &want_v, 64, 128);
    let (ag, aw) = percol_cos(&flat(&av), &want_a, ta, 128);
    eprintln!("video velocity: cos={vg:.6} per-row-min={vw:.6}");
    eprintln!("audio velocity: cos={ag:.6} per-row-min={aw:.6}");
    if !cuda {
        assert!(vw > 0.999, "video per-row cos {vw}");
        assert!(aw > 0.999, "audio per-row cos {aw}");
    }
}
