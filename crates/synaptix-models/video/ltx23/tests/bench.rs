//! Бенчмарк генерации видео (VideoDit, distilled 8 шагов + VAE). Гейт SYN_LTX_BENCH.
//! Размер латента из SYN_LTX_GRID="F',H',W'".
//! FullHD-цель: stage1 16×17×30 → upscaler → 16×34×60 (Tv=32640).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::VideoDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;
use synaptix_video_ltx23::pipeline::{generate_video, pixel_coords};
use synaptix_video_ltx23::vae::VaeDecoder;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const VENC: &str = "tests/reference_data/ltx_gemma/textcond_video_s128.safetensors";

#[test]
fn bench_video_generation() {
    if std::env::var("SYN_LTX_BENCH").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(VENC).exists() {
        eprintln!("skip bench: weights absent");
        return;
    }
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(None);
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);

    let venc = SafetensorsLoader::open(VENC).unwrap().with_device(dev).load("video_encoding").unwrap();
    let (tv, dv) = (venc.dims()[0], venc.dims()[1]);
    let venc = venc.reshape(vec![1, tv, dv]).unwrap();

    let grid = std::env::var("SYN_LTX_GRID").unwrap_or_else(|_| "8,17,30".into());
    let g: Vec<usize> = grid.split(',').map(|s| s.trim().parse().unwrap()).collect();
    let (fp, hp, wp) = (g[0], g[1], g[2]);
    let toks = fp * hp * wp;

    let t_load = std::time::Instant::now();
    let ckpt = LtxCheckpoint::open(CKPT, Device::Cpu, DType::BF16).unwrap();
    // SYN_LTX_MXFP8=1 → веса DiT mxfp8 РЕЗИДЕНТНО (22GB влезает в 24GB) → нет offload-стрима.
    let mxfp8 = std::env::var("SYN_LTX_MXFP8").as_deref() == Ok("1");
    let (qdt, offload) = if mxfp8 { (DType::MXFP8, false) } else { (DType::BF16, true) };
    // mxfp8-резидент: квант на GPU → ckpt.view_on(dev) (веса mmap→GPU перед quantize_to_mxfp8).
    let dit = if mxfp8 {
        VideoDit::load_with(&ckpt.view_on(dev), dev, DType::BF16, qdt, offload).expect("dit")
    } else {
        VideoDit::load_with(&ckpt, dev, DType::BF16, qdt, offload).expect("dit")
    };
    let vae = VaeDecoder::load(&ckpt, dev).expect("vae");
    eprintln!("[BENCH] load: {:.1}s | grid {fp}×{hp}×{wp} = {toks} токенов", t_load.elapsed().as_secs_f32());

    // Сохранить кадр для визуальной оценки fused-rms vs decomposed (детерм. seed).
    if std::env::var("SYN_LTX_SAVE_FRAMES").is_ok() {
        let save = |rgb: &Tensor, name: &str| {
            let frames = synaptix_video_ltx23::pipeline::rgb_to_frames(rgb).unwrap();
            let fr = &frames[frames.len() / 2];
            let (h, w) = (fr.dims()[1], fr.dims()[2]);
            let planar: Vec<f32> = fr.reshape(vec![3 * h * w]).unwrap().to_vec1::<f32>().unwrap();
            let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
            for y in 0..h { for x in 0..w { for c in 0..3 {
                buf.push((planar[c * h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8);
            }}}
            std::fs::write(format!("/tmp/ltx_frame_{name}.ppm"), buf).unwrap();
            eprintln!("WROTE /tmp/ltx_frame_{name}.ppm ({w}x{h})");
        };
        let rd = synaptix_core::grad::no_grad(|| generate_video(&dit, &vae, &venc, fp, hp, wp, 24.0, dev)).unwrap();
        save(&rd, "decomposed");
        let rf = synaptix_core::grad::no_grad(|| generate_video(&dit, &vae, &venc, fp, hp, wp, 24.0, dev)).unwrap();
        save(&rf, "fused");
        return;
    }

    // Recon компонентов attn1: flash vs rms-decomposed vs transpose на LTX-форме.
    if std::env::var("SYN_LTX_RECON_ATTN").is_ok() {
        let h = 32usize;
        let dh = 128usize;
        let t = toks;
        let mk = |sh: Vec<usize>| Tensor::randn(sh, Device::Cpu).unwrap().to_device(dev).unwrap().to_dtype(DType::BF16).unwrap();
        let sync = || { let _ = synaptix_core::device::cuda::synchronize(0); };
        let timeit = |label: &str, iters: usize, f: &dyn Fn()| {
            for _ in 0..5 { f(); }
            sync();
            let t0 = std::time::Instant::now();
            for _ in 0..iters { f(); }
            sync();
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            eprintln!("[RECON attn] {label}: {ms:.3} ms");
        };
        // flash [1,h,t,dh]
        let q = mk(vec![1, h, t, dh]);
        let k = mk(vec![1, h, t, dh]);
        let v = mk(vec![1, h, t, dh]);
        let scale = 1.0f32 / (dh as f32).sqrt();
        timeit("flash[1,32,T,128]", 20, &|| {
            let _ = std::hint::black_box(q.flash_attention(&k, &v, scale, false).unwrap());
        });
        // decomposed rms на [1,t,4096]
        let x = mk(vec![1, t, 4096]);
        timeit("rms_decomposed[1,T,4096]", 20, &|| {
            let xf = x.to_dtype(DType::F32).unwrap();
            let d = xf.sqr().unwrap().mean_keepdim(2).unwrap().add_scalar(1e-6).unwrap().sqrt().unwrap();
            let _ = std::hint::black_box(xf.broadcast_div(&d).unwrap().to_dtype(DType::BF16).unwrap().contiguous().unwrap());
        });
        // transpose+contiguous [1,t,32,128]->[1,32,t,128]
        let xt = mk(vec![1, t, h, dh]);
        timeit("transpose+contig", 20, &|| {
            let _ = std::hint::black_box(xt.transpose(1, 2).unwrap().contiguous().unwrap());
        });
        return;
    }

    // Реальный гейт качества: noise детерминирован (seed 0xDEADBEEF) → сравниваем
    // ПОЛНОЕ видео (вся траектория денойза) fused-rms vs decomposed пиксельно.
    if std::env::var("SYN_LTX_CHECK_VID").is_ok() {
        // offload bit-identical? повторный прогон prefetch+pinned пути.
        let rd = synaptix_core::grad::no_grad(|| generate_video(&dit, &vae, &venc, fp, hp, wp, 24.0, dev)).unwrap().to_dtype(DType::F32).unwrap();
        let rf = synaptix_core::grad::no_grad(|| generate_video(&dit, &vae, &venc, fp, hp, wp, 24.0, dev)).unwrap().to_dtype(DType::F32).unwrap();
        let diff = rd.sub(&rf).unwrap().abs().unwrap();
        let worst = diff.max_all().unwrap().to_scalar::<f32>().unwrap();
        let scale = rd.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        eprintln!("[CHECK vid] RGB run1 vs run2: max|Δ|={worst:.5} scale={scale:.3} rel={:.2e}", worst / scale.max(1e-9));
        return;
    }
    // Изоляция ядра rms_norm_fused vs decomposed F32 на одном тензоре.
    if std::env::var("SYN_LTX_CHECK_RMS1").is_ok() {
        let x = Tensor::randn(vec![toks, 4096], Device::Cpu).unwrap().to_device(dev).unwrap().to_dtype(DType::BF16).unwrap();
        let ones = Tensor::ones(vec![4096], DType::BF16, dev).unwrap();
        let yf = x.rms_norm_fused(&ones, 1e-6, false).unwrap().to_dtype(DType::F32).unwrap();
        let xf = x.to_dtype(DType::F32).unwrap();
        let denom = xf.sqr().unwrap().mean_keepdim(1).unwrap().add_scalar(1e-6).unwrap().sqrt().unwrap();
        let yd = xf.broadcast_div(&denom).unwrap();
        let diff = yd.sub(&yf).unwrap().abs().unwrap();
        let worst = diff.max([1usize]).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let scale = yd.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        eprintln!("[CHECK rms1] kernel vs decomposed: max|Δ|={worst:.6} scale={scale:.3} rel={:.2e}", worst / scale.max(1e-9));
        return;
    }
    // Детерминизм forward на фикс-входе: per-row max|Δ| двух прогонов.
    if std::env::var("SYN_LTX_CHECK_RMS").is_ok() {
        let positions = pixel_coords(fp, hp, wp, 24.0);
        let ctx = venc.to_dtype(DType::BF16).unwrap();
        let latent = Tensor::randn(vec![1, toks, 128], Device::Cpu).unwrap().to_device(dev).unwrap().to_dtype(DType::BF16).unwrap();
        let ts: Vec<f32> = vec![0.7; toks];
        let yd = synaptix_core::grad::no_grad(|| dit.forward(&latent, &ts, 0.7, &positions, &ctx)).unwrap().to_dtype(DType::F32).unwrap();
        let yf = synaptix_core::grad::no_grad(|| dit.forward(&latent, &ts, 0.7, &positions, &ctx)).unwrap().to_dtype(DType::F32).unwrap();
        let diff = yd.sub(&yf).unwrap().abs().unwrap();
        let last = diff.rank() - 1;
        let worst = diff.max([last]).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let scale = yd.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        eprintln!("[CHECK rms] per-row max|Δ|={worst:.6} scale={scale:.3} rel={:.2e}", worst / scale.max(1e-9));
        return;
    }

    let t0 = std::time::Instant::now();
    let rgb = synaptix_core::grad::no_grad(|| generate_video(&dit, &vae, &venc, fp, hp, wp, 24.0, dev)).expect("gen");
    let _ = synaptix_core::device::cuda::synchronize(0);
    let secs = t0.elapsed().as_secs_f32();
    let (f, h, w) = (rgb.dims()[2], rgb.dims()[3], rgb.dims()[4]);
    eprintln!("[BENCH] ИТОГО: {secs:.1}s — выход {f} кадров {h}×{w} ({toks} ток, 8 шагов)");
}
