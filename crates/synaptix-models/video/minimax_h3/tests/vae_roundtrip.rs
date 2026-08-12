use std::path::PathBuf;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_video_minimax_h3::config::VaeConfig;
use synaptix_video_minimax_h3::loader::{ComponentLoader, H3Paths};
use synaptix_video_minimax_h3::vae::{VaeDecoder, VaeEncoder};

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var("H3_MODEL_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("models/MiniMax-H3")
    });
    (p.join("FL2VA").is_dir() || p.join("transformer").is_dir()).then_some(p)
}

fn device() -> Option<Device> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let d = Device::Cuda(0);
    Tensor::zeros(vec![1], DType::F32, d).ok().map(|_| d)
}

fn pattern(frames: usize, h: usize, w: usize) -> Vec<f32> {
    let mut v = vec![0f32; 3 * frames * h * w];
    for f in 0..frames {
        for y in 0..h {
            for x in 0..w {
                let fx = x as f32 / w as f32;
                let fy = y as f32 / h as f32;
                let t = f as f32 / frames as f32;
                let disc = (((fx - 0.45 - 0.1 * t).powi(2) + (fy - 0.5).powi(2)).sqrt() * 6.0)
                    .cos();
                let base = f * h * w + y * w + x;
                v[base] = (0.8 * disc + 0.2 * (fy * 2.0 - 1.0)).clamp(-1.0, 1.0);
                v[frames * h * w + base] = (fx * 1.6 - 0.8).clamp(-1.0, 1.0);
                v[2 * frames * h * w + base] =
                    (0.6 * (fx + fy - 1.0) + 0.4 * disc).clamp(-1.0, 1.0);
            }
        }
    }
    v
}

fn blockiness(v: &[f32], frames: usize, h: usize, w: usize, block: usize) -> f32 {
    let (mut edge, mut inner, mut ne, mut ni) = (0.0f64, 0.0f64, 0usize, 0usize);
    for f in 0..frames {
        for y in 0..h {
            for x in 1..w {
                let a = v[f * h * w + y * w + x] as f64;
                let b = v[f * h * w + y * w + x - 1] as f64;
                let d = (a - b).abs();
                if x % block == 0 {
                    edge += d;
                    ne += 1;
                } else {
                    inner += d;
                    ni += 1;
                }
            }
        }
    }
    ((edge / ne.max(1) as f64) / (inner / ni.max(1) as f64).max(1e-9)) as f32
}

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().map(|x| *x as f64).sum::<f64>() / n, b.iter().map(|x| *x as f64).sum::<f64>() / n);
    let mut num = 0.0;
    let (mut da, mut db) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (u, v) = (*x as f64 - ma, *y as f64 - mb);
        num += u * v;
        da += u * u;
        db += v * v;
    }
    (num / (da.sqrt() * db.sqrt()).max(1e-12)) as f32
}

fn stats(name: &str, v: &[f32]) -> String {
    let n = v.len() as f64;
    let m = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let s = (v.iter().map(|x| (*x as f64 - m).powi(2)).sum::<f64>() / n).sqrt();
    let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |(l, h), x| (l.min(*x), h.max(*x)));
    format!("{name} μ {m:+.4} σ {s:.4} [{lo:+.3}, {hi:+.3}]")
}

#[test]
#[ignore]
fn video_vae_roundtrip_preserves_structure() {
    let Some(dev) = device() else { return };
    let Some(dir) = model_dir() else { return };
    let paths = H3Paths::open(&dir).expect("paths");
    let cfg = VaeConfig::from_dir(&paths.root).expect("config");

    let (frames, h, w) = (17usize, 256usize, 384usize);
    let src = pattern(frames, h, w);
    let x = Tensor::from_vec(src.clone(), vec![1, 3, frames, h, w], dev).expect("input");
    eprintln!("{}", stats("[vae] вход", &src));

    let w_enc = ComponentLoader::open_file(paths.video_vae_file(), dev).expect("веса");
    let t0 = Instant::now();
    let enc = VaeEncoder::load(&w_enc, cfg.clone(), dev, DType::BF16).expect("энкодер");
    let z = enc.encode(&x).expect("encode");
    eprintln!("[vae] латент {:?} за {:.1} с", z.dims(), t0.elapsed().as_secs_f32());
    let zv = z.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    eprintln!("{}", stats("[vae] латент", &zv));
    {
        let d = z.dims().to_vec();
        let (c, lt, lh, lw) = (d[1], d[2], d[3], d[4]);
        let at = |ci: usize, ti: usize, y: usize, x: usize| {
            zv[((ci * lt + ti) * lh + y) * lw + x] as f64
        };
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for ci in 0..c {
            for ti in 0..lt {
                for y in 0..lh {
                    for x in 1..lw {
                        pairs.push((at(ci, ti, y, x - 1), at(ci, ti, y, x)));
                    }
                }
            }
        }
        let n = pairs.len() as f64;
        let ma = pairs.iter().map(|p| p.0).sum::<f64>() / n;
        let mb = pairs.iter().map(|p| p.1).sum::<f64>() / n;
        let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
        for (a, b) in &pairs {
            let (u, v) = (a - ma, b - mb);
            num += u * v;
            da += u * u;
            db += v * v;
        }
        eprintln!("[vae] корреляция соседей латента по w {:.3}", num / (da.sqrt() * db.sqrt()));
    }
    drop(enc);

    let dec = VaeDecoder::load(&w_enc, cfg, dev, DType::BF16).expect("декодер");
    let t1 = Instant::now();
    let y = dec.decode(&z).expect("decode");
    eprintln!("[vae] декод {:?} за {:.1} с", y.dims(), t1.elapsed().as_secs_f32());
    let out = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    eprintln!("{}", stats("[vae] выход", &out));

    let n = out.len().min(src.len());
    let c = corr(&src[..n], &out[..n]);
    let of = y.dims()[2];
    eprintln!("[vae] корреляция вход/выход {c:.4}");
    eprintln!(
        "[vae] блочность вход {:.3} выход {:.3} (1.0 = швов нет)",
        blockiness(&src, frames, h, w, 16),
        blockiness(&out, of, h, w, 16)
    );
    assert!(
        c > 0.8,
        "round-trip развалился: корреляция {c:.4}. Латент содержит структуру, но декодер её не восстанавливает"
    );
}

#[test]
#[ignore]
fn decoder_on_a_smooth_latent_has_no_seams() {
    let Some(dev) = device() else { return };
    let Some(dir) = model_dir() else { return };
    let paths = H3Paths::open(&dir).expect("paths");
    let cfg = VaeConfig::from_dir(&paths.root).expect("config");
    let w = ComponentLoader::open_file(paths.video_vae_file(), dev).expect("веса");
    let dec = VaeDecoder::load(&w, cfg, dev, DType::BF16).expect("декодер");

    let (c, t, lh, lw) = (24usize, 4usize, 16usize, 24usize);
    let mut z = vec![0f32; c * t * lh * lw];
    for ci in 0..c {
        for ti in 0..t {
            for y in 0..lh {
                for x in 0..lw {
                    let fx = x as f32 / lw as f32;
                    let fy = y as f32 / lh as f32;
                    let phase = ci as f32 * 0.3 + ti as f32 * 0.2;
                    z[((ci * t + ti) * lh + y) * lw + x] =
                        ((fx * 2.0 + phase).sin() + (fy * 2.0 + phase).cos()) * 0.6;
                }
            }
        }
    }
    let zt = Tensor::from_vec(z.clone(), vec![1, c, t, lh, lw], dev).unwrap();
    let y = dec.decode(&zt).expect("decode");
    synaptix_video_minimax_h3::pipeline::dump_tensor("seam_z", &zt);
    synaptix_video_minimax_h3::pipeline::dump_tensor("seam_rgb", &y);
    let out = y.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let frames = y.dims()[2];
    let (h, wpx) = (lh * 16, lw * 16);
    eprintln!("{}", stats("[seam] выход", &out));
    let b = blockiness(&out, frames, h, wpx, 16);
    eprintln!("[seam] блочность {b:.3} на гладком латенте");
    assert!(b < 1.5, "декодер оставляет швы 16x16: блочность {b:.3}");
}
