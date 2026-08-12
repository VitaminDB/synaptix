use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_video_minimax_h3 as h3;

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var("H3_MODEL_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("models/MiniMax-H3")
    });
    (p.join("FL2VA").is_dir()).then_some(p)
}

fn dump_dir() -> Option<PathBuf> {
    std::env::var("H3_DUMP_DIR").ok().map(PathBuf::from).filter(|p| p.join("cond_hidden.f32").exists())
}

fn device() -> Option<Device> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let d = Device::Cuda(0);
    Tensor::zeros(vec![1], DType::F32, d).ok().map(|_| d)
}

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).expect("файл дампа");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|x| *x as f64).sum::<f64>() / n,
        b.iter().map(|x| *x as f64).sum::<f64>() / n,
    );
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (u, v) = (*x as f64 - ma, *y as f64 - mb);
        num += u * v;
        da += u * u;
        db += v * v;
    }
    (num / (da.sqrt() * db.sqrt()).max(1e-12)) as f32
}

fn sigma_of(v: &[f32]) -> f32 {
    let n = v.len() as f64;
    let m = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    ((v.iter().map(|x| (*x as f64 - m).powi(2)).sum::<f64>() / n).sqrt()) as f32
}

fn neigh_w(v: &[f32], dims: &[usize]) -> f32 {
    let (c, t, h, w) = (dims[1], dims[2], dims[3], dims[4]);
    let mut a = Vec::with_capacity(v.len());
    let mut b = Vec::with_capacity(v.len());
    for ci in 0..c {
        for ti in 0..t {
            for y in 0..h {
                let base = ((ci * t + ti) * h + y) * w;
                for x in 1..w {
                    a.push(v[base + x - 1]);
                    b.push(v[base + x]);
                }
            }
        }
    }
    corr(&a, &b)
}

fn latent_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

struct Setup {
    dev: Device,
    g: h3::pipeline::Geometry,
    zt: Tensor,
    zv: Vec<f32>,
    cond: h3::pipeline::Conditioning,
    ckpt: h3::loader::H3Checkpoint,
    dit: h3::dit::H3Dit,
}

fn setup() -> Option<Setup> {
    let dev = device()?;
    let dir = model_dir()?;
    let dump = dump_dir().or_else(|| {
        eprintln!("[probe] нет H3_DUMP_DIR с cond_hidden — пропуск");
        None
    })?;

    let paths = h3::loader::H3Paths::open(&dir).expect("paths");
    let vcfg = h3::config::VaeConfig::from_dir(&paths.root).expect("vae config");
    let (w, h, frames) = (384usize, 256usize, 39usize);
    let g = h3::pipeline::Geometry::new(w, h, frames);

    let n = 3 * frames * h * w;
    let mut px = vec![0f32; n];
    for f in 0..frames {
        for y in 0..h {
            for x in 0..w {
                let disc = (((x as f32 - 190.0).powi(2) + (y as f32 - 120.0).powi(2)).sqrt()
                    < 60.0 + f as f32) as i32 as f32;
                let base = f * h * w + y * w + x;
                px[base] = disc * 1.6 - 0.8;
                px[frames * h * w + base] = (x as f32 / w as f32) * 1.6 - 0.8;
                px[2 * frames * h * w + base] = (y as f32 / h as f32) * 1.6 - 0.8;
            }
        }
    }
    let img = Tensor::from_vec(px, vec![1, 3, frames, h, w], dev).expect("frames");
    let vw = h3::loader::ComponentLoader::open_file(paths.video_vae_file(), dev).expect("vae");
    let enc = h3::vae::VaeEncoder::load(&vw, vcfg, dev, DType::BF16).expect("encoder");
    let z = enc.encode(&img).expect("encode");
    drop(enc);
    let zt = z.to_dtype(DType::F32).unwrap().narrow(2, 0, g.latent_t).unwrap().contiguous().unwrap();
    let zv = latent_vec(&zt);
    eprintln!(
        "[probe] истинный латент {:?} σ {:.4} · соседи w {:.3}",
        zt.dims(),
        sigma_of(&zv),
        neigh_w(&zv, zt.dims())
    );

    let ctx = read_f32(&dump.join("cond_hidden.f32"));
    let tl = ctx.len() / 5120;
    let context = Tensor::from_vec(ctx, vec![1, tl, 5120], dev).expect("cond");
    let cond = h3::pipeline::Conditioning { context, text_tags: vec![1u8; tl] };

    let ckpt = h3::loader::H3Checkpoint::open(paths, dev, DType::BF16).expect("ckpt");
    let dit = h3::dit::H3Dit::load(&ckpt, dev, DType::BF16, DType::NVFP4).expect("dit");
    Some(Setup { dev, g, zt, zv, cond, ckpt, dit })
}

#[test]
#[ignore]
fn x0_structure_across_sigma() {
    let Some(s) = setup() else { return };
    let sched = h3::H3Scheduler::new(64, 12.0, 3.0);

    let mut req = h3::pipeline::DenoiseRequest::new(s.g, &s.cond);
    req.seed = Some(7);
    let prep = h3::pipeline::prepare(&s.dit, &req, &sched).expect("prepare");
    let cache = h3::pipeline::build_adaln_cache(&s.dit, &s.ckpt, &prep, DType::BF16).expect("cache");

    h3::pipeline::dump_tensor("probe_z", &s.zt);
    {
        let (_, a_noise) = h3::pipeline::init_latents(
            s.g,
            s.ckpt.config.latents_dim,
            s.ckpt.config.audio_latents_dim,
            s.dev,
            DType::BF16,
            Some(11),
        )
        .expect("noise");
        h3::pipeline::dump_tensor("probe_a", &a_noise);
    }
    let targets = [1.0f64, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.16];
    let mut steps: Vec<usize> = targets
        .iter()
        .map(|t| {
            (0..sched.steps())
                .min_by(|a, b| {
                    (sched.video_sigma(*a) - t)
                        .abs()
                        .partial_cmp(&(sched.video_sigma(*b) - t).abs())
                        .unwrap()
                })
                .unwrap()
        })
        .collect();
    steps.dedup();

    let base = cache.final_chunk(0, 1).expect("chunk0");
    for &step in &[0usize, 16, 32, 48, 60] {
        let tv = sched.video_t(step) as f32;
        let ta = sched.audio_t(step) as f32;
        let emb = s.dit.time_embed(&[tv, ta]).expect("emb");
        let direct = s
            .dit
            .adaln_final()
            .forward(&emb)
            .and_then(|d| d.narrow(0, 1, 1)?.contiguous()?.reshape(vec![cache.final_rows, cache.hidden]))
            .expect("direct");
        let cached = cache.final_chunk(step, 1).expect("chunk");
        let dv = latent_vec(&cached
            .to_dtype(DType::F32)
            .and_then(|c| c.sub(&direct.to_dtype(DType::F32)?))
            .expect("diff"));
        let d0 = latent_vec(&cached
            .to_dtype(DType::F32)
            .and_then(|c| c.sub(&base.to_dtype(DType::F32)?))
            .expect("diff0"));
        eprintln!(
            "[adaln] шаг {step} t_v {tv:.4} · |кэш−прямой| rms {:.5} · |кэш−шаг0| rms {:.5} · |кэш| rms {:.5}",
            sigma_of(&dv),
            sigma_of(&d0),
            sigma_of(&latent_vec(&cached))
        );
    }

    for step in steps {
        let sv = sched.video_sigma(step) as f32;
        let (noise, a_noise) = h3::pipeline::init_latents(
            s.g,
            s.ckpt.config.latents_dim,
            s.ckpt.config.audio_latents_dim,
            s.dev,
            DType::BF16,
            Some(11),
        )
        .expect("noise");
        let xv = noise
            .mul_scalar(sv)
            .and_then(|a| s.zt.to_dtype(DType::BF16).unwrap().mul_scalar(1.0 - sv).and_then(|b| a.add(&b)))
            .expect("noised");

        let mut r = h3::pipeline::DenoiseRequest::new(s.g, &s.cond);
        r.seed = Some(11);
        r.init_video = Some(xv.clone());
        r.init_audio = Some(a_noise);
        let out = h3::pipeline::denoise_one(&s.dit, &cache, &prep, &r, &sched, step).expect("forward");
        h3::pipeline::dump_tensor(&format!("probe_x_{step}"), &xv);
        h3::pipeline::dump_tensor(&format!("probe_v_{step}"), &out.0);
        let x0t = xv.sub(&out.0.to_dtype(DType::BF16).unwrap().mul_scalar(sv).unwrap()).unwrap();
        let x0 = latent_vec(&x0t);
        eprintln!(
            "[probe] σ {sv:.4} · x0 σ {:.4} · корр. с истиной {:.4} · соседи w {:.3}",
            sigma_of(&x0),
            corr(&s.zv, &x0),
            neigh_w(&x0, x0t.dims())
        );
    }
}

#[test]
#[ignore]
fn velocity_depends_on_timestep() {
    let Some(s) = setup() else { return };
    let sched = h3::H3Scheduler::new(64, 12.0, 3.0);

    let mut req = h3::pipeline::DenoiseRequest::new(s.g, &s.cond);
    req.seed = Some(7);
    let prep = h3::pipeline::prepare(&s.dit, &req, &sched).expect("prepare");
    let cache = h3::pipeline::build_adaln_cache(&s.dit, &s.ckpt, &prep, DType::BF16).expect("cache");

    let (noise, a_noise) = h3::pipeline::init_latents(
        s.g,
        s.ckpt.config.latents_dim,
        s.ckpt.config.audio_latents_dim,
        s.dev,
        DType::BF16,
        Some(11),
    )
    .expect("noise");
    let xv = noise
        .mul_scalar(0.5)
        .and_then(|a| s.zt.to_dtype(DType::BF16).unwrap().mul_scalar(0.5).and_then(|b| a.add(&b)))
        .expect("noised");

    let mut vels: Vec<(f32, Vec<f32>)> = Vec::new();
    for &step in &[0usize, 55, 62] {
        let mut r = h3::pipeline::DenoiseRequest::new(s.g, &s.cond);
        r.seed = Some(11);
        r.init_video = Some(xv.clone());
        r.init_audio = Some(a_noise.clone());
        let out = h3::pipeline::denoise_one(&s.dit, &cache, &prep, &r, &sched, step).expect("fwd");
        vels.push((sched.video_sigma(step) as f32, latent_vec(&out.0)));
    }
    for i in 0..vels.len() {
        for j in i + 1..vels.len() {
            let (sa, va) = &vels[i];
            let (sb, vb) = &vels[j];
            eprintln!(
                "[tdep] v(σ={sa:.3}) vs v(σ={sb:.3}): cos {:.4} · σ {:.3}/{:.3}",
                corr(va, vb),
                sigma_of(va),
                sigma_of(vb)
            );
        }
    }
}

#[test]
#[ignore]
fn positive_only_sampling_produces_structure() {
    let Some(s) = setup() else { return };
    let sched = h3::H3Scheduler::new(24, 12.0, 3.0);

    let mut req = h3::pipeline::DenoiseRequest::new(s.g, &s.cond);
    req.seed = Some(7);
    let prep = h3::pipeline::prepare(&s.dit, &req, &sched).expect("prepare");
    let cache = h3::pipeline::build_adaln_cache(&s.dit, &s.ckpt, &prep, DType::BF16).expect("cache");

    let hooks = h3::pipeline::DenoiseHooks::default();
    let out = h3::pipeline::denoise_av(&s.dit, &cache, &prep, &req, &sched, &hooks).expect("denoise");
    let v = latent_vec(&out.video_latent);
    eprintln!(
        "[probe] positive-only 24 шага: латент σ {:.4} · соседи w {:.3} · истина σ {:.4} соседи {:.3}",
        sigma_of(&v),
        neigh_w(&v, out.video_latent.dims()),
        sigma_of(&s.zv),
        neigh_w(&s.zv, s.zt.dims())
    );
}
