//! Сверка частей Seed-VC v2 с дампами эталона (`scripts/reference/seedvc_parts.py`,
//! `seedvc_full.py`): по каждой точке — макс. абс. разность и косинус, для
//! токенов — доля совпавших. `seedvc_check BUNDLE.syn REF_DIR [этапы через запятую]`
//! Этапы: mel, style, lr, dit, cfm, ar, hubert, astral, bigvgan, resample, full;
//! отдельно `v1` — с бандлом `seed-vc-v1.syn` и дампами `seedvc_v1.py`
//! (`SEEDVC_STEPS` — шагов в полном прогоне, как у эталона; по умолчанию 30).
//! Заданные входы — `a·sin(w·i + p)` и `(i·mul + add) mod m`, как в `seedvc_common.py`.
use std::path::Path;

use synaptix_core::{device::Device, tensor::Tensor};
use synaptix_vc_seed::ar::Ar;
use synaptix_vc_seed::astral::{Astral, Hubert};
use synaptix_vc_seed::campplus::{fbank, CamPlus};
use synaptix_vc_seed::cfm::{sample, CfmParams, Dit};
use synaptix_vc_seed::loader::Comp;
use synaptix_vc_seed::mel::mel_spectrogram;
use synaptix_vc_seed::nn::host;
use synaptix_vc_seed::pipeline::{wav_bytes, Waves};
use synaptix_vc_seed::regulator::CfmRegulator;
use synaptix_vc_seed::vocoder::Vocoder;
use synaptix_vc_seed::{SeedVc, SeedVcRequest};

fn find(dir: &Path, name: &str, ext: &str) -> Option<(std::path::PathBuf, Vec<usize>)> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let f = e.file_name().to_string_lossy().to_string();
        let parts: Vec<&str> = f.rsplitn(3, '.').collect();
        if parts.len() == 3 && parts[2] == name && parts[0] == ext {
            return Some((e.path(), parts[1].split('x').map(|d| d.parse().unwrap()).collect()));
        }
    }
    None
}

fn f32s(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let (p, d) = find(dir, name, "f32").unwrap_or_else(|| panic!("нет дампа {name}"));
    (std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(), d)
}

fn i32s(dir: &Path, name: &str) -> Vec<u32> {
    let (p, _) = find(dir, name, "i32").unwrap_or_else(|| panic!("нет дампа {name}"));
    std::fs::read(p).unwrap().chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect()
}

fn wave(shape: Vec<usize>, a: f64, w: f64, p: f64, dev: Device) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec((0..n).map(|i| (a * (w * i as f64 + p).sin()) as f32).collect::<Vec<f32>>(), shape, dev).unwrap()
}

fn toks(n: usize, m: usize, mul: usize, add: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * mul + add) % m) as u32).collect()
}

/// Печать метрики: макс. абс. разность, косинус, макс. |эталон|.
fn cmp(name: &str, ours: &[f32], reference: &[f32]) {
    if ours.len() != reference.len() {
        println!("{name}: РАЗМЕР {} против {} у эталона", ours.len(), reference.len());
        return;
    }
    let (mut maxd, mut dot, mut na, mut nb, mut maxr) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in ours.iter().zip(reference) {
        let (a, b) = (a as f64, b as f64);
        maxd = maxd.max((a - b).abs());
        maxr = maxr.max(b.abs());
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    println!("{name}: макс|Δ| = {maxd:.3e}, cos = {:.9}, макс|эталон| = {maxr:.3} (n = {})", dot / (na.sqrt() * nb.sqrt()).max(1e-30), ours.len());
}

fn cmp_tok(name: &str, ours: &[u32], reference: &[u32]) {
    let n = ours.len().min(reference.len());
    let same = (0..n).filter(|&i| ours[i] == reference[i]).count();
    let prefix = (0..n).take_while(|&i| ours[i] == reference[i]).count();
    println!("{name}: совпало {same} из {} (у нас {}), общий префикс {prefix}", reference.len(), ours.len());
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    synaptix_kernels_cpu::ensure_registered();
    let dev = if std::env::var("SEEDVC_DEVICE").as_deref() == Ok("cuda") {
        synaptix_kernels_cuda::ensure_registered();
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let (bundle, dir) = (Path::new(&a[0]), Path::new(&a[1]));
    let all = "mel,style,lr,dit,cfm,ar,hubert,astral,bigvgan,resample,full".to_string();
    let stages: Vec<String> = a.get(2).unwrap_or(&all).split(',').map(str::to_string).collect();
    let on = |s: &str| stages.iter().any(|x| x == s);
    let comp = |n: &str| Comp::open(bundle, n, dev).unwrap();
    synaptix_core::grad::no_grad(|| {
        if on("mel") {
            for n in ["src", "ref"] {
                let (w, _) = f32s(dir, &format!("{n}_22k"));
                cmp(&format!("mel_{n}"), &mel_spectrogram(&w).unwrap().0, &f32s(dir, &format!("mel_{n}")).0);
            }
        }
        if on("resample") {
            for n in ["src", "ref"] {
                let (w, _) = f32s(dir, &format!("{n}_22k"));
                let ours = synaptix_audio::resample(&w, 22_050, 16_000).unwrap();
                let (r, _) = f32s(dir, &format!("{n}_16k"));
                let k = ours.len().min(r.len());
                println!("resample_{n}: длина {} против {} у эталона (soxr_hq)", ours.len(), r.len());
                cmp(&format!("resample_{n}"), &ours[..k], &r[..k]);
                let ta = synaptix_audio::resample_torchaudio(&w, 22_050, 16_000).unwrap();
                let k = ta.len().min(r.len());
                cmp(&format!("resample_torchaudio_{n}"), &ta[..k], &r[..k]);
            }
            // Насколько своя передискретизация меняет токены содержания (эталонные — по волне soxr).
            if find(dir, "tok_wide_src", "i32").is_some() {
                let hub = Hubert::load(bundle, dev).unwrap();
                for (tag, c) in [("wide", "astral_wide"), ("narrow", "astral_narrow")] {
                    let q = Astral::load(&comp(c)).unwrap();
                    for n in ["src", "ref"] {
                        let w22 = f32s(dir, &format!("{n}_22k")).0;
                        let rt = i32s(dir, &format!("tok_{tag}_{n}"));
                        let ours = synaptix_audio::resample(&w22, 22_050, 16_000).unwrap();
                        cmp_tok(&format!("tok_{tag}_{n} (resample: sinc Блэкмана)"), &q.tokens(&hub.features(&ours).unwrap()).unwrap(), &rt);
                        let ta = synaptix_audio::resample_torchaudio(&w22, 22_050, 16_000).unwrap();
                        cmp_tok(&format!("tok_{tag}_{n} (resample_torchaudio)"), &q.tokens(&hub.features(&ta).unwrap()).unwrap(), &rt);
                    }
                }
            }
        }
        if on("style") {
            let (w, _) = f32s(dir, "ref_16k");
            let (f, frames) = fbank(&w);
            cmp("fbank_ref", &f, &f32s(dir, "fbank_ref").0);
            let cam = CamPlus::load(&comp("campplus")).unwrap();
            let (rf, _) = f32s(dir, "fbank_ref");
            cmp("style_ref (по эталонному fbank)", &cam.embed(&rf, rf.len() / 80, true).unwrap(), &f32s(dir, "style_ref").0);
            cmp("style_ref_nolen (по эталонному fbank)", &cam.embed(&rf, rf.len() / 80, false).unwrap(), &f32s(dir, "style_ref_nolen").0);
            cmp("style_ref (свой fbank)", &cam.embed(&f, frames, true).unwrap(), &f32s(dir, "style_ref").0);
        }
        if on("lr") || on("dit") || on("cfm") {
            let c = comp("cfm");
            let lr = CfmRegulator::load(&c).unwrap();
            let style = wave(vec![1, 192], 0.5, 0.31, 0.2, dev);
            if on("lr") {
                cmp("lr_cond", &host(&lr.forward(&toks(100, 2048, 7919, 13), 172).unwrap()).unwrap(), &f32s(dir, "lr_cond").0);
            }
            if on("dit") || on("cfm") {
                let dit = Dit::load(&c).unwrap();
                if on("dit") {
                    let t = 120;
                    let x = wave(vec![1, 80, t], 0.8, 0.11, 0.0, dev);
                    let mut p = host(&wave(vec![1, 80, t], 0.5, 0.07, 1.0, dev)).unwrap();
                    for ch in 0..80 {
                        for i in 50..t {
                            p[ch * t + i] = 0.0;
                        }
                    }
                    let prompt = Tensor::from_vec(p, vec![1, 80, t], dev).unwrap();
                    let cond = wave(vec![1, t, 512], 0.3, 0.013, 0.5, dev);
                    cmp("dit_v", &host(&dit.forward(&x, &prompt, 0.4, &style, &cond).unwrap()).unwrap(), &f32s(dir, "dit_v").0);
                }
                if on("cfm") {
                    let pc = lr.forward(&toks(60, 2048, 6151, 5), 103).unwrap();
                    let sc = lr.forward(&toks(100, 2048, 7919, 13), 172).unwrap();
                    let mu = Tensor::cat(&[&pc, &sc], 1).unwrap().contiguous().unwrap();
                    let prompt = wave(vec![1, 80, 103], 1.5, 0.05, 0.3, dev).add_scalar(-4.0).unwrap();
                    let (z, zd) = f32s(dir, "cfm_z");
                    let z = Tensor::from_vec(z, vec![1, zd[0], zd[1]], dev).unwrap();
                    for (name, ic, sc_, rv, steps) in [("cfm_mel_77", 0.7, 0.7, false, 6), ("cfm_mel_70", 0.7, 0.0, false, 2), ("cfm_mel_07", 0.0, 0.7, false, 2), ("cfm_mel_00", 0.0, 0.0, false, 2), ("cfm_mel_anon", 0.7, 0.7, true, 2)] {
                        let p = CfmParams { steps, intelligibility: ic, similarity: sc_, random_voice: rv };
                        let y = sample(&dit, &mu, &prompt, &style, &z, &p, &mut |_, _| true).unwrap();
                        cmp(name, &host(&y).unwrap(), &f32s(dir, name).0);
                    }
                }
            }
        }
        if on("ar") {
            let ar = Ar::load(&comp("ar")).unwrap();
            let narrow = [toks(40, 32, 7, 3), toks(60, 32, 11, 1)].concat();
            let target = toks(80, 2048, 6151, 5);
            cmp("ar_cond", &ar.cond_embeddings(&narrow).unwrap(), &f32s(dir, "ar_cond").0);
            let (rl, rd) = f32s(dir, "ar_logits");
            let t = std::time::Instant::now();
            let (trace, seq) = ar.greedy_trace(&narrow, &target, rd[0], 120).unwrap();
            let el = t.elapsed().as_secs_f64();
            for (i, l) in trace.iter().enumerate() {
                cmp(&format!("ar_logits[{i}]"), l, &rl[i * rd[1]..(i + 1) * rd[1]]);
            }
            cmp_tok("ar_greedy", &seq, &i32s(dir, "ar_greedy"));
            println!("ar: {} токенов за {el:.2} с", seq.len());
        }
        if on("hubert") || on("astral") {
            let hub = Hubert::load(bundle, dev).unwrap();
            let mut feats = Vec::new();
            for n in ["src", "ref"] {
                let (w, _) = f32s(dir, &format!("{n}_16k"));
                let t = std::time::Instant::now();
                let f = hub.features(&w).unwrap();
                println!("hubert_{n}: {:?} за {:.2} с", f.dims(), t.elapsed().as_secs_f64());
                cmp(&format!("hubert_{n}"), &host(&f).unwrap(), &f32s(dir, &format!("hubert_{n}")).0);
                feats.push(f);
            }
            if on("astral") {
                for (tag, c) in [("wide", "astral_wide"), ("narrow", "astral_narrow")] {
                    let q = Astral::load(&comp(c)).unwrap();
                    for (n, ours) in ["src", "ref"].iter().zip(&feats) {
                        // По эталонным признакам HuBERT — сам токенизатор; по своим — вся цепочка.
                        let (rf, rd) = f32s(dir, &format!("hubert_{n}"));
                        let h = q.encode(&Tensor::from_vec(rf, vec![1, rd[0], rd[1]], dev).unwrap()).unwrap();
                        cmp(&format!("astral_{tag}_hidden_{n} (эталонный HuBERT)"), &host(&h).unwrap(), &f32s(dir, &format!("astral_{tag}_hidden_{n}")).0);
                        let rt = i32s(dir, &format!("tok_{tag}_{n}"));
                        cmp_tok(&format!("tok_{tag}_{n} (эталонный HuBERT)"), &q.quantize(&h).unwrap(), &rt);
                        cmp_tok(&format!("tok_{tag}_{n} (свой HuBERT)"), &q.tokens(ours).unwrap(), &rt);
                    }
                }
            }
        }
        if on("bigvgan") {
            let voc = Vocoder::load(&comp("bigvgan")).unwrap();
            let (m, d) = f32s(dir, "mel_src");
            let t = std::time::Instant::now();
            let w = voc.decode(&Tensor::from_vec(m, vec![1, d[0], d[1]], dev).unwrap()).unwrap();
            println!("bigvgan: {} сэмплов за {:.2} с", w.len(), t.elapsed().as_secs_f64());
            cmp("bigvgan_src", &w, &f32s(dir, "bigvgan_src").0);
        }
        if on("v1") {
            // Бандл v1 (`seed-vc-v1.syn`); дампы — `scripts/reference/seedvc_v1.py`.
            use synaptix_vc_seed::v1;
            let m = synaptix_vc_seed::SeedVcV1::load(bundle, dev).unwrap();
            let (s22, r22) = (f32s(dir, "src_22k").0, f32s(dir, "ref_22k").0);
            let (s16, r16) = (f32s(dir, "v1_src_16k").0, f32s(dir, "v1_ref_16k").0);
            cmp("v1_src_16k (resample_torchaudio)", &synaptix_audio::resample_torchaudio(&s22, 22_050, 16_000).unwrap(), &s16);
            let fs = m.content.features_long(&s16).unwrap();
            let fr = m.content.features(&r16).unwrap();
            cmp("v1_whisper_src", &host(&fs).unwrap(), &f32s(dir, "v1_whisper_src").0);
            cmp("v1_whisper_ref", &host(&fr).unwrap(), &f32s(dir, "v1_whisper_ref").0);
            let (f, frames) = fbank(&r16);
            cmp("v1_style", &m.campplus.embed(&f, frames, false).unwrap(), &f32s(dir, "v1_style").0);
            let (rs, rsd) = f32s(dir, "v1_whisper_src");
            let (lr_ref, lrd) = f32s(dir, "v1_lr_src");
            let c = m.regulator.forward(&Tensor::from_vec(rs, vec![1, rsd[0], rsd[1]], dev).unwrap(), lrd[0]).unwrap();
            cmp("v1_lr_src (эталонный Whisper)", &host(&c).unwrap(), &lr_ref);
            cmp("v1_lr_src (свой Whisper)", &host(&m.regulator.forward(&fs, lrd[0]).unwrap()).unwrap(), &lr_ref);
            let t = 120;
            let x = wave(vec![1, 80, t], 0.8, 0.11, 0.0, dev);
            let mut p = host(&wave(vec![1, 80, t], 0.5, 0.07, 1.0, dev)).unwrap();
            for ch in 0..80 {
                for i in 50..t {
                    p[ch * t + i] = 0.0;
                }
            }
            let prompt = Tensor::from_vec(p, vec![1, 80, t], dev).unwrap();
            let v = m.dit.forward(&x, &prompt, 0.4, &wave(vec![1, 192], 0.5, 0.31, 0.2, dev), &wave(vec![1, t, 512], 0.3, 0.013, 0.5, dev)).unwrap();
            cmp("v1_dit_v", &host(&v).unwrap(), &f32s(dir, "v1_dit_v").0);
            let z: Vec<Vec<f32>> = (0..).map_while(|i| find(dir, &format!("v1_z{i}"), "f32").map(|_| f32s(dir, &format!("v1_z{i}")).0)).collect();
            let steps: usize = std::env::var("SEEDVC_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
            let req = v1::SeedVcV1Request { steps, ..Default::default() };
            let out = m.convert_waves(&Waves { src_22k: &s22, ref_22k: &r22, src_16k: &s16, ref_16k: &r16 }, &req, Some(&z), &mut |_, _, _| true).unwrap();
            cmp("v1_wave", &out.pcm, &f32s(dir, "v1_wave").0);
            std::fs::write(dir.join("ours_v1.wav"), wav_bytes(&out.pcm, out.sample_rate)).unwrap();
            return;
        }
        if on("full") {
            let m = SeedVc::load(bundle, dev).unwrap();
            let (s22, r22, s16, r16) = (f32s(dir, "src_22k").0, f32s(dir, "ref_22k").0, f32s(dir, "src_16k").0, f32s(dir, "ref_16k").0);
            let w = Waves { src_22k: &s22, ref_22k: &r22, src_16k: &s16, ref_16k: &r16 };
            // Без манеры: тот же шум, что у эталона → та же волна.
            let noise = |mode: &str| -> Vec<Vec<f32>> { (0..).map_while(|i| find(dir, &format!("full_{mode}_z{i}"), "f32").map(|_| f32s(dir, &format!("full_{mode}_z{i}")).0)).collect() };
            let z = noise("timbre");
            println!("full_timbre: окон {}", z.len());
            let steps: usize = std::env::var("SEEDVC_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
            let req = SeedVcRequest { steps, ..Default::default() };
            let out = m.convert_waves(&w, &req, Some(&z), &mut |_, _, _| true).unwrap();
            cmp("full_timbre_wave", &out.pcm, &f32s(dir, "full_timbre_wave").0);
            std::fs::write(dir.join("ours_full_timbre.wav"), wav_bytes(&out.pcm, out.sample_rate)).unwrap();
            // С манерой: шум сэмплера токенов в эталоне подменён на `nn::Rng` (сид 4321), шум CFM — эталона.
            let z = noise("style");
            if z.is_empty() {
                return;
            }
            let req = SeedVcRequest { convert_style: true, seed: 4321, steps, ..Default::default() };
            match m.convert_waves(&w, &req, Some(&z), &mut |_, _, _| true) {
                Ok(out) => {
                    cmp_tok("full_style_tokens", &out.ar_tokens, &i32s(dir, "full_style_tokens"));
                    cmp("full_style_wave", &out.pcm, &f32s(dir, "full_style_wave").0);
                    std::fs::write(dir.join("ours_full_style.wav"), wav_bytes(&out.pcm, out.sample_rate)).unwrap();
                }
                Err(e) => println!("full_style: {e} (при расхождении токенов длина шума эталона не подходит)"),
            }
        }
    });
}
