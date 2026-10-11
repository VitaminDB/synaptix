use synaptix_asr_wav2vec2::Wav2Vec2Ctc;
use synaptix_core::{device::Device, dtype::DType};

fn edit_distance(a: &[u32], b: &[u32]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (x != y) as usize).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

struct Stat {
    kl: f64,
    frames: usize,
    agree: usize,
    edits: usize,
    ref_len: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reference = &args[1];
    let dir = &args[2];
    let candidates: Vec<(String, DType)> = args[3..]
        .iter()
        .map(|a| {
            let (p, d) = a.rsplit_once('@').unwrap_or((a, "f16"));
            (p.to_string(), if d == "f32" { DType::F32 } else { DType::F16 })
        })
        .collect();
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let mut wavs: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "wav")).collect();
    wavs.sort();
    let limit: usize = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    wavs.truncate(limit);
    let audio: Vec<(Vec<f32>, u32)> = wavs.iter().map(|w| synaptix_audio::io::read_wav_mono_f32(w).unwrap()).collect();
    let r = Wav2Vec2Ctc::load(reference, &dev, DType::F32).unwrap();
    let refs: Vec<_> = audio.iter().map(|(p, sr)| r.log_probs(p, *sr).unwrap()).collect();
    drop(r);
    for (path, dt) in &candidates {
        let m = Wav2Vec2Ctc::load(path, &dev, *dt).unwrap();
        let mut s = Stat { kl: 0.0, frames: 0, agree: 0, edits: 0, ref_len: 0 };
        for ((p, sr), lr) in audio.iter().zip(&refs) {
            let lc = m.log_probs(p, *sr).unwrap();
            let t = lr.frames.min(lc.frames);
            for f in 0..t {
                let (a, b) = (lr.row(f), lc.row(f));
                let mut kl = 0.0f64;
                for (x, y) in a.iter().zip(b) {
                    kl += (x.exp() as f64) * (*x as f64 - *y as f64);
                }
                s.kl += kl.max(0.0);
                let am = |r: &[f32]| r.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
                s.agree += (am(a) == am(b)) as usize;
            }
            s.frames += t;
            let (ga, gb) = (m.greedy(lr), m.greedy(&lc));
            s.edits += edit_distance(&ga, &gb);
            s.ref_len += ga.len();
        }
        println!(
            "{path} @{dt:?}: KL {:.5} нат/кадр, argmax {:.2}%, расхождение фонем {:.2}% ({} записей, {} кадров)",
            s.kl / s.frames as f64,
            100.0 * s.agree as f64 / s.frames as f64,
            100.0 * s.edits as f64 / s.ref_len.max(1) as f64,
            audio.len(),
            s.frames
        );
    }
}
