//! Сверка с эталоном `reference/supertonic_ref.py` и замер скорости.
//!
//! supertonic_audit <bundle.syn> --text "..." [--lang en] [--voice M1] [--ref DIR]
//!     [--device cuda|cpu] [--steps 8] [--speed 1.05] [--seed 0] [--out out.wav] [--repeat N]

use std::path::{Path, PathBuf};
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_tts_supertonic::{Options, SupertonicPipeline};

fn npy(path: &Path) -> (Vec<usize>, String, Vec<u8>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = String::from_utf8_lossy(&b[off..off + hl]).to_string();
    let descr = h.split("'descr':").nth(1).unwrap().split('\'').nth(1).unwrap().to_string();
    let s = h.split("'shape':").nth(1).unwrap();
    let s = &s[s.find('(').unwrap() + 1..s.find(')').unwrap()];
    (s.split(',').filter_map(|x| x.trim().parse().ok()).collect(), descr, b[off + hl..].to_vec())
}

fn f32s(p: &Path) -> Vec<f32> {
    let (_, d, b) = npy(p);
    assert_eq!(d, "<f4");
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut d2, mut b2) = (0f64, 0f64);
    for i in 0..n {
        d2 += ((a[i] - b[i]) as f64).powi(2);
        b2 += (b[i] as f64).powi(2);
    }
    (d2 / b2.max(1e-30)).sqrt() as f32
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let device = if get("--device").as_deref() == Some("cpu") { Device::Cpu } else { Device::Cuda(0) };
    synaptix_kernels_cpu::ensure_registered();
    if device.is_cuda() {
        synaptix_kernels_cuda::ensure_registered();
    }
    let t = Instant::now();
    let p = SupertonicPipeline::from_syn(Path::new(&a[1]), device).expect("load");
    eprintln!("загрузка {:.2} с, голоса {:?}", t.elapsed().as_secs_f64(), p.voice_names());
    let text = get("--text").expect("--text");
    let lang = get("--lang").unwrap_or_else(|| "en".into());
    let voice = p.voice(&get("--voice").unwrap_or_else(|| "M1".into())).expect("голос").clone();
    let o = Options {
        steps: get("--steps").map(|s| s.parse().unwrap()).unwrap_or(8),
        speed: get("--speed").map(|s| s.parse().unwrap()).unwrap_or(1.05),
        seed: get("--seed").map(|s| s.parse().unwrap()).unwrap_or(0),
        ..Options::default()
    };
    if let Some(r) = get("--ref").map(PathBuf::from) {
        let ids = p.text_ids(&text, &lang).unwrap();
        let (_, _, rb) = npy(&r.join("ids.npy"));
        let rids: Vec<i64> = rb.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        eprintln!("ids: {} (эталон {}), {}", ids.len(), rids.len(), if ids == rids { "совпали" } else { "РАЗНЫЕ" });
        let te = {
            let n = ids.len();
            let it = Tensor::from_vec(ids.clone(), vec![n], device).unwrap();
            let _ = it;
            let (pcm, dur) = p.infer(&text, &lang, &voice, &o).unwrap();
            let rd = f32s(&r.join("duration.npy"))[0];
            eprintln!("длительность {dur:.4} с (эталон {:.4} / speed)", rd / o.speed);
            let rw = f32s(&r.join("wav.npy"));
            eprintln!("звук: {} vs {} семплов, отн. ошибка {:.2e}", pcm.len(), rw.len(), rel(&pcm, &rw));
            pcm
        };
        let _ = te;
    }
    let reps: usize = get("--repeat").map(|s| s.parse().unwrap()).unwrap_or(1);
    let mut pcm = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        pcm = p.synthesize(&text, &lang, &voice, &o).expect("synth");
        let el = t.elapsed().as_secs_f64();
        let tm = p.last_timings();
        eprintln!(
            "кусков {}, звук {:.2} с за {el:.3} с (DP+текст {:.3}, поток {:.3}, вокодер {:.3}), RTF {:.3}",
            tm.chunks, tm.audio_s, tm.dp_te_s, tm.flow_s, tm.vocoder_s, el / tm.audio_s.max(1e-9)
        );
    }
    if let Some(out) = get("--out") {
        synaptix_audio::io::write_wav_mono_f32(&out, &pcm, 44100).unwrap();
        eprintln!("→ {out}");
    }
}
