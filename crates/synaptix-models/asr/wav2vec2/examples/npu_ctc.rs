//! Замер CTC на NPU (как служба: F16, корзины длины) — время прохода и статистика NPU.
//! npu_ctc <model.syn[,model2.syn]> <wav[,wav2,…]> [кругов=2] [cpu]
//! Модели и записи чередуются, как у synspeak (обе модели на каждую фразу).
//! `SYNAPTIX_NPU_LOG=1` — каждый сегмент графа (узлы, время исполнения).
use std::time::Instant;
use synaptix_asr_wav2vec2::Wav2Vec2Ctc;
use synaptix_core::{device::Device, dtype::DType};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    synaptix_kernels_cpu::ensure_registered();
    let dev = if a.get(4).map(String::as_str) == Some("cpu") {
        Device::Cpu
    } else {
        let info = synaptix_kernels_npu::register().expect("npu");
        eprintln!("NPU: {} (soc {})", info.model.name, info.soc);
        Device::Npu(0)
    };
    let dt = if matches!(dev, Device::Npu(_)) { DType::F16 } else { DType::F32 };
    let models: Vec<(String, Wav2Vec2Ctc)> = a[1]
        .split(',')
        .map(|p| {
            let t = Instant::now();
            let m = Wav2Vec2Ctc::load_quant(p, &dev, dt, None).expect("load");
            eprintln!("загрузка {p}: {:.2} с", t.elapsed().as_secs_f64());
            (p.rsplit('/').nth(1).unwrap_or(p).to_string(), m)
        })
        .collect();
    let wavs: Vec<(String, Vec<f32>, u32)> = a[2]
        .split(',')
        .map(|w| {
            let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(w).expect("wav");
            (w.to_string(), pcm, sr)
        })
        .collect();
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    for round in 0..n {
        for (wn, pcm, sr) in &wavs {
            for (mn, m) in &models {
                let s0 = synaptix_kernels_npu::stats();
                let t = Instant::now();
                let r = m.log_probs(pcm, *sr);
                let el = t.elapsed().as_secs_f64();
                let s1 = synaptix_kernels_npu::stats();
                match r {
                    Ok(lp) => eprintln!(
                        "круг {round} {mn} {wn} ({:.1} с): {:.3} с, кадров {}; сегментов {}, сборок/открытий {}, NPU {:.0} мс, открытие {:.0} мс, на CPU {}",
                        pcm.len() as f64 / *sr as f64,
                        el,
                        lp.frames,
                        s1.flushes - s0.flushes,
                        s1.compiles - s0.compiles,
                        s1.exec_ms - s0.exec_ms,
                        s1.compile_ms - s0.compile_ms,
                        s1.cpu_fallbacks - s0.cpu_fallbacks
                    ),
                    Err(e) => eprintln!("круг {round} {mn} {wn}: ОШИБКА {e}"),
                }
            }
        }
    }
}
