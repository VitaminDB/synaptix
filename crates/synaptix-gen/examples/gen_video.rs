//! Генерация видео тем же кодом, что у службы synaptixd (`video::ltx`,
//! `video::h3`), — для замеров и проверок из терминала:
//! `gen_video ltx|h3 out.mp4 "описание" [--audio f] [--image f] [--video f (LipDub с --audio)] [--ref f]… [--w N] [--h N] [--secs S] [--steps N] [--seed N] [--quant q] [--two-stage]`.
//! Модели — из библиотеки `~/Storage/Нейросети/Модели/video` (или `SYN_VIDEO_DIR`).

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use synaptix_core::device::Device;
use synaptix_gen::video::{self, LtxPaths, VideoRequest};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!("использование: gen_video ltx|h3 out.mp4 \"описание\" [флаги]");
        std::process::exit(2);
    }
    let dir = std::env::var("SYN_VIDEO_DIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Storage/Нейросети/Модели/video"));
    let mut req = VideoRequest { prompt: a[2].clone(), seed: 7, ..Default::default() };
    let mut i = 3;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--audio" => req.audio = Some(v.into()),
            "--image" => req.first_frame = Some(v.into()),
            "--video" => req.control_video = Some(v.into()),
            "--ref" => req.refs.push(v.into()),
            "--w" => req.width = v.parse().ok(),
            "--h" => req.height = v.parse().ok(),
            "--secs" => req.seconds = v.parse().ok(),
            "--steps" => req.steps = v.parse().ok(),
            "--seed" => req.seed = v.parse().unwrap_or(7),
            "--quant" => req.quant = Some(v),
            "--two-stage" => {
                req.two_stage = true;
                i += 1;
                continue;
            }
            other => panic!("неизвестный флаг {other}"),
        }
        i += 2;
    }
    let out = PathBuf::from(&a[1]);
    let cancel = AtomicBool::new(false);
    let t0 = std::time::Instant::now();
    let progress = |s: usize, n: usize| eprint!("\r[{:>6.1} с] шаг {s}/{n}   ", t0.elapsed().as_secs_f32());
    let r = match a[0].as_str() {
        "ltx" => {
            let paths = LtxPaths {
                model: dir.join("ltx-2.3-22b-distilled-1.1.syn"),
                gemma: dir.join("ltx-gemma-3-12b-qat.syn"),
                upscaler: Some(dir.join("ltx-2.3-spatial-upscaler-x2-1.1.syn")),
                ic_lipdub: Some(dir.join("ltx-2.3-22b-ic-lora-lipdub-0.9.syn")),
                ic_union: Some(dir.join("ltx-2.3-22b-ic-lora-union-control-ref0.5.syn")),
                ic_motion: Some(dir.join("ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.syn")),
                ..Default::default()
            };
            video::ltx(&paths, Device::Cuda(0), &req, &out, &cancel, &progress)
        }
        "h3" => {
            let model = dir.join(if req.refs.is_empty() { "minimax-h3-fl2va.syn" } else { "minimax-h3-ref2va.syn" });
            video::h3(&model, Some(&dir.join("minimax-h3-qwen3vl-encoder.syn")), Device::Cuda(0), &req, &out, &cancel, &progress)
        }
        m => panic!("модель {m}?"),
    };
    eprintln!();
    match r {
        Ok(o) => println!("[{:.1} с] {} — {}×{} {} кадров", t0.elapsed().as_secs_f32(), o.label, o.width, o.height, o.frames),
        Err(e) => {
            eprintln!("ошибка: {e}");
            std::process::exit(1);
        }
    }
}
