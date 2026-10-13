//! Генерация картинки кодом службы synaptixd (`image::generate`):
//! `gen_image модель.syn out.png "описание" [--w N] [--h N] [--steps N] [--seed N] [--quant q]`.

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_gen::image::{self, ImageRequest};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!("использование: gen_image модель.syn out.png \"описание\" [флаги]");
        std::process::exit(2);
    }
    let mut req = ImageRequest { prompt: a[2].clone(), seed: 7, ..Default::default() };
    let mut i = 3;
    while i + 1 < a.len() {
        let v = a[i + 1].clone();
        match a[i].as_str() {
            "--w" => req.width = v.parse().ok(),
            "--h" => req.height = v.parse().ok(),
            "--steps" => req.steps = v.parse().ok(),
            "--seed" => req.seed = v.parse().unwrap_or(7),
            "--quant" => req.quant = Some(v),
            "--ref" => req.images.push(v.into()),
            other => panic!("неизвестный флаг {other}"),
        }
        i += 2;
    }
    let t0 = std::time::Instant::now();
    let mut progress = |s: usize, n: usize| {
        eprint!("\r[{:>6.1} с] шаг {s}/{n}   ", t0.elapsed().as_secs_f32());
        true
    };
    match image::generate(&PathBuf::from(&a[0]), Device::Cuda(0), &req, &mut progress) {
        Ok(o) => {
            image::save_png(&o.image, &PathBuf::from(&a[1])).unwrap();
            eprintln!();
            println!("[{:.1} с] {}", t0.elapsed().as_secs_f32(), o.label);
        }
        Err(e) => {
            eprintln!("\nошибка: {e}");
            std::process::exit(1);
        }
    }
}
