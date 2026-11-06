//! Замер и сверка моделей повышения разрешения на кадрах RGBA8.
//! vfx_bench <модель.syn> <кадры.rgba> <W> <H> [--device cpu|npu|cuda] [--ref cpu|cuda] [--frames N] [--out кадр.ppm]
//! кадры.rgba — подряд кадры W×H RGBA8 (ffmpeg -pix_fmt rgba -f rawvideo). `--device cuda` — свои CUDA-ядра
//! крейта; cpu/npu — путь Tensor API. `--ref` — второй путь для сверки (PSNR и max |Δ| по RGB).
use std::path::Path;
use synaptix_core::device::Device;
use synaptix_vfx::{weights::Weights, Upscaler};

fn open(dev: &str, w: &Weights, path: &Path) -> Box<dyn Upscaler> {
    match dev {
        "cuda" => synaptix_vfx::open_upscaler(path, 0).expect("cuda"),
        "npu" => synaptix_vfx::tensor::open_on(w, Device::Npu(0)).expect("npu"),
        _ => synaptix_vfx::tensor::open_on(w, Device::Cpu).expect("cpu"),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let (w, h): (usize, usize) = (a[3].parse().unwrap(), a[4].parse().unwrap());
    let raw = std::fs::read(&a[2]).expect("кадры");
    let fsz = w * h * 4;
    let nf = (raw.len() / fsz).min(get("--frames").map(|s| s.parse().unwrap()).unwrap_or(usize::MAX));
    let dev = get("--device").unwrap_or_else(|| "cpu".into());
    synaptix_kernels_cpu::ensure_registered();
    for d in [Some(dev.clone()), get("--ref")].into_iter().flatten() {
        if d == "npu" {
            eprintln!("NPU: {}", synaptix_kernels_npu::register().map(|i| i.model.name).unwrap_or("нет"));
        }
    }
    let path = Path::new(&a[1]);
    let wts = Weights::open(path).expect("веса");
    let mut m = open(&dev, &wts, path);
    eprintln!("модель {}, ×{}, {w}×{h}, кадров {nf}", m.name(), m.scale());
    let mut outs: Vec<Vec<u8>> = Vec::new();
    let mut times = Vec::new();
    for i in 0..nf {
        let t = std::time::Instant::now();
        m.push(&raw[i * fsz..(i + 1) * fsz], w as u32, h as u32, &mut |f| outs.push(f.rgba.to_vec())).expect("push");
        times.push(t.elapsed().as_secs_f64());
    }
    let first = times[0];
    let rest: Vec<f64> = times.iter().skip(1).copied().collect();
    let avg = if rest.is_empty() { first } else { rest.iter().sum::<f64>() / rest.len() as f64 };
    eprintln!("первый кадр {:.3} с; далее {:.1} мс/кадр = {:.2} fps", first, avg * 1e3, 1.0 / avg);
    if let Some(r) = get("--ref") {
        let mut rm = open(&r, &wts, path);
        let (mut se, mut n, mut mx) = (0f64, 0usize, 0u8);
        for i in 0..nf.min(outs.len()) {
            rm.push(&raw[i * fsz..(i + 1) * fsz], w as u32, h as u32, &mut |f| {
                for (p, q) in f.rgba.chunks_exact(4).zip(outs[i].chunks_exact(4)) {
                    for c in 0..3 {
                        let d = (p[c] as i32 - q[c] as i32).unsigned_abs() as u8;
                        mx = mx.max(d);
                        se += (d as f64).powi(2);
                        n += 1;
                    }
                }
            }).expect("ref");
        }
        let mse = se / n.max(1) as f64;
        eprintln!("сверка с {r}: PSNR {:.1} дБ, max |Δ| {mx}", if mse > 0.0 { 10.0 * (255.0f64 * 255.0 / mse).log10() } else { f64::INFINITY });
    }
    if let (Some(o), Some(f)) = (get("--out"), outs.first()) {
        let (ow, oh) = (w * m.scale() as usize, h * m.scale() as usize);
        let mut ppm = format!("P6\n{ow} {oh}\n255\n").into_bytes();
        for p in f.chunks_exact(4) { ppm.extend_from_slice(&p[..3]); }
        std::fs::write(o, ppm).unwrap();
    }
}
