//! RIFE: сверка с эталоном PyTorch (t = 0,5) и замер.
//!   rife_check МОДЕЛЬ ЭТАЛОН.safetensors
//!   rife_check МОДЕЛЬ --bench ШxВ [кадров] [масштаб потока]
use std::path::Path;
use std::time::Instant;

fn to_rgba(x: &[f32], h: usize, w: usize) -> Vec<u8> {
    let mut v = vec![255u8; h * w * 4];
    for p in 0..h * w {
        for c in 0..3 {
            v[p * 4 + c] = (x[c * h * w + p] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    v
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let wts = synaptix_vfx::weights::Weights::open(Path::new(&a[0])).expect("веса");
    let g = synaptix_vfx::gpu::Gpu::new(0).expect("gpu");
    let mut r = synaptix_vfx::rife::Rife::load(g, &wts).expect("rife");
    if a.get(1).map(String::as_str) == Some("--bench") {
        let (w, h) = a[2].split_once('x').map(|(x, y)| (x.parse::<usize>().unwrap(), y.parse::<usize>().unwrap())).unwrap();
        let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);
        if let Some(s) = a.get(4).and_then(|s| s.parse::<f32>().ok()) {
            r.set_flow_scale(s);
        }
        let mut f = vec![0u8; w * h * 4];
        let mut seed = 1u32;
        for p in f.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *p = (seed >> 24) as u8;
        }
        r.push(&f, w, h).unwrap();
        r.push(&f, w, h).unwrap();
        r.interpolate(0.5).unwrap();
        let t0 = Instant::now();
        for _ in 0..n {
            r.push(&f, w, h).unwrap();
            r.interpolate(0.5).unwrap();
        }
        let dt = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("{}: {w}x{h}: кадр + промежуточный — {dt:.1} мс", r.name());
        return;
    }
    let refw = synaptix_vfx::weights::Weights::open(Path::new(&a[1])).expect("эталон");
    let sh = refw.shape("img0").unwrap().to_vec();
    let (h, w) = (sh[2], sh[3]);
    r.push(&to_rgba(refw.get("img0").unwrap(), h, w), w, h).unwrap();
    r.push(&to_rgba(refw.get("img1").unwrap(), h, w), w, h).unwrap();
    let out = r.interpolate(0.5).unwrap().to_vec();
    let y = refw.get("y").unwrap();
    let (mut max, mut se) = (0f32, 0f64);
    for p in 0..h * w {
        for c in 0..3 {
            let d = (out[p * 4 + c] as f32 / 255.0 - y[c * h * w + p].clamp(0.0, 1.0)).abs();
            max = max.max(d);
            se += (d * d) as f64;
        }
    }
    let psnr = 10.0 * (1.0 / (se / (h * w * 3) as f64)).log10();
    println!("{}: {w}x{h} t=0,5: max |Δ| {:.4} ({:.1}/255), PSNR к эталону {psnr:.1} дБ", r.name(), max, max * 255.0);
}
