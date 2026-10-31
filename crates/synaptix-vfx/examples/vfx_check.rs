//! Сверка модели с эталоном PyTorch и замер скорости.
//!   vfx_check МОДЕЛЬ(.syn|каталог) ЭТАЛОН.safetensors [окно]   — x [T,3,H,W] → y [T,3,sH,sW], ошибка
//!   vfx_check МОДЕЛЬ --bench ШxВ [кадров]                       — время кадра на шуме
use std::path::Path;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let model = Path::new(&a[0]);
    let w = synaptix_vfx::weights::Weights::open(model).expect("веса");
    let g = synaptix_vfx::gpu::Gpu::new(0).expect("gpu");
    if a.get(1).map(String::as_str) == Some("--raw") {
        // --raw ВХОД.rgba ШxВ ВЫХОД.rgba [повторов] — один кадр (видеомодели: повторён до окна)
        let (bw, bh) = a[3].split_once('x').map(|(x, y)| (x.parse::<u32>().unwrap(), y.parse::<u32>().unwrap())).unwrap();
        let frame = std::fs::read(&a[2]).expect("вход");
        let mut up = synaptix_vfx::open_with(g.clone(), &w).expect("модель");
        let mut out: Option<Vec<u8>> = None;
        let mut dims = (0, 0);
        for _ in 0..up.latency() + 1 {
            up.push(&frame, bw, bh, &mut |f| {
                if out.is_none() {
                    out = Some(f.rgba.to_vec());
                    dims = (f.width, f.height);
                }
            })
            .expect("push");
        }
        up.flush(&mut |f| {
            if out.is_none() {
                out = Some(f.rgba.to_vec());
                dims = (f.width, f.height);
            }
        })
        .unwrap();
        std::fs::write(&a[4], out.expect("нет кадра")).unwrap();
        println!("{}x{}", dims.0, dims.1);
        return;
    }
    if a.get(1).map(String::as_str) == Some("--bench") {
        let (bw, bh) = a[2].split_once('x').map(|(x, y)| (x.parse::<u32>().unwrap(), y.parse::<u32>().unwrap())).unwrap();
        let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(60);
        let mut up = synaptix_vfx::open_with(g.clone(), &w).expect("модель");
        let mut frame = vec![0u8; (bw * bh * 4) as usize];
        let mut seed = 1u32;
        for p in frame.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *p = (seed >> 24) as u8;
        }
        let mut got = 0usize;
        let mut first = None;
        let t0 = Instant::now();
        for _ in 0..n {
            up.push(&frame, bw, bh, &mut |_f| {
                got += 1;
                if first.is_none() {
                    first = Some(t0.elapsed());
                }
            })
            .expect("push");
        }
        let dt = t0.elapsed().as_secs_f64();
        println!("{}: {bw}x{bh} → ×{}: {got} кадров за {dt:.2} с — {:.1} мс/кадр ({:.1} кадр/с), первый через {:?}", up.name(), up.scale(), dt * 1000.0 / got.max(1) as f64, got as f64 / dt, first.unwrap_or_default());
        return;
    }
    let r = synaptix_vfx::weights::Weights::open(Path::new(&a[1])).expect("эталон");
    let xs = r.shape("x").unwrap().to_vec();
    let (t, h, wd) = (xs[0], xs[2], xs[3]);
    let x = r.get("x").unwrap();
    let y = r.get("y").unwrap();
    let ys = r.shape("y").unwrap().to_vec();
    let (oh, ow) = (ys[2], ys[3]);
    let win: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(t);
    let mut up: Box<dyn synaptix_vfx::Upscaler> = match w.arch() {
        "nanovsr" => Box::new(synaptix_vfx::nanovsr::NanoVsr::load(g.clone(), &w).unwrap().with_window(win, 0)),
        _ => synaptix_vfx::open_with(g.clone(), &w).unwrap(),
    };
    let mut outs: Vec<Vec<u8>> = Vec::new();
    for i in 0..t {
        let mut rgba = vec![255u8; h * wd * 4];
        for p in 0..h * wd {
            for c in 0..3 {
                rgba[p * 4 + c] = (x[((i * 3 + c) * h * wd) + p] * 255.0).round() as u8;
            }
        }
        up.push(&rgba, wd as u32, h as u32, &mut |f| outs.push(f.rgba.to_vec())).unwrap();
    }
    up.flush(&mut |f| outs.push(f.rgba.to_vec())).unwrap();
    assert_eq!(outs.len(), t, "кадров на выходе");
    let (mut max, mut sum, mut se) = (0f32, 0f64, 0f64);
    for i in 0..t {
        for p in 0..oh * ow {
            for c in 0..3 {
                let refv = y[((i * 3 + c) * oh * ow) + p].clamp(0.0, 1.0);
                let got = outs[i][p * 4 + c] as f32 / 255.0;
                let d = (got - refv).abs();
                max = max.max(d);
                sum += d as f64;
                se += (d * d) as f64;
            }
        }
    }
    let n = (t * oh * ow * 3) as f64;
    let psnr = 10.0 * (1.0 / (se / n)).log10();
    println!("{}: {t} кадров {wd}x{h} → {ow}x{oh}: max |Δ| {:.4} ({:.1}/255), mean {:.5}, PSNR к эталону {psnr:.1} дБ", up.name(), max, max * 255.0, sum / n);
}
