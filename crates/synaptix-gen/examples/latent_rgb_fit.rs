//! Подбор линейной проекции латента в RGB (превью генерации) методом
//! наименьших квадратов. Входы — пары «латент — ролик/картинка»:
//! `latent_rgb_fit [--scale S] [--kernel K] [--reg R] OUT.f32 C ВРЕМ_ШАГ LAT1.FxHxW.f32 VID1 [LAT2 VID2 …]`
//! Латент — сырой f32 `[C, F, H, W]` (имя оканчивается на `FxHxW.f32`);
//! `ВРЕМ_ШАГ` — число (у LTX 8, у картинок 1) или `h3` (точная сетка клипов
//! MiniMax-H3: 5 латентов = 17 кадров).
//! `--scale S` (по умолчанию 1): блок `S×S` пикселей на токен.
//! `--kernel K` (`1` или `3`, по умолчанию 1): пространственное окно токенов
//! (`3` — свёртка 3×3 + PixelShuffle S×S, без сеточных швов на стыках токенов).
use std::process::Command;

enum TimeMap {
    Fixed(usize),
    H3,
}

impl TimeMap {
    fn parse(s: &str) -> Self {
        if s.eq_ignore_ascii_case("h3") {
            Self::H3
        } else {
            Self::Fixed(s.parse().expect("ВРЕМ_ШАГ: число или 'h3'"))
        }
    }

    fn frame(&self, k: usize) -> usize {
        match *self {
            Self::Fixed(ts) => (ts * k).saturating_sub(ts / 2),
            Self::H3 => {
                let m = k / 5;
                let r = k % 5;
                if r == 0 { 17 * m } else { 17 * m + 4 * r - 2 }
            }
        }
    }
}

fn load_pair(lp: &str, c: usize) -> (usize, usize, usize, Vec<f32>) {
    let dims: Vec<usize> = lp
        .trim_end_matches(".f32")
        .rsplit('.')
        .next()
        .unwrap()
        .split('x')
        .map(|x| x.parse().unwrap())
        .collect();
    let (f, h, w) = (dims[0], dims[1], dims[2]);
    let lat: Vec<f32> = std::fs::read(lp)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(lat.len(), c * f * h * w, "{lp}");
    (f, h, w, lat)
}

fn decode_frame(vp: &str, frame: usize, out_w: usize, out_h: usize) -> Vec<u8> {
    let o = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            vp,
            "-vf",
            &format!("select=eq(n\\,{frame}),scale={out_w}:{out_h}:flags=area"),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .unwrap();
    o.stdout
}

fn fill_feat(lat: &[f32], c: usize, f: usize, h: usize, w: usize, k: usize, ty: usize, tx: usize, kernel: usize, out: &mut [f64]) {
    let hw = h * w;
    if kernel == 1 {
        let ti = ty * w + tx;
        for ch in 0..c {
            out[ch] = lat[(ch * f + k) * hw + ti] as f64;
        }
        out[c] = 1.0;
    } else {
        for ky in 0..3 {
            let ny = (ty + ky).saturating_sub(1).min(h - 1);
            for kx in 0..3 {
                let nx = (tx + kx).saturating_sub(1).min(w - 1);
                let ti = ny * w + nx;
                let base = (ky * 3 + kx) * c;
                for ch in 0..c {
                    out[base + ch] = lat[(ch * f + k) * hw + ti] as f64;
                }
            }
        }
        out[9 * c] = 1.0;
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut scale: usize = std::env::var("SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut kernel: usize = std::env::var("KERNEL").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut reg: f64 = 1e-2;
    let mut pos = Vec::new();
    let mut idx = 0;
    while idx < raw.len() {
        match raw[idx].as_str() {
            "--scale" => {
                scale = raw[idx + 1].parse().expect("--scale N");
                idx += 2;
            }
            "--kernel" => {
                kernel = raw[idx + 1].parse().expect("--kernel 1|3");
                idx += 2;
            }
            "--reg" => {
                reg = raw[idx + 1].parse().expect("--reg R");
                idx += 2;
            }
            _ => {
                pos.push(raw[idx].as_str());
                idx += 1;
            }
        }
    }
    assert!(
        (kernel == 1 || kernel == 3) && scale >= 1 && pos.len() >= 5 && (pos.len() - 3) % 2 == 0,
        "использование: latent_rgb_fit [--scale S] [--kernel 1|3] [--reg R] OUT.f32 C ВРЕМ_ШАГ LAT1 VID1 ..."
    );
    let out = pos[0];
    let c: usize = pos[1].parse().unwrap();
    let tmap = TimeMap::parse(pos[2]);
    let in_dim = kernel * kernel * c;
    let n = in_dim + 1;
    let m = 3 * scale * scale;
    let mut ata = vec![0f64; n * n];
    let mut atb = vec![0f64; m * n];
    let mut samples = 0usize;
    let mut x = vec![0f64; n];

    let mut cached: Vec<(usize, usize, usize, Vec<f32>, Vec<(usize, Vec<u8>)>)> = Vec::new();

    for pair in pos[3..].chunks(2) {
        let (lp, vp) = (pair[0], pair[1]);
        let (f, h, w, lat) = load_pair(lp, c);
        let (out_w, out_h) = (w * scale, h * scale);
        let mut frames = Vec::new();
        for k in 0..f {
            let frame = tmap.frame(k);
            let rgb = decode_frame(vp, frame, out_w, out_h);
            if rgb.len() != out_h * out_w * 3 {
                continue;
            }
            for ty in 0..h {
                for tx in 0..w {
                    fill_feat(&lat, c, f, h, w, k, ty, tx, kernel, &mut x);
                    for r in 0..n {
                        let xr = x[r];
                        let row = &mut ata[r * n..r * n + n];
                        for q in r..n {
                            row[q] += xr * x[q];
                        }
                    }
                    for dy in 0..scale {
                        let py = ty * scale + dy;
                        for dx in 0..scale {
                            let px = tx * scale + dx;
                            let off = (py * out_w + px) * 3;
                            let base = (dy * scale + dx) * 3;
                            for ch in 0..3 {
                                let y = rgb[off + ch] as f64 / 255.0;
                                let row = &mut atb[(base + ch) * n..(base + ch + 1) * n];
                                for r in 0..n {
                                    row[r] += x[r] * y;
                                }
                            }
                        }
                    }
                    samples += 1;
                }
            }
            frames.push((k, rgb));
        }
        cached.push((f, h, w, lat, frames));
    }
    assert!(samples > 0, "нет ни одного образца");
    for r in 0..n {
        for q in 0..r {
            ata[r * n + q] = ata[q * n + r];
        }
        if r < in_dim {
            // Не регуляризуем свободный член (смещение).
            ata[r * n + r] += reg * samples as f64 / n as f64;
        }
    }
    // Гаусс с выбором главного элемента, `m` правых частей.
    let mut mat = ata;
    let mut b = atb;
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&a, &b_| mat[a * n + col].abs().partial_cmp(&mat[b_ * n + col].abs()).unwrap())
            .unwrap();
        for j in 0..n {
            mat.swap(col * n + j, piv * n + j);
        }
        for o in 0..m {
            b.swap(o * n + col, o * n + piv);
        }
        let d = mat[col * n + col];
        for r in col + 1..n {
            let k = mat[r * n + col] / d;
            if k == 0.0 {
                continue;
            }
            for j in col..n {
                mat[r * n + j] -= k * mat[col * n + j];
            }
            for o in 0..m {
                b[o * n + r] -= k * b[o * n + col];
            }
        }
    }
    let mut sol = vec![0f32; m * n];
    for o in 0..m {
        for r in (0..n).rev() {
            let mut v = b[o * n + r];
            for j in r + 1..n {
                v -= mat[r * n + j] * sol[o * n + j] as f64;
            }
            sol[o * n + r] = (v / mat[r * n + r]) as f32;
        }
    }
    std::fs::write(out, sol.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();

    // Оценка качества реконструкции (RMSE / PSNR).
    let (mut sq_all, mut cnt_all) = (0f64, 0usize);
    let (mut sq_r0, mut cnt_r0) = (0f64, 0usize);
    let (mut sq_rn, mut cnt_rn) = (0f64, 0usize);
    for (f, h, w, lat, frames) in &cached {
        let out_w = w * scale;
        for (k, rgb) in frames {
            for ty in 0..*h {
                for tx in 0..*w {
                    fill_feat(lat, c, *f, *h, *w, *k, ty, tx, kernel, &mut x);
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let off = ((ty * scale + dy) * out_w + tx * scale + dx) * 3;
                            let base = (dy * scale + dx) * 3;
                            for ch in 0..3 {
                                let wrow = &sol[(base + ch) * n..(base + ch + 1) * n];
                                let mut pred = wrow[in_dim] as f64;
                                for lc in 0..in_dim {
                                    pred += wrow[lc] as f64 * x[lc];
                                }
                                let err = pred.clamp(0.0, 1.0) - rgb[off + ch] as f64 / 255.0;
                                sq_all += err * err;
                                cnt_all += 1;
                                if k % 5 == 0 {
                                    sq_r0 += err * err;
                                    cnt_r0 += 1;
                                } else {
                                    sq_rn += err * err;
                                    cnt_rn += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let rmse = (sq_all / cnt_all as f64).sqrt();
    let psnr = -20.0 * rmse.max(1e-9).log10();
    println!("образцов {samples} (окно {kernel}×{kernel}, блок {scale}×{scale}), RMSE {rmse:.4}, PSNR {psnr:.2} дБ, записано {out}");
    if matches!(tmap, TimeMap::H3) && cnt_r0 > 0 && cnt_rn > 0 {
        let p0 = -20.0 * (sq_r0 / cnt_r0 as f64).sqrt().max(1e-9).log10();
        let pn = -20.0 * (sq_rn / cnt_rn as f64).sqrt().max(1e-9).log10();
        println!("  H3 по фазам клипа: r==0 PSNR {p0:.2} дБ, r>0 PSNR {pn:.2} дБ");
    }
}
