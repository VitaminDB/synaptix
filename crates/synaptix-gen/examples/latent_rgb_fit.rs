//! Подбор линейной проекции латента в RGB (превью генерации) методом
//! наименьших квадратов. Входы — пары «латент — ролик»:
//! `latent_rgb_fit OUT.f32 C ВРЕМ_ШАГ LAT1.FxHxW.f32 VID1.mp4 [LAT2 VID2 …]`
//! Латент — сырой f32 `[C, F, H, W]` (имя оканчивается на `FxHxW.f32`);
//! латентному кадру k соответствует кадр ролика `max(0, ВРЕМ_ШАГ·k − ВРЕМ_ШАГ/2)`
//! (у LTX шаг 8), уменьшенный до H×W. Выход — `3·(C+1)` чисел f32: веса и смещение.
use std::process::Command;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (out, c, ts): (&str, usize, usize) = (&a[0], a[1].parse().unwrap(), a[2].parse().unwrap());
    let n = c + 1;
    let mut ata = vec![0f64; n * n];
    let mut atb = vec![0f64; 3 * n];
    let mut samples = 0usize;
    for pair in a[3..].chunks(2) {
        let (lp, vp) = (&pair[0], &pair[1]);
        let dims: Vec<usize> = lp.trim_end_matches(".f32").rsplit('.').next().unwrap().split('x').map(|x| x.parse().unwrap()).collect();
        let (f, h, w) = (dims[0], dims[1], dims[2]);
        let lat: Vec<f32> = std::fs::read(lp).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(lat.len(), c * f * h * w, "{lp}");
        for k in 0..f {
            let frame = (ts * k).saturating_sub(ts / 2);
            let o = Command::new("ffmpeg")
                .args(["-v", "error", "-i", vp, "-vf", &format!("select=eq(n\\,{frame}),scale={w}:{h}:flags=area"), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                .output()
                .unwrap();
            if o.stdout.len() != h * w * 3 {
                continue;
            }
            for i in 0..h * w {
                let x: Vec<f64> = (0..c).map(|ch| lat[((ch * f + k) * h * w) + i] as f64).chain(std::iter::once(1.0)).collect();
                for r in 0..n {
                    for q in r..n {
                        ata[r * n + q] += x[r] * x[q];
                    }
                    for ch in 0..3 {
                        atb[ch * n + r] += x[r] * o.stdout[i * 3 + ch] as f64 / 255.0;
                    }
                }
                samples += 1;
            }
        }
    }
    for r in 0..n {
        for q in 0..r {
            ata[r * n + q] = ata[q * n + r];
        }
        ata[r * n + r] += 1e-3 * samples as f64 / n as f64; // лёгкая регуляризация
    }
    // Гаусс с выбором главного элемента, три правые части.
    let mut m = ata;
    let mut b = atb;
    for col in 0..n {
        let piv = (col..n).max_by(|&x, &y| m[x * n + col].abs().partial_cmp(&m[y * n + col].abs()).unwrap()).unwrap();
        for j in 0..n {
            m.swap(col * n + j, piv * n + j);
        }
        for ch in 0..3 {
            b.swap(ch * n + col, ch * n + piv);
        }
        let d = m[col * n + col];
        for r in col + 1..n {
            let k = m[r * n + col] / d;
            if k == 0.0 {
                continue;
            }
            for j in col..n {
                m[r * n + j] -= k * m[col * n + j];
            }
            for ch in 0..3 {
                b[ch * n + r] -= k * b[ch * n + col];
            }
        }
    }
    let mut sol = vec![0f32; 3 * n];
    for ch in 0..3 {
        for r in (0..n).rev() {
            let mut v = b[ch * n + r];
            for j in r + 1..n {
                v -= m[r * n + j] * sol[ch * n + j] as f64;
            }
            sol[ch * n + r] = (v / m[r * n + r]) as f32;
        }
    }
    std::fs::write(out, sol.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    println!("образцов {samples}, записано {out}");
}
