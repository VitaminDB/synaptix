//! Проверка режимов свёртки `conv_mma` против прямой свёртки на процессоре (f32): размеры ядра,
//! конкатенация входов, ширина блока, активации, остаток, внимание SPAB, ворота, pixel shuffle, срез
//! каналов. Запуск: `cargo run -p synaptix-vfx --example conv_test`.

use synaptix_vfx::gpu::{pad16, Act, Conv, Gpu, In, Out, Res, Src};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// NHWC [h,w,c] f32 → буфер с шагом cs (f16 на GPU).
fn to_gpu(g: &Gpu, x: &[f32], px: usize, c: usize, cs: usize) -> cudarc::driver::CudaSlice<half::f16> {
    let mut v = vec![0f32; px * cs];
    for p in 0..px {
        v[p * cs..p * cs + c].copy_from_slice(&x[p * c..p * c + c]);
    }
    g.upload_f16(&v).unwrap()
}

fn from_gpu(g: &Gpu, b: &cudarc::driver::CudaSlice<half::f16>, px: usize, c: usize, cs: usize) -> Vec<f32> {
    let mut h = vec![half::f16::ZERO; px * cs];
    g.stream.memcpy_dtoh(b, &mut h).unwrap();
    g.sync().unwrap();
    let mut out = vec![0f32; px * c];
    for p in 0..px {
        for k in 0..c {
            out[p * c + k] = h[p * cs + k].to_f32();
        }
    }
    out
}

/// Прямая свёртка NHWC: x [h,w,cin], weight [cout,cin,k,k] → [h,w,cout].
fn conv_ref(x: &[f32], h: usize, w: usize, cin: usize, wt: &[f32], b: &[f32], cout: usize, k: usize) -> Vec<f32> {
    let p = (k / 2) as isize;
    let mut out = vec![0f32; h * w * cout];
    for y in 0..h {
        for xx in 0..w {
            for o in 0..cout {
                let mut s = b[o];
                for dy in 0..k {
                    for dx in 0..k {
                        let (sy, sx) = (y as isize + dy as isize - p, xx as isize + dx as isize - p);
                        if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                            continue;
                        }
                        let base = (sy as usize * w + sx as usize) * cin;
                        for i in 0..cin {
                            s += x[base + i] * wt[((o * cin + i) * k + dy) * k + dx];
                        }
                    }
                }
                out[(y * w + xx) * cout + o] = s;
            }
        }
    }
    out
}

fn report(name: &str, got: &[f32], want: &[f32]) -> bool {
    let mut max = 0f32;
    let mut scale = 0f32;
    for (a, b) in got.iter().zip(want) {
        max = max.max((a - b).abs());
        scale = scale.max(b.abs());
    }
    let ok = max <= 0.02 * scale.max(1.0);
    println!("{} {name}: max |Δ| {max:.4} (масштаб {scale:.2})", if ok { "ok  " } else { "FAIL" });
    ok
}

fn main() {
    let g = Gpu::new(0).unwrap();
    let mut r = Rng(7);
    let (h, w) = (9, 70);
    let px = h * w;
    let mut all = true;
    for &(k, cin, cout) in &[(3usize, 48usize, 48usize), (1, 48, 48), (3, 16, 12), (3, 32, 32), (3, 64, 128), (5, 16, 16), (17, 16, 16)] {
        let x: Vec<f32> = (0..px * cin).map(|_| r.f()).collect();
        let wt: Vec<f32> = (0..cout * cin * k * k).map(|_| r.f() * 0.1).collect();
        let b: Vec<f32> = (0..cout).map(|_| r.f()).collect();
        let c = Conv::new(&g, &wt, Some(&b), cout, k, &[cin], 1).unwrap();
        let xg = to_gpu(&g, &x, px, cin, pad16(cin));
        let mut og = g.alloc_f16(px * pad16(cout)).unwrap();
        g.run(&c, In::one(&xg), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut og, cs: pad16(cout) }).unwrap();
        all &= report(&format!("{k}×{k} {cin}→{cout}"), &from_gpu(&g, &og, px, cout, pad16(cout)), &conv_ref(&x, h, w, cin, &wt, &b, cout, k));
    }
    // конкатенация 4 входов, 1×1
    {
        let (c1, cout) = (48, 48);
        let xs: Vec<Vec<f32>> = (0..4).map(|_| (0..px * c1).map(|_| r.f()).collect()).collect();
        let wt: Vec<f32> = (0..cout * 4 * c1).map(|_| r.f() * 0.1).collect();
        let b: Vec<f32> = (0..cout).map(|_| r.f()).collect();
        let c = Conv::new(&g, &wt, Some(&b), cout, 1, &[c1; 4], 1).unwrap();
        let gs: Vec<_> = xs.iter().map(|x| to_gpu(&g, x, px, c1, c1)).collect();
        let mut og = g.alloc_f16(px * cout).unwrap();
        g.run(&c, In::cat(&[&gs[0], &gs[1], &gs[2], &gs[3]]), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut og, cs: cout }).unwrap();
        let mut cat = vec![0f32; px * 4 * c1];
        for p in 0..px {
            for (i, x) in xs.iter().enumerate() {
                cat[p * 4 * c1 + i * c1..p * 4 * c1 + (i + 1) * c1].copy_from_slice(&x[p * c1..(p + 1) * c1]);
            }
        }
        all &= report("cat4 1×1 192→48", &from_gpu(&g, &og, px, cout, cout), &conv_ref(&cat, h, w, 4 * c1, &wt, &b, cout, 1));
    }
    // внимание SPAB и ворота
    for (act, name) in [(Act::SpabAttention, "SPAB"), (Act::Gate, "ворота"), (Act::Silu, "SiLU"), (Act::Mish, "Mish")] {
        let cn = 48;
        let x: Vec<f32> = (0..px * cn).map(|_| r.f()).collect();
        let wt: Vec<f32> = (0..cn * cn * 9).map(|_| r.f() * 0.1).collect();
        let b: Vec<f32> = (0..cn).map(|_| r.f()).collect();
        let c = Conv::new(&g, &wt, Some(&b), cn, 3, &[cn], 1).unwrap();
        let xg = to_gpu(&g, &x, px, cn, cn);
        let mut og = g.alloc_f16(px * cn).unwrap();
        let res = if matches!(act, Act::SpabAttention | Act::Gate) { Res::Before(&xg) } else { Res::None };
        g.run(&c, In::one(&xg), h, w, act, res, 1, Out::F16 { buf: &mut og, cs: cn }).unwrap();
        let v = conv_ref(&x, h, w, cn, &wt, &b, cn, 3);
        let sig = |z: f32| 1.0 / (1.0 + (-z).exp());
        let want: Vec<f32> = v
            .iter()
            .zip(&x)
            .map(|(&z, &xi)| match act {
                Act::SpabAttention => (z + xi) * (sig(z) - 0.5),
                Act::Gate => xi * sig(z),
                Act::Silu => z * sig(z),
                _ => z * ((1.0 + z.exp()).ln()).tanh(),
            })
            .collect();
        all &= report(name, &from_gpu(&g, &og, px, cn, cn), &want);
    }
    // pixel shuffle ×2 в RGBA
    {
        let (cn, cout, r2) = (48, 12, 2);
        let x: Vec<f32> = (0..px * cn).map(|_| r.f()).collect();
        let wt: Vec<f32> = (0..cout * cn * 9).map(|_| r.f() * 0.05).collect();
        let b: Vec<f32> = (0..cout).map(|_| 0.5).collect();
        let c = Conv::new(&g, &wt, Some(&b), cout, 3, &[cn], r2).unwrap();
        let xg = to_gpu(&g, &x, px, cn, cn);
        let mut og = g.alloc_u8(px * 4 * 4).unwrap();
        g.run(&c, In::one(&xg), h, w, Act::None, Res::None, r2, Out::Rgba { buf: &mut og, base: None, nearest: false }).unwrap();
        let mut host = vec![0u8; px * 16];
        g.stream.memcpy_dtoh(&og, &mut host).unwrap();
        g.sync().unwrap();
        let v = conv_ref(&x, h, w, cn, &wt, &b, cout, 3);
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for y in 0..h {
            for xx in 0..w {
                for n in 0..cout {
                    let (ch, sub) = (n / 4, n % 4);
                    let (yy, xo) = (y * 2 + sub / 2, xx * 2 + sub % 2);
                    got.push(host[(yy * w * 2 + xo) * 4 + ch] as f32 / 255.0);
                    want.push(v[(y * w + xx) * cout + n].clamp(0.0, 1.0));
                }
            }
        }
        all &= report("shuffle×2 → RGBA", &got, &want);
    }
    // срез каналов: свёртка по первым 16 из 64
    {
        let (cs, pd) = (64, 16);
        let x: Vec<f32> = (0..px * cs).map(|_| r.f()).collect();
        let wt: Vec<f32> = (0..pd * pd * 9).map(|_| r.f() * 0.1).collect();
        let b: Vec<f32> = (0..pd).map(|_| r.f()).collect();
        let c = Conv::new(&g, &wt, Some(&b), pd, 3, &[pd], 1).unwrap();
        let xg = to_gpu(&g, &x, px, cs, cs);
        let mut og = g.alloc_f16(px * cs).unwrap();
        g.run(&c, In::srcs(vec![Src { buf: &xg, off: 0, ps: cs }]), h, w, Act::None, Res::None, 1, Out::F16 { buf: &mut og, cs }).unwrap();
        let mut xs = vec![0f32; px * pd];
        for p in 0..px {
            xs[p * pd..(p + 1) * pd].copy_from_slice(&x[p * cs..p * cs + pd]);
        }
        let got = from_gpu(&g, &og, px, cs, cs);
        let gotp: Vec<f32> = (0..px).flat_map(|p| got[p * cs..p * cs + pd].to_vec()).collect();
        all &= report("срез 16 из 64", &gotp, &conv_ref(&xs, h, w, pd, &wt, &b, pd, 3));
    }
    println!("{}", if all { "все режимы совпали" } else { "ЕСТЬ РАСХОЖДЕНИЯ" });
    std::process::exit(i32::from(!all));
}
