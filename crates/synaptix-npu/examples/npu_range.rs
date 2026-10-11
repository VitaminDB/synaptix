use half::f16;
use synaptix_npu::graph::UnaryKind;
use synaptix_npu::{Binding, DType, Graph};

fn gelu(x: f32) -> f32 {
    let t = x as f64 / std::f64::consts::SQRT_2;
    let erf = {
        let s = t.signum();
        let a = t.abs();
        let k = 1.0 / (1.0 + 0.3275911 * a);
        let y = 1.0 - (((((1.061405429 * k - 1.453152027) * k) + 1.421413741) * k - 0.284496736) * k + 0.254829592) * k * (-a * a).exp();
        s * y
    };
    (0.5 * x as f64 * (1.0 + erf)) as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let npu = synaptix_npu::open()?;
    let n = 4096usize;
    let xs: Vec<f32> = (0..n).map(|i| -30.0 + 330.0 * i as f32 / (n - 1) as f32).collect();
    for kind in [UnaryKind::Gelu, UnaryKind::Sigmoid, UnaryKind::Tanh] {
        let mut g = Graph::new();
        let x = g.input(DType::F16, &[1, 1, n as u32]);
        let y = g.unary(kind, x);
        g.output(y)?;
        let cg = npu.compile(&format!("range_{kind:?}"), &g)?;
        let mut xb = npu.alloc(n * 2)?;
        let yb = npu.alloc(n * 2)?;
        for (d, v) in xb.typed_mut::<f16>(0, n).iter_mut().zip(&xs) {
            *d = f16::from_f32(*v);
        }
        cg.run(&[Binding { tensor: x, buffer: &xb, offset: 0 }, Binding { tensor: y, buffer: &yb, offset: 0 }])?;
        let out = yb.typed::<f16>(0, n);
        let mut worst = (0f32, 0f32, 0f32, 0f32);
        for (i, &xv) in xs.iter().enumerate() {
            let xq = f16::from_f32(xv).to_f32();
            let r = match kind {
                UnaryKind::Gelu => gelu(xq),
                UnaryKind::Sigmoid => 1.0 / (1.0 + (-xq).exp()),
                _ => xq.tanh(),
            };
            let e = (out[i].to_f32() - r).abs() / r.abs().max(1.0);
            if e > worst.0 {
                worst = (e, xq, r, out[i].to_f32());
            }
        }
        println!("{kind:?}: худшая отн. ошибка {:.4} при x={:.2}: эталон {:.3}, NPU {:.3}", worst.0, worst.1, worst.2, worst.3);
    }
    for target in [100.0f32, 240.0, 260.0, 300.0, 1000.0, 5000.0] {
        let (k, nn) = (5120usize, 64usize);
        let mut g = Graph::new();
        let x = g.input(DType::F16, &[1, 4, k as u32]);
        let wv = target / (k as f32 * 0.1);
        let w: Vec<f16> = (0..nn * k).map(|i| f16::from_f32(if i / k == 0 { wv } else { wv * 0.5 })).collect();
        let wi = g.weight(DType::F16, &[nn as u32, k as u32], bytemuck::cast_slice(&w))?;
        let y = g.linear(x, wi, None)?;
        g.output(y)?;
        let cg = npu.compile(&format!("fc_range_{target}"), &g)?;
        let mut xb = npu.alloc(4 * k * 2)?;
        let yb = npu.alloc(4 * nn * 2)?;
        for d in xb.typed_mut::<f16>(0, 4 * k).iter_mut() {
            *d = f16::from_f32(0.1);
        }
        cg.run(&[Binding { tensor: x, buffer: &xb, offset: 0 }, Binding { tensor: y, buffer: &yb, offset: 0 }])?;
        let out = yb.typed::<f16>(0, 4 * nn);
        println!("FC: цель {target}: NPU {:.2} / {:.2} (половина)", out[0].to_f32(), out[1].to_f32());
    }
    Ok(())
}
