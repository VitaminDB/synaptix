use half::f16;
use synaptix_npu::{Binding, DType, Graph};

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let (m, k, n) = (*args.first().unwrap_or(&64), *args.get(1).unwrap_or(&1024), *args.get(2).unwrap_or(&1024));
    let iters = *args.get(3).unwrap_or(&50);

    let npu = synaptix_npu::open()?;
    let info = npu.info();
    println!("NPU: {} ({}), SoC {}, VTCM {} МБ, рантайм {}", info.model.name, info.model.id, info.soc, info.vtcm_mb, info.runtime);

    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let x: Vec<f16> = (0..m * k).map(|_| f16::from_f32(rnd(&mut seed))).collect();
    let w: Vec<f16> = (0..n * k).map(|_| f16::from_f32(rnd(&mut seed) / (k as f32).sqrt())).collect();
    let b: Vec<f16> = (0..n).map(|_| f16::from_f32(rnd(&mut seed) * 0.1)).collect();

    let mut g = Graph::new();
    let xi = g.input(DType::F16, &[m as u32, k as u32]);
    let dyn_w = std::env::var("NPU_DYN_W").is_ok();
    let wi = if dyn_w { g.input(DType::F16, &[n as u32, k as u32]) } else { g.weight(DType::F16, &[n as u32, k as u32], bytemuck::cast_slice(&w))? };
    let bi = g.weight(DType::F16, &[n as u32], bytemuck::cast_slice(&b))?;
    let y = g.linear(xi, wi, Some(bi))?;
    g.output(y)?;

    let t0 = std::time::Instant::now();
    let cg = npu.compile(&format!("npu_linear_{m}x{k}x{n}_{dyn_w}"), &g)?;
    println!(
        "граф: {} за {:.1} мс (сборка на NPU {} мс)",
        if cg.from_cache() { "из кэша" } else { "собран" },
        t0.elapsed().as_secs_f64() * 1e3,
        cg.build_time().as_millis()
    );

    let mut xb = npu.alloc(m * k * 2)?;
    let yb = npu.alloc(m * n * 2)?;
    xb.cpu_begin();
    xb.typed_mut::<f16>(0, m * k).copy_from_slice(&x);
    xb.cpu_end();
    let mut wb = npu.alloc(n * k * 2)?;
    wb.typed_mut::<f16>(0, n * k).copy_from_slice(&w);
    let mut bind = vec![
        Binding { tensor: xi, buffer: &xb, offset: 0 },
        Binding { tensor: y, buffer: &yb, offset: 0 },
    ];
    if dyn_w {
        bind.push(Binding { tensor: wi, buffer: &wb, offset: 0 });
    }
    let mut npu_us = Vec::new();
    let mut wall = Vec::new();
    for _ in 0..iters.max(1) {
        let t = std::time::Instant::now();
        npu_us.push(cg.run(&bind)?.as_secs_f64() * 1e3);
        wall.push(t.elapsed().as_secs_f64() * 1e3);
    }
    yb.cpu_begin();
    let out: Vec<f32> = yb.typed::<f16>(0, m * n).iter().map(|v| v.to_f32()).collect();
    yb.cpu_end();

    let tc = std::time::Instant::now();
    let mut worst_cos = 1.0f64;
    let mut max_err = 0.0f32;
    for i in 0..m {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for j in 0..n {
            let mut s = b[j].to_f32();
            for t in 0..k {
                s += x[i * k + t].to_f32() * w[j * k + t].to_f32();
            }
            let o = out[i * n + j];
            max_err = max_err.max((o - s).abs());
            dot += o as f64 * s as f64;
            na += (o as f64).powi(2);
            nb += (s as f64).powi(2);
        }
        worst_cos = worst_cos.min(dot / (na.sqrt() * nb.sqrt()).max(1e-30));
    }
    let cpu_ms = tc.elapsed().as_secs_f64() * 1e3;

    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (e, wl) = (med(&mut npu_us), med(&mut wall));
    let flops = 2.0 * (m * n * k) as f64;
    println!("NPU: исполнение {e:.3} мс (с RPC демона {wl:.3} мс), {:.1} GFLOP/s", flops / (wl * 1e6));
    println!("CPU-эталон (f32, 1 поток): {cpu_ms:.1} мс");
    println!("сверка по строкам: худший cos {worst_cos:.6}, макс. |Δ| {max_err:.4}");
    if worst_cos < 0.999 {
        return Err("расхождение с CPU".into());
    }
    Ok(())
}
