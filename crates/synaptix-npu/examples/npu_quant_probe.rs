use synaptix_npu::graph::{Op, Quant};
use synaptix_npu::{DType, Graph, Npu, TensorId};

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn newest_size(dir: &str) -> u64 {
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok().map(|m| (m.modified().ok(), m.len())))
        .max_by_key(|(t, _)| *t)
        .map(|(_, l)| l)
        .unwrap_or(0)
}

fn try_graph(npu: &dyn Npu, name: &str, build: impl FnOnce(&mut Graph) -> TensorId) {
    let mut g = Graph::new();
    let y = build(&mut g);
    g.output(y).unwrap();
    let t0 = std::time::Instant::now();
    match npu.compile(&format!("qprobe_{name}"), &g) {
        Ok(_) => {
            let cache = std::env::var("SYNNPU_CACHE_DIR").unwrap_or_default();
            println!("ok    {name}: {:.1} с, контекст {} КБ", t0.elapsed().as_secs_f64(), newest_size(&cache) / 1024);
        }
        Err(e) => println!("FAIL  {name}: {e}"),
    }
}

fn weights(n: usize, k: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<(f32, i32)>, Vec<u8>) {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let w: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed) / (k as f32).sqrt()).collect();
    let f16: Vec<u8> = w.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
    let mut i8 = vec![0u8; n * k];
    let mut i4 = vec![0u8; n * k / 2];
    let mut scales = Vec::with_capacity(n);
    let mut scales4 = Vec::with_capacity(n);
    for r in 0..n {
        let row = &w[r * k..(r + 1) * k];
        let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let s = amax / 127.0;
        let s4 = amax / 7.0;
        scales.push((s, 0i32));
        scales4.push((s4, 0i32));
        for (i, &v) in row.iter().enumerate() {
            i8[r * k + i] = (v / s).round().clamp(-127.0, 127.0) as i8 as u8;
            let q4 = (v / s4).round().clamp(-8.0, 7.0) as i8 as u8 & 0x0f;
            let idx = r * k + i;
            if idx % 2 == 0 {
                i4[idx / 2] |= q4;
            } else {
                i4[idx / 2] |= q4 << 4;
            }
        }
    }
    let _ = scales4;
    (f16, i8, i4, scales, Vec::new())
}

fn main() {
    let npu = synaptix_npu::open().unwrap();
    let n = npu.as_ref();
    let (m, k, nn): (usize, usize, usize) = (
        std::env::var("M").ok().and_then(|v| v.parse().ok()).unwrap_or(64),
        std::env::var("K").ok().and_then(|v| v.parse().ok()).unwrap_or(2048),
        std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(2048),
    );
    let (f16, i8, i4, scales, _) = weights(nn, k);
    let shape = [nn as u32, k as u32];
    let only = std::env::args().nth(1);
    let want = |name: &str| only.as_deref().map_or(true, |o| o == name);

    if want("fc_f16") {
        try_graph(n, "fc_f16", |g| {
            let x = g.input(DType::F16, &[m as u32, k as u32]);
            let w = g.weight(DType::F16, &shape, &f16).unwrap();
            g.linear(x, w, None).unwrap()
        });
    }
    if want("fc_i8_axis") {
        try_graph(n, "fc_i8_axis", |g| {
            let x = g.input(DType::F16, &[m as u32, k as u32]);
            let w = g.weight_quant(DType::I8, &shape, Quant::Axis { axis: 0, scales: scales.clone() }, &i8).unwrap();
            g.linear(x, w, None).unwrap()
        });
    }
    if want("fc_i8_tensor") {
        try_graph(n, "fc_i8_tensor", |g| {
            let x = g.input(DType::F16, &[m as u32, k as u32]);
            let w = g.weight_quant(DType::I8, &shape, Quant::Tensor { scale: scales[0].0, offset: 0 }, &i8).unwrap();
            g.linear(x, w, None).unwrap()
        });
    }
    if want("fc_i8_deq") {
        try_graph(n, "fc_i8_deq", |g| {
            let x = g.input(DType::F16, &[m as u32, k as u32]);
            let wq = g.weight_quant(DType::I8, &shape, Quant::Axis { axis: 0, scales: scales.clone() }, &i8).unwrap();
            let w = g.node(DType::F16, &shape, |out| Op::Dequantize { x: wq, out });
            g.linear(x, w, None).unwrap()
        });
    }
    if want("fc_i4_axis") {
        try_graph(n, "fc_i4_axis", |g| {
            let x = g.input(DType::F16, &[m as u32, k as u32]);
            let w = g.weight_quant(DType::I4, &shape, Quant::Axis { axis: 0, scales: scales.clone() }, &i4).unwrap();
            g.linear(x, w, None).unwrap()
        });
    }
    if want("fc_i4_deq") {
        try_graph(n, "fc_i4_deq", |g| {
            let x = g.input(DType::F16, &[m as u32, k as u32]);
            let wq = g.weight_quant(DType::I4, &shape, Quant::Axis { axis: 0, scales: scales.clone() }, &i4).unwrap();
            let w = g.node(DType::F16, &shape, |out| Op::Dequantize { x: wq, out });
            g.linear(x, w, None).unwrap()
        });
    }
    if want("mm_i8_deq") {
        try_graph(n, "mm_i8_deq", |g| {
            let x = g.input(DType::F16, &[m as u32, k as u32]);
            let wq = g.weight_quant(DType::I8, &shape, Quant::Axis { axis: 0, scales: scales.clone() }, &i8).unwrap();
            let w = g.node(DType::F16, &shape, |out| Op::Dequantize { x: wq, out });
            g.node(DType::F16, &[m as u32, nn as u32], |out| Op::MatMul { a: x, b: w, transpose_a: false, transpose_b: true, out })
        });
    }
}
