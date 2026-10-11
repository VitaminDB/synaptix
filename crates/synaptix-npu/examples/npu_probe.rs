use synaptix_npu::graph::{Op, ReduceKind, UnaryKind};
use synaptix_npu::{DType, Graph, Npu, TensorId};

fn try_graph(npu: &dyn Npu, name: &str, build: impl FnOnce(&mut Graph) -> TensorId) {
    let mut g = Graph::new();
    let y = build(&mut g);
    g.output(y).unwrap();
    match npu.compile(&format!("probe_{name}"), &g) {
        Ok(_) => println!("ok    {name}"),
        Err(e) => println!("FAIL  {name}: {e}"),
    }
}

fn reduce(g: &mut Graph, dt: DType, shape: &[u32], axis: u32, kind: ReduceKind, keep: bool) -> TensorId {
    let x = g.input(dt, shape);
    let mut os = shape.to_vec();
    if keep {
        os[axis as usize] = 1;
    } else {
        os.remove(axis as usize);
    }
    g.node(dt, &os, |out| Op::Reduce { kind, x, axes: vec![axis], keep_dims: keep, out })
}

fn slice_then(g: &mut Graph, dt: DType, trailing: bool, ln: bool) -> TensorId {
    let x = g.input(dt, &[1, 100, 256]);
    let s = if trailing {
        let r = g.reshape(x, &[100, 256, 1]).unwrap();
        let sl = g.node(dt, &[98, 256, 1], |out| Op::Slice { x: r, begin: vec![1, 0, 0], end: vec![99, 256, 1], stride: vec![1, 1, 1], out });
        g.reshape(sl, &[1, 98, 256]).unwrap()
    } else {
        g.node(dt, &[1, 98, 256], |out| Op::Slice { x, begin: vec![0, 1, 0], end: vec![1, 99, 256], stride: vec![1, 1, 1], out })
    };
    if ln {
        let gm = g.weight(dt, &[256], &vec![0u8; dt.bytes(256)]).unwrap();
        let bt = g.weight(dt, &[256], &vec![0u8; dt.bytes(256)]).unwrap();
        g.node(dt, &[1, 98, 256], |out| Op::LayerNorm { x: s, gamma: gm, beta: Some(bt), eps: 1e-5, axis: 2, out })
    } else {
        g.node(dt, &[1, 98, 1], |out| Op::Reduce { kind: ReduceKind::Mean, x: s, axes: vec![2], keep_dims: true, out })
    }
}

fn main() {
    let npu = synaptix_npu::open().unwrap();
    let n = npu.as_ref();
    for dt in [DType::F16, DType::F32] {
        for trailing in [true, false] {
            for ln in [true, false] {
                try_graph(n, &format!("slice_trailing{trailing}_ln{ln}_{dt:?}"), |g| slice_then(g, dt, trailing, ln));
            }
        }
    }
    if std::env::args().nth(1).as_deref() != Some("all") {
        return;
    }
    for dt in [DType::F16, DType::F32] {
        for (shape, axis) in [(vec![1, 98, 256], 2), (vec![1, 1, 98, 256], 3), (vec![98, 256], 1), (vec![1, 98, 256], 1)] {
            for kind in [ReduceKind::Mean, ReduceKind::Sum, ReduceKind::Max] {
                for keep in [true, false] {
                    let name = format!("{kind:?}_{dt:?}_{shape:?}_ax{axis}_keep{keep}");
                    try_graph(n, &name, |g| reduce(g, dt, &shape, axis, kind, keep));
                }
            }
        }
    }
    for dt in [DType::F16, DType::F32] {
        try_graph(n, &format!("rsqrt_{dt:?}"), |g| {
            let x = g.input(dt, &[1, 98, 1]);
            g.unary(UnaryKind::Rsqrt, x)
        });
        try_graph(n, &format!("layernorm_{dt:?}"), |g| {
            let x = g.input(dt, &[1, 98, 256]);
            let gm = g.weight(dt, &[256], &vec![0u8; dt.bytes(256)]).unwrap();
            let bt = g.weight(dt, &[256], &vec![0u8; dt.bytes(256)]).unwrap();
            g.node(dt, &[1, 98, 256], |out| Op::LayerNorm { x, gamma: gm, beta: Some(bt), eps: 1e-5, axis: 2, out })
        });
    }
}
