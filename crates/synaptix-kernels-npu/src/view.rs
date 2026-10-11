use synaptix_core::tensor::layout::Layout;
use synaptix_npu::graph::Op;
use synaptix_npu::{DType, Graph, Result, TensorId};

fn u32s(v: &[usize]) -> Vec<u32> {
    v.iter().map(|&x| x as u32).collect()
}

fn reshape_node(g: &mut Graph, x: TensorId, shape: &[u32]) -> TensorId {
    if g.tensor(x).shape == shape {
        return x;
    }
    let root = g.reshape_root(x);
    if g.tensor(root).shape == shape {
        return root;
    }
    let dt = g.tensor(root).dtype;
    g.node(dt, shape, |out| Op::Reshape { x: root, out })
}

pub fn fresh(g: &mut Graph, x: TensorId) -> TensorId {
    let d = g.tensor(x);
    let (dt, s) = (d.dtype, d.shape.clone());
    g.node(dt, &s, |out| Op::Reshape { x, out })
}

fn tile(g: &mut Graph, x: TensorId, shape: &[usize], materialize: bool) -> TensorId {
    let cur = g.tensor(x).shape.clone();
    if !materialize || cur == u32s(shape) {
        return x;
    }
    let multiples: Vec<u32> = shape.iter().zip(&cur).map(|(&s, &c)| if c == 1 { s as u32 } else { 1 }).collect();
    let dt = g.tensor(x).dtype;
    g.node(dt, &u32s(shape), |out| Op::Tile { x, multiples, out })
}

fn gather(g: &mut Graph, flat: TensorId, shape: &[usize], strides: &[isize], offset: usize) -> Result<TensorId> {
    let numel: usize = shape.iter().product();
    let mut idx = Vec::with_capacity(numel);
    let mut coord = vec![0usize; shape.len()];
    for _ in 0..numel {
        let mut p = offset as isize;
        for (c, s) in coord.iter().zip(strides) {
            p += *c as isize * s;
        }
        idx.push(p as i32);
        for d in (0..shape.len()).rev() {
            coord[d] += 1;
            if coord[d] < shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }
    let bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ind = g.weight(DType::I32, &[numel as u32], &bytes)?;
    let dt = g.tensor(flat).dtype;
    let gathered = g.node(dt, &[numel as u32], |out| Op::Gather { x: flat, indices: ind, axis: 0, out });
    Ok(reshape_node(g, gathered, &u32s(shape)))
}

pub fn view(g: &mut Graph, base: TensorId, lo: &Layout, materialize: bool) -> Result<TensorId> {
    let n_base = g.tensor(base).elems();
    let shape = lo.dims().to_vec();
    let strides = lo.strides().as_slice().to_vec();
    let off = lo.offset();
    let numel: usize = shape.iter().product();
    if lo.is_contiguous() && off == 0 && numel == n_base {
        return Ok(reshape_node(g, base, &u32s(&shape)));
    }
    let flat = reshape_node(g, base, &[n_base as u32]);
    if strides.iter().any(|&s| s < 0) {
        return gather(g, flat, &shape, &strides, off);
    }
    let real: Vec<usize> = (0..shape.len()).filter(|&i| shape[i] > 1 && strides[i] > 0).collect();
    let mut order = real.clone();
    order.sort_by(|&a, &b| strides[b].cmp(&strides[a]));
    let ts: Vec<usize> = order.iter().map(|&i| strides[i] as usize).collect();
    let sz: Vec<usize> = order.iter().map(|&i| shape[i]).collect();
    let r = order.len();

    let structured = (|| -> Option<(Vec<usize>, Vec<usize>, usize)> {
        if r == 0 {
            return (off < n_base).then(|| (vec![n_base], vec![off], n_base));
        }
        for j in 0..r.saturating_sub(1) {
            if ts[j] % ts[j + 1] != 0 || ts[j] < ts[j + 1] * sz[j + 1] {
                return None;
            }
        }
        let a0 = n_base / ts[0];
        let last = off + (0..r).map(|j| (sz[j] - 1) * ts[j]).sum::<usize>();
        if a0 == 0 || last >= a0 * ts[0] {
            return None;
        }
        let mut axes = vec![a0];
        for j in 0..r - 1 {
            axes.push(ts[j] / ts[j + 1]);
        }
        if ts[r - 1] > 1 {
            axes.push(ts[r - 1]);
        }
        let mut digits = Vec::with_capacity(r + 1);
        let mut rem = off;
        for t in &ts {
            digits.push(rem / t);
            rem %= t;
        }
        if ts[r - 1] > 1 {
            digits.push(rem);
        }
        for j in 0..r {
            if digits[j] + sz[j] > axes[j] {
                return None;
            }
        }
        Some((axes, digits, a0 * ts[0]))
    })();

    let Some((axes, digits, used)) = structured else {
        return gather(g, flat, &shape, &strides, off);
    };

    let mut x = flat;
    if used != n_base {
        let dt = g.tensor(x).dtype;
        x = g.node(dt, &[used as u32], |out| Op::Slice {
            x: flat,
            begin: vec![0],
            end: vec![used as i32],
            stride: vec![1],
            out,
        });
    }
    x = reshape_node(g, x, &u32s(&axes));
    let mut want: Vec<usize> = sz.clone();
    if r > 0 && ts[r - 1] > 1 {
        want.push(1);
    }
    if r == 0 {
        want = vec![1];
    }
    if axes != want || digits.iter().any(|&d| d != 0) {
        let begin: Vec<i32> = digits.iter().map(|&d| d as i32).collect();
        let end: Vec<i32> = digits.iter().zip(&want).map(|(&d, &w)| (d + w) as i32).collect();
        let stride = vec![1; begin.len()];
        let dt = g.tensor(x).dtype;
        x = g.node(dt, &u32s(&want), |out| Op::Slice { x, begin, end, stride, out });
    }
    if r > 0 {
        x = reshape_node(g, x, &u32s(&sz));
        let perm: Vec<u32> = real.iter().map(|ri| order.iter().position(|o| o == ri).unwrap() as u32).collect();
        x = g.transpose(x, &perm);
    }
    let with_ones: Vec<usize> = (0..shape.len()).map(|i| if real.contains(&i) { shape[i] } else { 1 }).collect();
    x = reshape_node(g, x, &u32s(&with_ones));
    Ok(tile(g, x, &shape, materialize))
}
