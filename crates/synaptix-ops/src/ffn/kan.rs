use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Значения B-сплайн-базиса степени `degree` в точке `t` (Cox-de Boor).
/// Возвращает вектор длины `grid.len() − degree − 1`. Последний интервал — закрытый
/// (чтобы `t == grid[last]` не давал нулевой базис).
fn bspline_basis(t: f32, grid: &[f32], degree: usize) -> Vec<f32> {
    let m = grid.len();
    // степень 0: B_{i,0}
    let mut b = vec![0.0f32; m - 1];
    for i in 0..m - 1 {
        let in_interval = grid[i] <= t
            && (t < grid[i + 1] || (i == m - 2 && t <= grid[i + 1]));
        b[i] = if in_interval { 1.0 } else { 0.0 };
    }
    for p in 1..=degree {
        let len = m - 1 - p;
        let mut b2 = vec![0.0f32; len];
        for i in 0..len {
            let d1 = grid[i + p] - grid[i];
            let d2 = grid[i + p + 1] - grid[i + 1];
            let t1 = if d1 > 0.0 { (t - grid[i]) / d1 * b[i] } else { 0.0 };
            let t2 = if d2 > 0.0 { (grid[i + p + 1] - t) / d2 * b[i + 1] } else { 0.0 };
            b2[i] = t1 + t2;
        }
        b = b2;
    }
    b
}

/// KAN (Kolmogorov-Arnold Network) — поэлементная B-сплайн-активация:
///   `y = Σ_i coeff[i]·B_{i,degree}(clamp(x))`.
/// `grid:[num_knots]` (возрастающие узлы), `coeff:[num_knots − degree − 1]`.
/// Одна и та же сплайн-функция применяется ко всем элементам `x` (вход клампится
/// в `[grid[0], grid[last]]`).
pub fn kan_forward(x: &Tensor, grid: &Tensor, coeff: &Tensor, degree: usize) -> Result<Tensor> {
    if grid.rank() != 1 || coeff.rank() != 1 {
        return Err(SynaptixError::Unsupported("kan: grid и coeff должны быть rank-1"));
    }
    let g = f32v(grid)?;
    let c = f32v(coeff)?;
    let num_knots = g.len();
    if num_knots < degree + 2 {
        return Err(SynaptixError::Unsupported("kan: grid слишком мал для degree"));
    }
    if c.len() != num_knots - degree - 1 {
        return Err(SynaptixError::Unsupported("kan: coeff.len() != num_knots - degree - 1"));
    }
    let lo = g[0];
    let hi = g[num_knots - 1];
    let dtype_in = x.dtype();
    let xf = f32v(x)?;

    let mut out = vec![0.0f32; xf.len()];
    for (o, &v) in out.iter_mut().zip(xf.iter()) {
        let t = v.clamp(lo, hi);
        let basis = bspline_basis(t, &g, degree);
        let mut acc = 0.0f32;
        for (bi, &cv) in basis.iter().zip(c.iter()) {
            acc += cv * bi;
        }
        *o = acc;
    }
    Tensor::from_vec::<_, f32>(out, x.dims().to_vec(), x.device())?.to_dtype(dtype_in)
}
