//! Численная проверка `conv1d_backward` / `conv2d_backward`.
//!
//! Сравниваем `(grad_input, grad_weight, grad_bias)` с central-difference numerical grad от
//! `Σ conv*(input, weight, bias)`. Реальные конв-ops живут в `synaptix-ops::conv`.

use synaptix_autograd::grad_fn::conv::{conv1d_backward, conv2d_backward};
use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::conv::{conv1d, conv2d};

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape.to_vec(), Device::Cpu).unwrap()
}

fn flat(t: &Tensor) -> Vec<f32> {
    let n: usize = t.dims().iter().product();
    t.contiguous().unwrap().reshape((n,)).unwrap().to_vec1::<f32>().unwrap()
}

fn scalar_sum_of_conv1d(
    inp: &[f32],
    inp_shape: &[usize],
    w: &[f32],
    w_shape: &[usize],
    b: Option<&[f32]>,
    stride: usize,
    padding: usize,
) -> f32 {
    let input = t(inp.to_vec(), inp_shape);
    let weight = t(w.to_vec(), w_shape);
    let bias = b.map(|v| t(v.to_vec(), &[w_shape[0]]));
    let out = conv1d(&input, &weight, bias.as_ref(), stride, padding).unwrap();
    let s = out.contiguous().unwrap();
    s.reshape((s.dims().iter().product::<usize>(),))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .sum()
}

fn scalar_sum_of_conv2d(
    inp: &[f32],
    inp_shape: &[usize],
    w: &[f32],
    w_shape: &[usize],
    b: Option<&[f32]>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> f32 {
    let input = t(inp.to_vec(), inp_shape);
    let weight = t(w.to_vec(), w_shape);
    let bias = b.map(|v| t(v.to_vec(), &[w_shape[0]]));
    let out = conv2d(&input, &weight, bias.as_ref(), stride, padding, (1, 1)).unwrap();
    let s = out.contiguous().unwrap();
    s.reshape((s.dims().iter().product::<usize>(),))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .sum()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "длина град. векторов");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        let scale = a.abs().max(e.abs()).max(1.0);
        assert!(
            diff / scale < tol,
            "idx {i}: actual={a} expected={e} (diff={diff})"
        );
    }
}

#[test]
fn conv1d_backward_matches_numerical_stride1_pad0() {
    ensure_registered();
    // B=1, C_in=2, L=5; C_out=3, K=2 → out_len = 4.
    let inp: Vec<f32> = (1..=10).map(|x| x as f32 * 0.1).collect();
    let inp_shape = vec![1, 2, 5];
    let w: Vec<f32> = (1..=12).map(|x| x as f32 * 0.05).collect();
    let w_shape = vec![3, 2, 2];
    let b_data: Vec<f32> = vec![0.01, -0.02, 0.03];
    let stride = 1;
    let padding = 0;

    let input = t(inp.clone(), &inp_shape);
    let weight = t(w.clone(), &w_shape);
    let bias = t(b_data.clone(), &[3]);
    let out = conv1d(&input, &weight, Some(&bias), stride, padding).unwrap();
    let grad_output = Tensor::ones(out.dims().to_vec(), out.dtype(), Device::Cpu).unwrap();

    let (grad_in, grad_w, grad_b) =
        conv1d_backward(&grad_output, &input, &weight, stride, padding).unwrap();

    // Численный grad для input — central diff по Σ conv1d(...)
    let eps = 1e-3f32;
    let mut num_gi = vec![0.0f32; inp.len()];
    for i in 0..inp.len() {
        let mut p = inp.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv1d(&p, &inp_shape, &w, &w_shape, Some(&b_data), stride, padding);
        let mut m = inp.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv1d(&m, &inp_shape, &w, &w_shape, Some(&b_data), stride, padding);
        num_gi[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_in), &num_gi, 1e-2);

    // Численный grad для weight.
    let mut num_gw = vec![0.0f32; w.len()];
    for i in 0..w.len() {
        let mut p = w.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv1d(&inp, &inp_shape, &p, &w_shape, Some(&b_data), stride, padding);
        let mut m = w.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv1d(&inp, &inp_shape, &m, &w_shape, Some(&b_data), stride, padding);
        num_gw[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_w), &num_gw, 1e-2);

    // Численный grad для bias.
    let mut num_gb = vec![0.0f32; b_data.len()];
    for i in 0..b_data.len() {
        let mut p = b_data.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv1d(&inp, &inp_shape, &w, &w_shape, Some(&p), stride, padding);
        let mut m = b_data.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv1d(&inp, &inp_shape, &w, &w_shape, Some(&m), stride, padding);
        num_gb[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_b.unwrap()), &num_gb, 1e-2);
}

#[test]
fn conv1d_backward_stride2_padding1() {
    ensure_registered();
    // B=2, C_in=2, L=6; C_out=2, K=3, stride=2, padding=1 → L_pad=8, out_len=3.
    let inp: Vec<f32> = (1..=24).map(|x| x as f32 * 0.07).collect();
    let inp_shape = vec![2, 2, 6];
    let w: Vec<f32> = (1..=12).map(|x| x as f32 * 0.04).collect();
    let w_shape = vec![2, 2, 3];
    let stride = 2;
    let padding = 1;

    let input = t(inp.clone(), &inp_shape);
    let weight = t(w.clone(), &w_shape);
    let out = conv1d(&input, &weight, None, stride, padding).unwrap();
    let grad_output = Tensor::ones(out.dims().to_vec(), out.dtype(), Device::Cpu).unwrap();

    let (grad_in, grad_w, _grad_b) =
        conv1d_backward(&grad_output, &input, &weight, stride, padding).unwrap();

    let eps = 1e-3f32;
    let mut num_gi = vec![0.0f32; inp.len()];
    for i in 0..inp.len() {
        let mut p = inp.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv1d(&p, &inp_shape, &w, &w_shape, None, stride, padding);
        let mut m = inp.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv1d(&m, &inp_shape, &w, &w_shape, None, stride, padding);
        num_gi[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_in), &num_gi, 1e-2);

    let mut num_gw = vec![0.0f32; w.len()];
    for i in 0..w.len() {
        let mut p = w.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv1d(&inp, &inp_shape, &p, &w_shape, None, stride, padding);
        let mut m = w.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv1d(&inp, &inp_shape, &m, &w_shape, None, stride, padding);
        num_gw[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_w), &num_gw, 1e-2);
}

#[test]
fn conv2d_backward_matches_numerical_stride1_pad0() {
    ensure_registered();
    // B=1, C_in=2, H=4, W=4; C_out=2, KH=3, KW=3, stride=(1,1), padding=(0,0).
    let inp: Vec<f32> = (1..=32).map(|x| x as f32 * 0.03).collect();
    let inp_shape = vec![1, 2, 4, 4];
    let w: Vec<f32> = (1..=36).map(|x| x as f32 * 0.02).collect();
    let w_shape = vec![2, 2, 3, 3];
    let b_data: Vec<f32> = vec![0.1, -0.05];
    let stride = (1usize, 1usize);
    let padding = (0usize, 0usize);

    let input = t(inp.clone(), &inp_shape);
    let weight = t(w.clone(), &w_shape);
    let bias = t(b_data.clone(), &[2]);
    let out = conv2d(&input, &weight, Some(&bias), stride, padding, (1, 1)).unwrap();
    let grad_output = Tensor::ones(out.dims().to_vec(), out.dtype(), Device::Cpu).unwrap();

    let (grad_in, grad_w, grad_b) =
        conv2d_backward(&grad_output, &input, &weight, stride, padding, (1, 1)).unwrap();

    let eps = 1e-3f32;
    let mut num_gi = vec![0.0f32; inp.len()];
    for i in 0..inp.len() {
        let mut p = inp.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv2d(&p, &inp_shape, &w, &w_shape, Some(&b_data), stride, padding);
        let mut m = inp.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv2d(&m, &inp_shape, &w, &w_shape, Some(&b_data), stride, padding);
        num_gi[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_in), &num_gi, 1e-2);

    let mut num_gw = vec![0.0f32; w.len()];
    for i in 0..w.len() {
        let mut p = w.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv2d(&inp, &inp_shape, &p, &w_shape, Some(&b_data), stride, padding);
        let mut m = w.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv2d(&inp, &inp_shape, &m, &w_shape, Some(&b_data), stride, padding);
        num_gw[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_w), &num_gw, 1e-2);

    let mut num_gb = vec![0.0f32; b_data.len()];
    for i in 0..b_data.len() {
        let mut p = b_data.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv2d(&inp, &inp_shape, &w, &w_shape, Some(&p), stride, padding);
        let mut m = b_data.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv2d(&inp, &inp_shape, &w, &w_shape, Some(&m), stride, padding);
        num_gb[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_b.unwrap()), &num_gb, 1e-2);
}

#[test]
fn conv2d_backward_same_padding() {
    ensure_registered();
    // B=1, C_in=1, H=5, W=5; C_out=2, KH=3, KW=3, stride=(1,1), padding=(1,1) → out=(5,5).
    // Это «same convolution» — самый частый случай в CNN-моделях.
    let inp: Vec<f32> = (1..=25).map(|x| x as f32 * 0.04).collect();
    let inp_shape = vec![1, 1, 5, 5];
    let w: Vec<f32> = (1..=18).map(|x| x as f32 * 0.03).collect();
    let w_shape = vec![2, 1, 3, 3];
    let stride = (1usize, 1usize);
    let padding = (1usize, 1usize);

    let input = t(inp.clone(), &inp_shape);
    let weight = t(w.clone(), &w_shape);
    let out = conv2d(&input, &weight, None, stride, padding, (1, 1)).unwrap();
    let grad_output = Tensor::ones(out.dims().to_vec(), out.dtype(), Device::Cpu).unwrap();

    let (grad_in, grad_w, _) =
        conv2d_backward(&grad_output, &input, &weight, stride, padding, (1, 1)).unwrap();

    let eps = 1e-3f32;
    let mut num_gi = vec![0.0f32; inp.len()];
    for i in 0..inp.len() {
        let mut p = inp.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv2d(&p, &inp_shape, &w, &w_shape, None, stride, padding);
        let mut m = inp.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv2d(&m, &inp_shape, &w, &w_shape, None, stride, padding);
        num_gi[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_in), &num_gi, 1e-2);

    let mut num_gw = vec![0.0f32; w.len()];
    for i in 0..w.len() {
        let mut p = w.clone();
        p[i] += eps;
        let yp = scalar_sum_of_conv2d(&inp, &inp_shape, &p, &w_shape, None, stride, padding);
        let mut m = w.clone();
        m[i] -= eps;
        let ym = scalar_sum_of_conv2d(&inp, &inp_shape, &m, &w_shape, None, stride, padding);
        num_gw[i] = (yp - ym) / (2.0 * eps);
    }
    assert_close(&flat(&grad_w), &num_gw, 1e-2);
}

#[test]
fn conv2d_backward_rejects_dilation() {
    ensure_registered();
    let input = t(vec![1.0; 16], &[1, 1, 4, 4]);
    let weight = t(vec![1.0; 9], &[1, 1, 3, 3]);
    let grad_output = t(vec![1.0; 4], &[1, 1, 2, 2]);
    let err = conv2d_backward(&grad_output, &input, &weight, (1, 1), (0, 0), (2, 2));
    assert!(err.is_err(), "dilation != 1 должна давать Err");
}
