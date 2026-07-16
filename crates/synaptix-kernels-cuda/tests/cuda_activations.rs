#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};

use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::elementwise::activations::{
    run_bf16 as act_bf16, run_bias_act, run_f16 as act_f16, run_f32 as act_f32, run_swish_beta,
    Activation, ActivationsKernels, BiasActivation,
};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f * scale
        })
        .collect()
}

fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

fn ref_act(v: f32, act: Activation) -> f32 {
    match act {
        Activation::Silu => v / (1.0 + (-v).exp()),
        Activation::GeluExact => {
            let z = v as f64 * 0.70710678118654752;
            (0.5 * v as f64 * (1.0 + erf_f64(z))) as f32
        }
        Activation::GeluTanh => {
            let v3 = v * v * v;
            0.5 * v * (1.0 + (0.7978845608028654 * (v + 0.044715 * v3)).tanh())
        }
        Activation::QuickGelu => v / (1.0 + (-1.702 * v).exp()),
        Activation::Softplus => {
            if v > 20.0 {
                v
            } else {
                (v.exp()).ln_1p()
            }
        }
        Activation::Mish => {
            let sp = if v > 20.0 { v } else { (v.exp()).ln_1p() };
            v * sp.tanh()
        }
        Activation::Softsign => v / (1.0 + v.abs()),
    }
}

#[test]
fn act_f32_all() {
    let Some((ctx, stream)) = setup() else { return };
    let k = ActivationsKernels::for_context(&ctx).expect("compile");
    let n = 4096_u32;
    let x = det_f32(0xA1, n as usize, 2.0);
    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    for act in [
        Activation::Silu,
        Activation::GeluExact,
        Activation::GeluTanh,
        Activation::QuickGelu,
        Activation::Softplus,
        Activation::Mish,
        Activation::Softsign,
    ] {
        act_f32(&k, &stream, &dev_x, &mut dev_y, n, act).unwrap();
        stream.synchronize().unwrap();
        let gpu = stream.clone_dtoh(&dev_y).unwrap();
        for i in 0..n as usize {
            let r = ref_act(x[i], act);
            let d = (gpu[i] - r).abs();
            assert!(d < 1e-4, "{:?}[{i}] gpu={} ref={}", act, gpu[i], r);
        }
    }
}

#[test]
fn act_f16_silu_gelu_tanh() {
    let Some((ctx, stream)) = setup() else { return };
    let k = ActivationsKernels::for_context(&ctx).expect("compile");
    let n = 8192_u32;
    let x_f32 = det_f32(0xA2, n as usize, 1.5);
    let x: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x).unwrap();
    let mut dev_y: CudaSlice<f16> = stream.alloc_zeros(n as usize).unwrap();

    for act in [Activation::Silu, Activation::GeluTanh] {
        act_f16(&k, &stream, &dev_x, &mut dev_y, n, act).unwrap();
        stream.synchronize().unwrap();
        let gpu_f16 = stream.clone_dtoh(&dev_y).unwrap();
        let x_back: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
        for i in 0..n as usize {
            let r = ref_act(x_back[i], act);
            let g = gpu_f16[i].to_f32();
            let d = (g - r).abs();
            assert!(d < 5e-3, "{:?}[{i}] gpu={g} ref={r}", act);
        }
    }
}

#[test]
fn act_bf16_silu() {
    let Some((ctx, stream)) = setup() else { return };
    let k = ActivationsKernels::for_context(&ctx).expect("compile");
    let n = 4096_u32;
    let x_f32 = det_f32(0xA3, n as usize, 1.5);
    let x: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dev_x: CudaSlice<bf16> = stream.clone_htod(&x).unwrap();
    let mut dev_y: CudaSlice<bf16> = stream.alloc_zeros(n as usize).unwrap();

    act_bf16(&k, &stream, &dev_x, &mut dev_y, n, Activation::Silu).unwrap();
    stream.synchronize().unwrap();
    let gpu_bf16 = stream.clone_dtoh(&dev_y).unwrap();
    let x_back: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    for i in 0..n as usize {
        let r = ref_act(x_back[i], Activation::Silu);
        let g = gpu_bf16[i].to_f32();
        let d = (g - r).abs();
        assert!(d < 2e-2, "silu[{i}] gpu={g} ref={r}");
    }
}

#[test]
fn act_swish_beta() {
    let Some((ctx, stream)) = setup() else { return };
    let k = ActivationsKernels::for_context(&ctx).expect("compile");
    let n = 2048_u32;
    let beta = 0.7_f32;
    let x = det_f32(0xA4, n as usize, 2.0);
    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();

    run_swish_beta::<f32>(&k, &stream, &dev_x, &mut dev_y, n, beta, DType::F32).unwrap();
    stream.synchronize().unwrap();
    let gpu = stream.clone_dtoh(&dev_y).unwrap();
    for i in 0..n as usize {
        let r = x[i] / (1.0 + (-beta * x[i]).exp());
        let d = (gpu[i] - r).abs();
        assert!(d < 1e-4, "swish_beta[{i}] gpu={} ref={}", gpu[i], r);
    }
}

#[test]
fn act_bias_silu_gelu() {
    let Some((ctx, stream)) = setup() else { return };
    let k = ActivationsKernels::for_context(&ctx).expect("compile");
    let rows = 64_u32;
    let cols = 128_u32;
    let x = det_f32(0xB1, (rows * cols) as usize, 1.0);
    let bias = det_f32(0xB2, cols as usize, 0.3);
    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let dev_b: CudaSlice<f32> = stream.clone_htod(&bias).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros((rows * cols) as usize).unwrap();

    for (bact, refact) in [
        (BiasActivation::Silu, Activation::Silu),
        (BiasActivation::GeluTanh, Activation::GeluTanh),
    ] {
        run_bias_act::<f32>(
            &k,
            &stream,
            &dev_x,
            &dev_b,
            &mut dev_y,
            rows,
            cols,
            bact,
            DType::F32,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let gpu = stream.clone_dtoh(&dev_y).unwrap();
        for r in 0..rows as usize {
            for c in 0..cols as usize {
                let i = r * cols as usize + c;
                let v = x[i] + bias[c];
                let exp = ref_act(v, refact);
                let d = (gpu[i] - exp).abs();
                assert!(d < 1e-4, "{:?}[{r},{c}] gpu={} ref={}", bact, gpu[i], exp);
            }
        }
    }
}

#[test]
fn act_bias_relu() {
    let Some((ctx, stream)) = setup() else { return };
    let k = ActivationsKernels::for_context(&ctx).expect("compile");
    let rows = 32_u32;
    let cols = 64_u32;
    let x = det_f32(0xC1, (rows * cols) as usize, 1.0);
    let bias = det_f32(0xC2, cols as usize, 0.2);
    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();
    let dev_b: CudaSlice<f32> = stream.clone_htod(&bias).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros((rows * cols) as usize).unwrap();
    run_bias_act::<f32>(
        &k,
        &stream,
        &dev_x,
        &dev_b,
        &mut dev_y,
        rows,
        cols,
        BiasActivation::Relu,
        DType::F32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gpu = stream.clone_dtoh(&dev_y).unwrap();
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            let i = r * cols as usize + c;
            let v = x[i] + bias[c];
            let exp = v.max(0.0);
            assert!((gpu[i] - exp).abs() < 1e-5, "relu[{r},{c}]");
        }
    }
}
