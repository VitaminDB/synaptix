
use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::elementwise::rope::{
    apply_partial_bf16, apply_partial_f16, apply_partial_f32, RopeKernels,
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

fn build_cos_sin_f32(max_seq_len: usize, rotary_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0_f32; max_seq_len * rotary_dim];
    let mut sin = vec![0.0_f32; max_seq_len * rotary_dim];
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / rotary_dim as f32))
        .collect();
    for t in 0..max_seq_len {
        for i in 0..half {
            let theta = t as f32 * inv_freq[i];
            let c = theta.cos();
            let s = theta.sin();
            cos[t * rotary_dim + i] = c;
            cos[t * rotary_dim + i + half] = c;
            sin[t * rotary_dim + i] = s;
            sin[t * rotary_dim + i + half] = s;
        }
    }
    (cos, sin)
}

fn cpu_rope_ref(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    b: usize,
    h: usize,
    t: usize,
    head_dim: usize,
    rotary_dim: usize,
    start_pos: usize,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    let mut out = vec![0.0_f32; b * h * t * head_dim];
    for bi in 0..b {
        for hi in 0..h {
            for ti in 0..t {
                let pos = start_pos + ti;
                let base_x = ((bi * h + hi) * t + ti) * head_dim;
                for d in 0..head_dim {
                    if d >= rotary_dim {
                        out[base_x + d] = x[base_x + d];
                        continue;
                    }
                    let low = d < half;
                    let partner = if low { d + half } else { d - half };
                    let c = cos[pos * rotary_dim + d];
                    let s = sin[pos * rotary_dim + d];
                    let xv = x[base_x + d];
                    let xp = x[base_x + partner];
                    let rot = if low { -xp } else { xp };
                    out[base_x + d] = xv * c + rot * s;
                }
            }
        }
    }
    out
}

#[test]
fn rope_f32_basic() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = RopeKernels::for_context(&ctx).expect("compile rope");

    let b = 1usize;
    let h = 2usize;
    let t = 4usize;
    let head_dim = 128usize;
    let rotary_dim = 128usize;
    let max_seq = 64usize;
    let start_pos = 8usize;
    let x_host = det_f32(0xA110_C8E1, b * h * t * head_dim, 0.5);
    let (cos_host, sin_host) = build_cos_sin_f32(max_seq, rotary_dim, 10000.0);

    let dev_x: CudaSlice<f32> = stream.clone_htod(&x_host).unwrap();
    let mut dev_y: CudaSlice<f32> = stream.alloc_zeros(b * h * t * head_dim).unwrap();
    let dev_cos: CudaSlice<f32> = stream.clone_htod(&cos_host).unwrap();
    let dev_sin: CudaSlice<f32> = stream.clone_htod(&sin_host).unwrap();
    let dev_pos: CudaSlice<u32> = stream.clone_htod(&[start_pos as u32]).unwrap();

    apply_partial_f32(
        &kernels,
        &stream,
        &dev_x,
        &mut dev_y,
        &dev_cos,
        &dev_sin,
        &dev_pos,
        b as u32,
        h as u32,
        t as u32,
        head_dim as u32,
        rotary_dim as u32,
    )
    .expect("apply f32");
    stream.synchronize().unwrap();

    let got: Vec<f32> = stream.clone_dtoh(&dev_y).unwrap();
    let expected = cpu_rope_ref(
        &x_host, &cos_host, &sin_host, b, h, t, head_dim, rotary_dim, start_pos,
    );
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[rope_f32 b={b} h={h} t={t} d={head_dim} r={rotary_dim}] max_abs={max_abs:.6}");
    assert!(max_abs < 1e-5, "rope f32 max_abs={max_abs}");
}

#[test]
fn rope_f16_partial() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = RopeKernels::for_context(&ctx).expect("compile rope");

    let b = 2usize;
    let h = 4usize;
    let t = 8usize;
    let head_dim = 128usize;
    let rotary_dim = 64usize; // partial RoPE
    let max_seq = 64usize;
    let start_pos = 0usize;
    let x_f32 = det_f32(0xC0DE_BA5E, b * h * t * head_dim, 0.5);
    let x_host: Vec<f16> = x_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let (cos_f32, sin_f32) = build_cos_sin_f32(max_seq, rotary_dim, 10000.0);
    let cos_host: Vec<f16> = cos_f32.iter().map(|v| f16::from_f32(*v)).collect();
    let sin_host: Vec<f16> = sin_f32.iter().map(|v| f16::from_f32(*v)).collect();

    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();
    let mut dev_y: CudaSlice<f16> = stream.alloc_zeros(b * h * t * head_dim).unwrap();
    let dev_cos: CudaSlice<f16> = stream.clone_htod(&cos_host).unwrap();
    let dev_sin: CudaSlice<f16> = stream.clone_htod(&sin_host).unwrap();
    let dev_pos: CudaSlice<u32> = stream.clone_htod(&[start_pos as u32]).unwrap();

    apply_partial_f16(
        &kernels,
        &stream,
        &dev_x,
        &mut dev_y,
        &dev_cos,
        &dev_sin,
        &dev_pos,
        b as u32,
        h as u32,
        t as u32,
        head_dim as u32,
        rotary_dim as u32,
    )
    .expect("apply f16");
    stream.synchronize().unwrap();

    let got_f16: Vec<f16> = stream.clone_dtoh(&dev_y).unwrap();
    let got: Vec<f32> = got_f16.iter().map(|v| v.to_f32()).collect();
    let x_back: Vec<f32> = x_host.iter().map(|v| v.to_f32()).collect();
    let cos_back: Vec<f32> = cos_host.iter().map(|v| v.to_f32()).collect();
    let sin_back: Vec<f32> = sin_host.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_rope_ref(
        &x_back, &cos_back, &sin_back, b, h, t, head_dim, rotary_dim, start_pos,
    );
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!(
        "[rope_f16_partial b={b} h={h} t={t} d={head_dim} r={rotary_dim}] max_abs={max_abs:.4}"
    );
    assert!(max_abs < 0.01, "rope f16 partial max_abs={max_abs}");
}

#[test]
fn rope_bf16_basic() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = RopeKernels::for_context(&ctx).expect("compile rope");

    let b = 1usize;
    let h = 4usize;
    let t = 4usize;
    let head_dim = 256usize;
    let rotary_dim = 256usize;
    let max_seq = 64usize;
    let start_pos = 4usize;
    let x_f32 = det_f32(0xDEAD_BEEF, b * h * t * head_dim, 0.4);
    let x_host: Vec<bf16> = x_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let (cos_f32, sin_f32) = build_cos_sin_f32(max_seq, rotary_dim, 10000.0);
    let cos_host: Vec<bf16> = cos_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let sin_host: Vec<bf16> = sin_f32.iter().map(|v| bf16::from_f32(*v)).collect();

    let dev_x: CudaSlice<bf16> = stream.clone_htod(&x_host).unwrap();
    let mut dev_y: CudaSlice<bf16> = stream.alloc_zeros(b * h * t * head_dim).unwrap();
    let dev_cos: CudaSlice<bf16> = stream.clone_htod(&cos_host).unwrap();
    let dev_sin: CudaSlice<bf16> = stream.clone_htod(&sin_host).unwrap();
    let dev_pos: CudaSlice<u32> = stream.clone_htod(&[start_pos as u32]).unwrap();

    apply_partial_bf16(
        &kernels,
        &stream,
        &dev_x,
        &mut dev_y,
        &dev_cos,
        &dev_sin,
        &dev_pos,
        b as u32,
        h as u32,
        t as u32,
        head_dim as u32,
        rotary_dim as u32,
    )
    .expect("apply bf16");
    stream.synchronize().unwrap();

    let got_bf: Vec<bf16> = stream.clone_dtoh(&dev_y).unwrap();
    let got: Vec<f32> = got_bf.iter().map(|v| v.to_f32()).collect();
    let x_back: Vec<f32> = x_host.iter().map(|v| v.to_f32()).collect();
    let cos_back: Vec<f32> = cos_host.iter().map(|v| v.to_f32()).collect();
    let sin_back: Vec<f32> = sin_host.iter().map(|v| v.to_f32()).collect();
    let expected = cpu_rope_ref(
        &x_back, &cos_back, &sin_back, b, h, t, head_dim, rotary_dim, start_pos,
    );
    let mut max_abs = 0.0_f32;
    for i in 0..got.len() {
        max_abs = max_abs.max((got[i] - expected[i]).abs());
    }
    eprintln!("[rope_bf16 b={b} h={h} t={t} d={head_dim} r={rotary_dim}] max_abs={max_abs:.4}");
    assert!(max_abs < 0.05, "rope bf16 max_abs={max_abs}");
}
