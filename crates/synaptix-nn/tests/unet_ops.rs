use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::unet::{
    sinusoidal_timestep_embedding, ResNetBlock, TimeEmbedding, UNet2d, UNet3d,
    UNetAttnBlock, UNetCrossAttnBlock,
};

const D: Device = Device::Cpu;

fn t1(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(data, shape, D).unwrap()
}

fn approx_eq(a: f32, b: f32, atol: f32) {
    assert!((a - b).abs() < atol, "expected {b}, got {a}, |Δ|={:.3e}", (a - b).abs());
}

// ── sinusoidal timestep embedding: t=0 → cos(0)=1, sin(0)=0 → [1,1,...,0,0,...]. ──
#[test]
fn sinusoidal_t_zero_is_cos_ones() {
    ensure_registered();
    let t = t1(&[0.0], &[1]);
    let emb = sinusoidal_timestep_embedding(&t, 4).unwrap();
    let v = emb.to_vec2::<f32>().unwrap();
    assert_eq!(v[0].len(), 4);
    approx_eq(v[0][0], 1.0, 1e-6);
    approx_eq(v[0][1], 1.0, 1e-6);
    approx_eq(v[0][2], 0.0, 1e-6);
    approx_eq(v[0][3], 0.0, 1e-6);
}

#[test]
fn sinusoidal_in_dim_odd_errors() {
    let t = t1(&[1.0], &[1]);
    assert!(sinusoidal_timestep_embedding(&t, 5).is_err());
}

// ── TimeEmbedding shape: [B] → [B, out_dim]. ──
#[test]
fn time_embedding_shape() {
    ensure_registered();
    let te = TimeEmbedding::new(8, 16, 12, D, DType::F32).unwrap();
    let t = t1(&[1.0, 50.0, 100.0], &[3]);
    let y = te.forward(&t).unwrap();
    assert_eq!(y.dims(), &[3, 12]);
}

// ── ResNetBlock: zero conv1+conv2 + identity shortcut → output = x (residual). ──
#[test]
fn resnet_block_zero_branches_equals_x_plus_skip() {
    ensure_registered();
    let in_ch = 4;
    let out_ch = 4;
    let te = 6;
    let n1w = Tensor::ones(vec![in_ch], DType::F32, D).unwrap();
    let n1b = Tensor::zeros(vec![in_ch], DType::F32, D).unwrap();
    let c1w = Tensor::zeros(vec![out_ch, in_ch], DType::F32, D).unwrap();
    let c1b = Tensor::zeros(vec![out_ch], DType::F32, D).unwrap();
    let n2w = Tensor::ones(vec![out_ch], DType::F32, D).unwrap();
    let n2b = Tensor::zeros(vec![out_ch], DType::F32, D).unwrap();
    let c2w = Tensor::zeros(vec![out_ch, out_ch], DType::F32, D).unwrap();
    let c2b = Tensor::zeros(vec![out_ch], DType::F32, D).unwrap();
    let tew = Tensor::zeros(vec![out_ch, te], DType::F32, D).unwrap();
    let teb = Tensor::zeros(vec![out_ch], DType::F32, D).unwrap();
    let r = ResNetBlock::from_weights(
        n1w, n1b, c1w, Some(c1b),
        n2w, n2b, c2w, Some(c2b),
        tew, Some(teb),
        None, 1e-5,
    ).unwrap();
    let x = t1(&[1.0, 2.0, 3.0, 4.0,   5.0, 6.0, 7.0, 8.0], &[1, 2, 4]);
    let time_emb = t1(&[0.0; 6], &[1, 6]);
    let y = r.forward(&x, &time_emb).unwrap();
    let v = y.to_vec3::<f32>().unwrap();
    // h_branch = 0 → output = 0 + shortcut(x) = x (так как in_channels == out_channels).
    approx_eq(v[0][0][0], 1.0, 1e-5);
    approx_eq(v[0][1][3], 8.0, 1e-5);
}

// ── UNetAttnBlock: zero out_proj → output = x. ──
#[test]
fn unet_attn_zero_outproj_equals_x() {
    ensure_registered();
    let h = 4;
    let nw = Tensor::ones(vec![h], DType::F32, D).unwrap();
    let nb = Tensor::zeros(vec![h], DType::F32, D).unwrap();
    let qw = Tensor::from_slice(&[0.1f32; 16], &[h, h], D).unwrap();
    let kw = Tensor::from_slice(&[0.1f32; 16], &[h, h], D).unwrap();
    let vw = Tensor::from_slice(&[0.1f32; 16], &[h, h], D).unwrap();
    let ow = Tensor::zeros(vec![h, h], DType::F32, D).unwrap();
    let attn = UNetAttnBlock::from_weights(nw, nb, qw, kw, vw, ow, 2, 1e-5).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * h).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[2, 3, h]);
    let y = attn.forward(&x).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for b in 0..2 {
        for t in 0..3 {
            for k in 0..h {
                approx_eq(v_y[b][t][k], v_x[b][t][k], 1e-5);
            }
        }
    }
}

// ── UNetCrossAttnBlock: zero out_proj → output = x. ──
#[test]
fn unet_cross_attn_zero_outproj_equals_x() {
    ensure_registered();
    let h = 4;
    let ctx_d = 6;
    let nw = Tensor::ones(vec![h], DType::F32, D).unwrap();
    let nb = Tensor::zeros(vec![h], DType::F32, D).unwrap();
    let qw = Tensor::from_slice(&[0.1f32; 16], &[h, h], D).unwrap();
    let kv_buf = vec![0.1f32; h * ctx_d];
    let kw = Tensor::from_slice(&kv_buf, &[h, ctx_d], D).unwrap();
    let vw = Tensor::from_slice(&kv_buf, &[h, ctx_d], D).unwrap();
    let ow = Tensor::zeros(vec![h, h], DType::F32, D).unwrap();
    let attn = UNetCrossAttnBlock::from_weights(nw, nb, qw, kw, vw, ow, 2, 1e-5).unwrap();
    let x_data: Vec<f32> = (0..1 * 3 * h).map(|i| (i as f32) * 0.1).collect();
    let x = t1(&x_data, &[1, 3, h]);
    let ctx_data: Vec<f32> = (0..1 * 2 * ctx_d).map(|i| (i as f32) * 0.05).collect();
    let ctx = t1(&ctx_data, &[1, 2, ctx_d]);
    let y = attn.forward(&x, &ctx).unwrap();
    let v_x = x.to_vec3::<f32>().unwrap();
    let v_y = y.to_vec3::<f32>().unwrap();
    for t in 0..3 {
        for k in 0..h {
            approx_eq(v_y[0][t][k], v_x[0][t][k], 1e-5);
        }
    }
}

// ── UNet2d shape preservation: [B, T, in_ch] → [B, T, out_ch]. ──
#[test]
fn unet_2d_shape_preserves() {
    ensure_registered();
    let u = UNet2d::new(4, 4, 8, 2, 12, 8, 16, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4).map(|i| (i as f32) * 0.01).collect();
    let x = t1(&x_data, &[2, 3, 4]);
    let ts = t1(&[10.0, 100.0], &[2]);
    let ctx_data: Vec<f32> = (0..2 * 5 * 12).map(|i| (i as f32) * 0.005).collect();
    let ctx = t1(&ctx_data, &[2, 5, 12]);
    let y = u.forward(&x, &ts, &ctx).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
}

// ── UNet3d shape preservation: [B, T, S, in_ch] → [B, T, S, out_ch]. ──
#[test]
fn unet_3d_shape_preserves() {
    ensure_registered();
    let u = UNet3d::new(4, 4, 8, 2, 8, 16, D, DType::F32).unwrap();
    let x_data: Vec<f32> = (0..2 * 3 * 4 * 4).map(|i| (i as f32) * 0.01).collect();
    let x = t1(&x_data, &[2, 3, 4, 4]);
    let ts = t1(&[10.0, 100.0], &[2]);
    let y = u.forward(&x, &ts).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4, 4]);
}
