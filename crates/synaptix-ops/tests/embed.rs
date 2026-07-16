use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_ops::embed::{
    LogMelConfig, log_mel_spectrogram, patch_embed_2d, patch_embed_3d, select_anyres_grid,
    timestep_embedding, timestep_projection, token_embedding,
};

fn approx_eq(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() <= tol }

#[test]
fn token_embedding_picks_correct_rows() {
    synaptix_kernels_cpu::ensure_registered();
    let weight = Tensor::from_vec(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        (3, 3),
        Device::Cpu,
    )
    .unwrap();
    let ids = Tensor::from_vec(vec![0_u32, 2, 1, 0], (4,), Device::Cpu).unwrap();
    let out = token_embedding(&ids, &weight).unwrap();
    assert_eq!(out.dims(), &[4, 3]);
    let v: Vec<f32> = out.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    assert_eq!(v[0..3], [1.0, 2.0, 3.0]);
    assert_eq!(v[3..6], [7.0, 8.0, 9.0]);
    assert_eq!(v[6..9], [4.0, 5.0, 6.0]);
    assert_eq!(v[9..12], [1.0, 2.0, 3.0]);
}

#[test]
fn token_embedding_2d_ids_keeps_shape() {
    synaptix_kernels_cpu::ensure_registered();
    let weight = Tensor::from_vec((0..20).map(|i| i as f32).collect::<Vec<_>>(), (5, 4), Device::Cpu).unwrap();
    let ids = Tensor::from_vec(vec![0_u32, 1, 2, 3], (2, 2), Device::Cpu).unwrap();
    let out = token_embedding(&ids, &weight).unwrap();
    assert_eq!(out.dims(), &[2, 2, 4]);
}

#[test]
fn patch_embed_2d_predictable_output() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(
        (1..=16).map(|i| i as f32).collect::<Vec<_>>(),
        (1, 1, 4, 4),
        Device::Cpu,
    )
    .unwrap();
    let weight = Tensor::from_vec(vec![1.0_f32; 1 * 1 * 2 * 2], (1, 1, 2, 2), Device::Cpu).unwrap();
    let out = patch_embed_2d(&x, &weight, None, 2, None).unwrap();
    assert_eq!(out.dims(), &[1, 2, 2, 1]);
    let v: Vec<f32> = out.to_vec1::<f32>().ok().unwrap_or_else(|| {
        let t = out.contiguous().unwrap().reshape((4usize,)).unwrap();
        t.to_vec1::<f32>().unwrap()
    });
    assert!(approx_eq(v[0], 14.0, 1e-5));
    assert!(approx_eq(v[1], 22.0, 1e-5));
    assert!(approx_eq(v[2], 46.0, 1e-5));
    assert!(approx_eq(v[3], 54.0, 1e-5));
}

#[test]
fn patch_embed_3d_shapes() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(vec![1.0_f32; 1 * 2 * 2 * 4 * 4], (1, 2, 2, 4, 4), Device::Cpu).unwrap();
    let weight = Tensor::from_vec(
        vec![0.1_f32; 8 * 2 * 1 * 2 * 2],
        (8, 2, 1, 2, 2),
        Device::Cpu,
    )
    .unwrap();
    let out = patch_embed_3d(&x, &weight, None, 1, 2, 2, None, None, None).unwrap();
    assert_eq!(out.dims(), &[1, 2, 2, 2, 8]);
}

#[test]
fn timestep_embedding_shape_and_zero_t() {
    synaptix_kernels_cpu::ensure_registered();
    let t = Tensor::from_vec(vec![0.0_f32, 100.0], (2,), Device::Cpu).unwrap();
    let emb = timestep_embedding(&t, 8, 10000.0).unwrap();
    assert_eq!(emb.dims(), &[2, 8]);
    let v: Vec<f32> = emb.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    for k in 0..4usize {
        assert!(approx_eq(v[k], 1.0, 1e-5), "cos(0)=1 expected at {k}");
        assert!(approx_eq(v[4 + k], 0.0, 1e-5), "sin(0)=0 expected at {k}");
    }
}

#[test]
fn log_mel_spectrogram_synthetic_sine() {
    synaptix_kernels_cpu::ensure_registered();
    let cfg = LogMelConfig {
        n_fft: 64,
        hop: 32,
        win: 64,
        n_mels: 16,
        sample_rate: 1000,
        fmin: 0.0,
        fmax: 500.0,
        log_offset: 1e-8,
    };
    let n = 256usize;
    let freq = 100.0_f32;
    let pcm: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32) / 1000.0).sin())
        .collect();
    let pcm_t = Tensor::from_vec(pcm, (n,), Device::Cpu).unwrap();
    let mel = log_mel_spectrogram(&pcm_t, cfg).unwrap();
    assert_eq!(mel.dims(), &[16, 7]);
    let v: Vec<f32> = mel.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    let frame = 3usize;
    let mut argmax = 0usize;
    let mut best = f32::NEG_INFINITY;
    for m in 0..16 {
        let val = v[m * 7 + frame];
        if val > best {
            best = val;
            argmax = m;
        }
    }
    assert!(argmax >= 1 && argmax <= 8, "peak at mel {argmax}");
}

#[test]
fn select_anyres_grid_chooses_close_aspect() {
    let grid = select_anyres_grid(
        720,
        1280,
        &[(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)],
        336,
    )
    .unwrap();
    assert_eq!((grid.grid_h, grid.grid_w), (1, 2));
}

#[test]
fn timestep_projection_pipeline_runs() {
    synaptix_kernels_cpu::ensure_registered();
    let t = Tensor::from_vec(vec![1.0_f32, 50.0, 999.0], (3,), Device::Cpu).unwrap();
    let w1 = Tensor::from_vec(vec![0.01_f32; 8 * 4], (8, 4), Device::Cpu).unwrap();
    let w2 = Tensor::from_vec(vec![0.01_f32; 8 * 8], (8, 8), Device::Cpu).unwrap();
    let out = timestep_projection(&t, 4, 10000.0, &w1, None, &w2, None).unwrap();
    assert_eq!(out.dims(), &[3, 8]);
}
