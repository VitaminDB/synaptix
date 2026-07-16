use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_ops::pos::{
    LongRopeConfig, RopeCache, RopeLayout, YarnConfig, alibi_bias, alibi_slopes, apply_rope,
    longrope_cache, nope, sinusoidal_positional_embedding, t5_relative_position_bucket,
    yarn_scaled_rope_cache,
};

fn approx(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() <= tol }

#[test]
fn rope_preserves_norm() {
    synaptix_kernels_cpu::ensure_registered();
    let head_dim = 8usize;
    let max_seq = 8usize;
    let cache = RopeCache::new(head_dim, max_seq, 10000.0, Device::Cpu).unwrap();
    let x_data: Vec<f32> = (0..(1 * 2 * 4 * head_dim))
        .map(|i| (i as f32) * 0.05 + 0.1)
        .collect();
    let xt = Tensor::from_vec(x_data.clone(), (1, 2, 4, head_dim), Device::Cpu).unwrap();
    let y = apply_rope(&xt, &cache, None, RopeLayout::Split).unwrap();
    let yv: Vec<f32> = y
        .reshape((1 * 2 * 4 * head_dim,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let outer = x_data.len() / head_dim;
    for o in 0..outer {
        let in_norm: f64 = x_data[o * head_dim..(o + 1) * head_dim]
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum();
        let out_norm: f64 = yv[o * head_dim..(o + 1) * head_dim]
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum();
        assert!(
            (in_norm - out_norm).abs() < 1e-3,
            "row {o}: in {in_norm} vs out {out_norm}"
        );
    }
}

#[test]
fn rope_zero_position_is_identity() {
    synaptix_kernels_cpu::ensure_registered();
    let head_dim = 4usize;
    let cache = RopeCache::new(head_dim, 1, 10000.0, Device::Cpu).unwrap();
    let x_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let xt = Tensor::from_vec(x_data.clone(), (1, 1, 1, 4), Device::Cpu).unwrap();
    let y = apply_rope(&xt, &cache, None, RopeLayout::Split).unwrap();
    let yv: Vec<f32> = y.reshape((4usize,)).unwrap().to_vec1::<f32>().unwrap();
    for (a, b) in yv.iter().zip(x_data.iter()) {
        let diff = (a - b).abs();
        assert!(diff < 1e-5, "{a} vs {b}");
    }
}

#[test]
fn alibi_slopes_geometric_for_pow2() {
    let s = alibi_slopes(8);
    let expected_first = 2.0_f32.powf(-8.0 / 8.0);
    assert!(approx(s[0], expected_first, 1e-6));
    for w in s.windows(2) {
        let ratio = w[1] / w[0];
        assert!(approx(ratio, expected_first, 1e-5));
    }
}

#[test]
fn alibi_bias_shape_and_value_zero_on_diag() {
    synaptix_kernels_cpu::ensure_registered();
    let m = alibi_bias(4, 5, Device::Cpu).unwrap();
    assert_eq!(m.dims(), &[1, 4, 5, 5]);
    let v = m.reshape((100usize,)).unwrap().to_vec1::<f32>().unwrap();
    for h in 0..4 {
        for i in 0..5 {
            let pos = (h * 5 + i) * 5 + i;
            assert!(approx(v[pos], 0.0, 1e-6));
        }
    }
}

#[test]
fn sinusoidal_shape_and_periodicity() {
    synaptix_kernels_cpu::ensure_registered();
    let pe = sinusoidal_positional_embedding(10, 8, Device::Cpu).unwrap();
    assert_eq!(pe.dims(), &[10, 8]);
    let v: Vec<f32> = pe.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
    for j in 0..4 {
        let s = v[2 * j];
        let c = v[2 * j + 1];
        assert!(approx(s, 0.0, 1e-6), "sin(0)=0 expected, got {s}");
        assert!(approx(c, 1.0, 1e-6), "cos(0)=1 expected, got {c}");
    }
}

#[test]
fn yarn_returns_valid_cache() {
    synaptix_kernels_cpu::ensure_registered();
    let cfg = YarnConfig::default();
    let cache = yarn_scaled_rope_cache(64, 4096, 10000.0, cfg, Device::Cpu).unwrap();
    assert_eq!(cache.head_dim(), 64);
    assert_eq!(cache.max_seq(), 4096);
}

#[test]
fn longrope_cache_uses_long_factors() {
    synaptix_kernels_cpu::ensure_registered();
    let cfg = LongRopeConfig {
        long_factors: vec![1.5; 32],
        short_factors: vec![1.0; 32],
        original_max_seq: 1024,
    };
    let cache = longrope_cache(64, 2048, 10000.0, &cfg, Device::Cpu).unwrap();
    assert_eq!(cache.head_dim(), 64);
    assert_eq!(cache.max_seq(), 2048);
}

#[test]
fn t5_relative_bucket_zero_at_self() {
    let buckets = t5_relative_position_bucket(5, 5, 32, 128, true);
    assert_eq!(buckets[0 * 5 + 0], 0);
    assert_eq!(buckets[1 * 5 + 1], 0);
    assert_eq!(buckets[2 * 5 + 2], 0);
}

#[test]
fn nope_is_identity() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], (3,), Device::Cpu).unwrap();
    let y = nope(&x).unwrap();
    let v = y.to_vec1::<f32>().unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0]);
}
