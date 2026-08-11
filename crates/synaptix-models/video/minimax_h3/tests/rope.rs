use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_video_minimax_h3::rope::RopeTables;

fn inv_freq() -> Vec<f32> {
    (0..16).map(|i| 1.0 / 10000f32.powf(i as f32 / 16.0)).collect()
}

#[test]
fn tables_have_three_axis_blocks() {
    synaptix_kernels_cpu::ensure_registered();
    let pos = vec![[0.0f64, 0.0, 0.0], [1.0, 2.0, 3.0]];
    let inv = inv_freq();
    let t = RopeTables::build(&pos, &inv, Device::Cpu).unwrap();
    assert_eq!(t.rot_dim, 96);
    assert_eq!(t.seq_len, 2);
    assert_eq!(t.cos.dims(), &[2, 48]);

    let cos = t.cos.to_vec2::<f32>().unwrap();
    let sin = t.sin.to_vec2::<f32>().unwrap();
    for i in 0..48 {
        assert!((cos[0][i] - 1.0).abs() < 1e-6);
        assert!(sin[0][i].abs() < 1e-6);
    }
    for (axis, p) in [1.0f32, 2.0, 3.0].iter().enumerate() {
        for i in 0..16 {
            let ang = p * inv[i];
            let o = axis * 16 + i;
            assert!((cos[1][o] - ang.cos()).abs() < 1e-5, "axis={axis} i={i}");
            assert!((sin[1][o] - ang.sin()).abs() < 1e-5, "axis={axis} i={i}");
        }
    }
}

#[test]
fn partial_rotation_leaves_tail_untouched() {
    synaptix_kernels_cpu::ensure_registered();
    let s = 3usize;
    let head_dim = 128usize;
    let heads = 2usize;
    let pos: Vec<[f64; 3]> = (0..s).map(|i| [i as f64, 0.5 * i as f64, 0.25 * i as f64]).collect();
    let t = RopeTables::build(&pos, &inv_freq(), Device::Cpu).unwrap();

    let n = heads * s * head_dim;
    let data: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
    let x = Tensor::from_vec(data.clone(), vec![heads, s, head_dim], Device::Cpu).unwrap();
    let y = t.apply(&x).unwrap();
    assert_eq!(y.dims(), &[heads, s, head_dim]);

    let out = y.reshape(vec![n]).unwrap().to_vec1::<f32>().unwrap();
    for h in 0..heads {
        for si in 0..s {
            let base = (h * s + si) * head_dim;
            for d in 96..head_dim {
                assert!(
                    (out[base + d] - data[base + d]).abs() < 1e-5,
                    "h={h} s={si} d={d}: {} vs {}",
                    out[base + d],
                    data[base + d]
                );
            }
        }
    }
}

#[test]
fn rotation_matches_split_half_formula() {
    synaptix_kernels_cpu::ensure_registered();
    let s = 2usize;
    let head_dim = 128usize;
    let half = 48usize;
    let pos = vec![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let inv = inv_freq();
    let t = RopeTables::build(&pos, &inv, Device::Cpu).unwrap();

    let n = s * head_dim;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let x = Tensor::from_vec(data.clone(), vec![1, s, head_dim], Device::Cpu).unwrap();
    let y = t.apply(&x).unwrap();
    let out = y.reshape(vec![n]).unwrap().to_vec1::<f32>().unwrap();

    let cos = t.cos.to_vec2::<f32>().unwrap();
    let sin = t.sin.to_vec2::<f32>().unwrap();
    for si in 0..s {
        let base = si * head_dim;
        for d in 0..half {
            let lo = data[base + d];
            let hi = data[base + d + half];
            let want_lo = lo * cos[si][d] - hi * sin[si][d];
            let want_hi = hi * cos[si][d] + lo * sin[si][d];
            assert!((out[base + d] - want_lo).abs() < 1e-5, "s={si} d={d}");
            assert!((out[base + d + half] - want_hi).abs() < 1e-5, "s={si} d={d}+half");
        }
    }
}

#[test]
fn zero_position_is_identity() {
    synaptix_kernels_cpu::ensure_registered();
    let pos = vec![[0.0f64, 0.0, 0.0]; 4];
    let t = RopeTables::build(&pos, &inv_freq(), Device::Cpu).unwrap();
    let data: Vec<f32> = (0..4 * 128).map(|i| (i as f32) * 0.03 - 1.0).collect();
    let x = Tensor::from_vec(data.clone(), vec![1, 4, 128], Device::Cpu).unwrap();
    let y = t.apply(&x).unwrap();
    let out = y.reshape(vec![4 * 128]).unwrap().to_vec1::<f32>().unwrap();
    for (a, b) in out.iter().zip(data.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn bf16_path_stays_close_to_f32() {
    synaptix_kernels_cpu::ensure_registered();
    let pos: Vec<[f64; 3]> = (0..8).map(|i| [i as f64, i as f64 * 0.5, 0.0]).collect();
    let t = RopeTables::build(&pos, &inv_freq(), Device::Cpu).unwrap();
    let data: Vec<f32> = (0..8 * 128).map(|i| ((i % 23) as f32) * 0.05).collect();
    let x32 = Tensor::from_vec(data, vec![1, 8, 128], Device::Cpu).unwrap();
    let x16 = x32.to_dtype(DType::BF16).unwrap();
    let y32 = t.apply(&x32).unwrap();
    let y16 = t.apply(&x16).unwrap().to_dtype(DType::F32).unwrap();
    let a = y32.reshape(vec![8 * 128]).unwrap().to_vec1::<f32>().unwrap();
    let b = y16.reshape(vec![8 * 128]).unwrap().to_vec1::<f32>().unwrap();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 0.05, "{x} vs {y}");
    }
}
