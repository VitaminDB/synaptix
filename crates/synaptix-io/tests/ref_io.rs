#[cfg(any(feature = "audio", feature = "image"))]
use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_test_utils::{assert_allclose, assert_exact_eq, load_case, reference_data_path};

fn setup() {
    ensure_registered();
}

#[test]
fn t12_1_safetensors_f32() {
    setup();
    let t = load_case("io", "safetensors_f32");
    let expected = &t["tensor"];
    let loaded = synaptix_io::weights::safetensors::load_file(
        reference_data_path("io", "safetensors_f32.safetensors"),
        Device::Cpu,
    )
    .unwrap();
    let actual = loaded.get("tensor").expect("tensor missing");
    assert_allclose(actual, expected, 1e-7, 1e-7);
}

#[test]
fn t12_2_safetensors_dtypes() {
    setup();
    let loaded = synaptix_io::weights::safetensors::load_file(
        reference_data_path("io", "safetensors_dtypes.safetensors"),
        Device::Cpu,
    )
    .unwrap();
    let expected = load_case("io", "safetensors_dtypes");
    for key in ["f16", "bf16", "i32", "i64", "f32"] {
        let exp = &expected[key];
        let got = loaded.get(key).expect("dtype tensor missing");
        assert_eq!(got.dtype(), exp.dtype(), "dtype mismatch for {key}");
        if matches!(exp.dtype(), DType::F16 | DType::BF16 | DType::F32) {
            assert_allclose(got, exp, 1e-7, 1e-7);
        } else {
            assert_exact_eq(got, exp);
        }
    }
}

#[test]
fn t12_3_safetensors_metadata() {
    setup();
    let path = reference_data_path("io", "safetensors_metadata.safetensors");
    let bytes = std::fs::read(&path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let names = st.names();
    assert!(names.iter().any(|s| *s == "data"));

    let (_n_bytes, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let metadata = meta.metadata().clone().expect("metadata missing");
    assert_eq!(metadata.get("model_name").map(|s| s.as_str()), Some("synaptix-test"));
    assert_eq!(metadata.get("version").map(|s| s.as_str()), Some("1.0"));
    assert_eq!(metadata.get("author").map(|s| s.as_str()), Some("synaptix"));
    assert_eq!(metadata.get("custom_key").map(|s| s.as_str()), Some("hello world"));
}

#[cfg(feature = "audio")]
#[test]
fn t12_5_wav_sine_440hz() {
    setup();
    let buf = synaptix_io::audio::wav::read_wav(reference_data_path(
        "io",
        "wav_sine_440hz.wav",
    ))
    .unwrap();
    assert_eq!(buf.sample_rate, 16000);
    assert_eq!(buf.channels, 1);
    let expected = load_case("io", "wav_sine_440hz");
    let expected_samples = expected["samples_f32"].to_vec1::<f32>().unwrap();
    assert_eq!(buf.samples.len(), expected_samples.len());
    for (i, (got, exp)) in buf.samples.iter().zip(expected_samples.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(diff < 1e-4, "sample[{}] diff {} > 1e-4", i, diff);
    }
}

#[cfg(feature = "audio")]
#[test]
fn t12_6_wav_round_trip() {
    setup();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("rt.wav");

    let original = synaptix_io::audio::AudioBuffer::new(
        (0..8000).map(|i| (i as f32 * 0.001).sin() * 0.3).collect(),
        16000,
        1,
    );
    synaptix_io::audio::wav::write_wav(&original, &path).unwrap();
    let readback = synaptix_io::audio::wav::read_wav(&path).unwrap();
    assert_eq!(readback.sample_rate, 16000);
    assert_eq!(readback.channels, 1);
    assert_eq!(readback.samples.len(), original.samples.len());
    for (i, (a, b)) in readback.samples.iter().zip(original.samples.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(diff < 1e-4, "sample[{}] roundtrip diff {} > 1e-4", i, diff);
    }
}

#[cfg(feature = "image")]
#[test]
fn t12_7_png_rgb_exact() {
    setup();
    let tensor = synaptix_io::image::png::load_image(
        reference_data_path("io", "png_rgb_exact.png"),
        Device::Cpu,
    )
    .unwrap();
    assert_eq!(tensor.dims(), &[3, 64, 64]);
    let expected_i32 = load_case("io", "png_rgb_exact_ref");
    let expected = &expected_i32["pixels"];
    let expected_vals = expected.to_vec3::<i32>().unwrap();
    let flat: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
    let chw_to_hwc = synaptix_io::image::png::chw_to_hwc(&flat, 3, 64, 64);
    for row in 0..64 {
        for col in 0..64 {
            for ch in 0..3 {
                let got = (chw_to_hwc[(row * 64 + col) * 3 + ch] * 255.0).round() as i32;
                let exp = expected_vals[row][col][ch];
                assert_eq!(got, exp, "pixel[{},{},{}] mismatch", row, col, ch);
            }
        }
    }
}

#[cfg(feature = "image")]
#[test]
fn t12_8_png_round_trip() {
    setup();
    let tmp = tempfile::tempdir().unwrap();
    let path: PathBuf = tmp.path().join("rt.png");

    let expected_i32 = load_case("io", "png_round_trip_ref");
    let expected_vals = expected_i32["pixels"].to_vec3::<i32>().unwrap();
    let h = expected_vals.len();
    let w = expected_vals[0].len();
    let c = expected_vals[0][0].len();
    assert_eq!(c, 3);
    let mut hwc_f32 = vec![0.0f32; h * w * c];
    for r in 0..h {
        for col in 0..w {
            for ch in 0..c {
                hwc_f32[(r * w + col) * c + ch] = expected_vals[r][col][ch] as f32 / 255.0;
            }
        }
    }
    let chw = synaptix_io::image::png::hwc_to_chw(&hwc_f32, h, w, c);
    let bytes = synaptix_io::image::png::f32_to_bytes(&chw);
    let tensor = synaptix_core::tensor::Tensor::from_raw_bytes(
        bytes,
        vec![c, h, w],
        DType::F32,
        Device::Cpu,
    )
    .unwrap();
    synaptix_io::image::png::save_image(&tensor, &path).unwrap();

    let loaded = synaptix_io::image::png::load_image(&path, Device::Cpu).unwrap();
    let flat: Vec<f32> = loaded.flatten_all().unwrap().to_vec1().unwrap();
    let hwc_back = synaptix_io::image::png::chw_to_hwc(&flat, c, h, w);
    for r in 0..h {
        for col in 0..w {
            for ch in 0..c {
                let got = (hwc_back[(r * w + col) * c + ch] * 255.0).round() as i32;
                let exp = expected_vals[r][col][ch];
                assert_eq!(got, exp, "pixel[{},{},{}] mismatch", r, col, ch);
            }
        }
    }
}
