use rand::SeedableRng;
use rand::rngs::StdRng;
use synaptix_core::device::Device;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_vision::transforms::pad::PadFill;
use synaptix_vision::{
    any_res_tiles, center_crop, color_jitter, flip_horizontal, flip_vertical, load_rgb_image,
    nms_iou, normalize, pad_to_multiple, random_crop, resize_bilinear, resize_nearest, rotate90,
    rgb_to_tensor_chw, save_rgb_image, tensor_chw_to_rgb, AnyResConfig, BBox, ColorJitterConfig,
    RgbImage, IMAGENET_MEAN, IMAGENET_STD,
};
use synaptix_vision::video::frame_sample::dense_sample;
use synaptix_vision::{uniform_sample, temporal_crop};

fn setup() {
    ensure_registered();
}

fn make_test_image() -> RgbImage {
    let w = 4usize;
    let h = 3usize;
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            data.push((x as f32) / 3.0);
            data.push((y as f32) / 2.0);
            data.push(0.5);
        }
    }
    RgbImage::new_hwc(data, w, h, 3).unwrap()
}

#[test]
fn resize_bilinear_identity() {
    let img = make_test_image();
    let r = resize_bilinear(&img, 4, 3).unwrap();
    assert_eq!(r.width, 4);
    assert_eq!(r.height, 3);
    for y in 0..3 {
        for x in 0..4 {
            for c in 0..3 {
                let diff = (r.pixel(x, y, c) - img.pixel(x, y, c)).abs();
                assert!(diff < 1e-4, "diff {diff} at ({x},{y},{c})");
            }
        }
    }
}

#[test]
fn resize_bilinear_upscale_smooth() {
    let img = make_test_image();
    let r = resize_bilinear(&img, 8, 6).unwrap();
    assert_eq!(r.width, 8);
    assert_eq!(r.height, 6);
    for c in 0..3 {
        for y in 0..6 {
            for x in 0..8 {
                let v = r.pixel(x, y, c);
                assert!(v >= 0.0 && v <= 1.0, "v={v}");
            }
        }
    }
}

#[test]
fn resize_nearest_keeps_values() {
    let img = make_test_image();
    let r = resize_nearest(&img, 8, 6).unwrap();
    assert_eq!(r.width, 8);
    assert_eq!(r.height, 6);
}

#[test]
fn normalize_subtracts_mean_divides_std() {
    let img = make_test_image();
    let n = normalize(&img, &IMAGENET_MEAN, &IMAGENET_STD).unwrap();
    for y in 0..n.height {
        for x in 0..n.width {
            for c in 0..n.channels {
                let expected = (img.pixel(x, y, c) - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                let actual = n.pixel(x, y, c);
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }
}

#[test]
fn center_crop_keeps_center() {
    let img = make_test_image();
    let c = center_crop(&img, 2, 1).unwrap();
    assert_eq!(c.width, 2);
    assert_eq!(c.height, 1);
}

#[test]
fn random_crop_within_bounds() {
    let img = make_test_image();
    let mut rng = StdRng::seed_from_u64(42);
    let c = random_crop(&img, 2, 2, &mut rng).unwrap();
    assert_eq!(c.width, 2);
    assert_eq!(c.height, 2);
}

#[test]
fn flip_horizontal_mirrors() {
    let img = make_test_image();
    let f = flip_horizontal(&img).unwrap();
    for y in 0..img.height {
        for x in 0..img.width {
            for c in 0..img.channels {
                assert_eq!(
                    f.pixel(img.width - 1 - x, y, c),
                    img.pixel(x, y, c),
                );
            }
        }
    }
}

#[test]
fn flip_vertical_mirrors() {
    let img = make_test_image();
    let f = flip_vertical(&img).unwrap();
    for y in 0..img.height {
        for x in 0..img.width {
            for c in 0..img.channels {
                assert_eq!(
                    f.pixel(x, img.height - 1 - y, c),
                    img.pixel(x, y, c),
                );
            }
        }
    }
}

#[test]
fn rotate_4_times_returns_original() {
    let img = make_test_image();
    let r1 = rotate90(&img, 1).unwrap();
    let r2 = rotate90(&r1, 1).unwrap();
    let r3 = rotate90(&r2, 1).unwrap();
    let r4 = rotate90(&r3, 1).unwrap();
    assert_eq!(r4.width, img.width);
    assert_eq!(r4.height, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            for c in 0..img.channels {
                let diff = (r4.pixel(x, y, c) - img.pixel(x, y, c)).abs();
                assert!(diff < 1e-6);
            }
        }
    }
}

#[test]
fn color_jitter_clamps_to_0_1() {
    let img = make_test_image();
    let mut rng = StdRng::seed_from_u64(7);
    let cfg = ColorJitterConfig { brightness: 0.4, contrast: 0.4, saturation: 0.4 };
    let j = color_jitter(&img, &cfg, &mut rng).unwrap();
    for v in &j.data {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn pad_to_multiple_pads_correctly() {
    let img = make_test_image();
    let p = pad_to_multiple(&img, 8, PadFill::Zero).unwrap();
    assert_eq!(p.width, 8);
    assert_eq!(p.height, 8);
    for y in 0..img.height {
        for x in 0..img.width {
            for c in 0..img.channels {
                assert_eq!(p.pixel(x, y, c), img.pixel(x, y, c));
            }
        }
    }
    for x in img.width..p.width {
        for y in 0..p.height {
            for c in 0..p.channels {
                assert_eq!(p.pixel(x, y, c), 0.0);
            }
        }
    }
}

#[test]
fn tensor_round_trip() {
    setup();
    let img = make_test_image();
    let t = rgb_to_tensor_chw(&img, Device::Cpu).unwrap();
    assert_eq!(t.dims(), &[3, img.height, img.width]);
    let back = tensor_chw_to_rgb(&t).unwrap();
    let back_hwc = back.to_hwc();
    for y in 0..img.height {
        for x in 0..img.width {
            for c in 0..img.channels {
                assert_eq!(back_hwc.pixel(x, y, c), img.pixel(x, y, c));
            }
        }
    }
}

#[test]
fn any_res_tiles_fits_max() {
    let img = RgbImage::zeros_hwc(80, 56, 3);
    let cfg = AnyResConfig { tile: 32, max_tiles: 4 };
    let tiles = any_res_tiles(&img, &cfg).unwrap();
    assert!(tiles.len() <= 4);
    for t in &tiles {
        assert_eq!(t.width, 32);
        assert_eq!(t.height, 32);
    }
}

#[test]
fn nms_iou_keeps_best() {
    let boxes = vec![
        BBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0, score: 0.9 },
        BBox { x1: 1.0, y1: 1.0, x2: 9.0, y2: 9.0, score: 0.8 },
        BBox { x1: 20.0, y1: 20.0, x2: 30.0, y2: 30.0, score: 0.7 },
    ];
    let kept = nms_iou(&boxes, 0.3);
    assert_eq!(kept, vec![0, 2]);
}

#[test]
fn bbox_iou_correct() {
    let a = BBox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0, score: 1.0 };
    let b = BBox { x1: 5.0, y1: 5.0, x2: 15.0, y2: 15.0, score: 1.0 };
    let iou = a.iou(&b);
    assert!((iou - (25.0 / 175.0)).abs() < 1e-5);
}

#[test]
fn frame_uniform_sample() {
    let s = uniform_sample(10, 3).unwrap();
    assert_eq!(s, vec![0, 5, 9]);
    let s_one = uniform_sample(10, 1).unwrap();
    assert_eq!(s_one, vec![5]);
    let s_empty = uniform_sample(0, 5).unwrap();
    assert!(s_empty.is_empty());
}

#[test]
fn frame_dense_sample() {
    let s = dense_sample(10, 2).unwrap();
    assert_eq!(s, vec![0, 2, 4, 6, 8]);
}

#[test]
fn temporal_crop_basic() {
    let frames = vec!["a", "b", "c", "d", "e"];
    let c = temporal_crop(&frames, 1, 3).unwrap();
    assert_eq!(c, vec!["b", "c", "d"]);
}

#[test]
fn png_round_trip() {
    let img = make_test_image();
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("test.png");
    save_rgb_image(&img, &p).unwrap();
    let back = load_rgb_image(&p).unwrap();
    assert_eq!(back.width, img.width);
    assert_eq!(back.height, img.height);
    for y in 0..img.height {
        for x in 0..img.width {
            for c in 0..img.channels {
                let diff = (back.pixel(x, y, c) - img.pixel(x, y, c)).abs();
                assert!(diff < 1.0 / 255.0 + 1e-5, "diff {diff} at ({x},{y},{c})");
            }
        }
    }
}
