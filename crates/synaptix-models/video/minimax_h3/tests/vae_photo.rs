use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_video_minimax_h3 as h3;

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var("H3_MODEL_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("models/MiniMax-H3")
    });
    (p.join("FL2VA").is_dir()).then_some(p)
}

fn device() -> Option<Device> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let d = Device::Cuda(0);
    Tensor::zeros(vec![1], DType::F32, d).ok().map(|_| d)
}

fn blockiness(v: &[f32], frames: usize, h: usize, w: usize, block: usize) -> f32 {
    let (mut edge, mut inner, mut ne, mut ni) = (0.0f64, 0.0f64, 0usize, 0usize);
    for f in 0..frames {
        for y in 0..h {
            for x in 1..w {
                let d = (v[f * h * w + y * w + x] as f64 - v[f * h * w + y * w + x - 1] as f64).abs();
                if x % block == 0 {
                    edge += d;
                    ne += 1;
                } else {
                    inner += d;
                    ni += 1;
                }
            }
        }
    }
    ((edge / ne.max(1) as f64) / (inner / ni.max(1) as f64).max(1e-9)) as f32
}

#[test]
#[ignore]
fn photo_roundtrip_is_clean() {
    let Some(dev) = device() else { return };
    let Some(dir) = model_dir() else { return };
    let photo = std::env::var("H3_PHOTO").expect("H3_PHOTO");
    let out_dir = PathBuf::from(std::env::var("H3_OUT").expect("H3_OUT"));

    let paths = h3::loader::H3Paths::open(&dir).expect("paths");
    let cfg = h3::config::VaeConfig::from_dir(&paths.root).expect("config");

    let img = synaptix_io::image::png::load_image(&photo, dev).expect("фото");
    let d = img.dims().to_vec();
    let (c, h, w) = (d[0], d[1], d[2]);
    let frames = 5usize;
    let img5 = img
        .reshape(vec![1, c, 1, h, w])
        .unwrap()
        .mul_scalar(2.0)
        .unwrap()
        .add_scalar(-1.0)
        .unwrap();
    let parts: Vec<Tensor> = (0..frames).map(|_| img5.clone()).collect();
    let refs: Vec<&Tensor> = parts.iter().collect();
    let x = Tensor::cat(&refs, 2).unwrap();

    let wts = h3::loader::ComponentLoader::open_file(paths.video_vae_file(), dev).expect("веса");
    let enc = h3::vae::VaeEncoder::load(&wts, cfg.clone(), dev, DType::BF16).expect("enc");
    let z = enc.encode(&x).expect("encode");
    drop(enc);
    h3::pipeline::dump_tensor("photo_x", &x);
    h3::pipeline::dump_tensor("photo_z", &z);
    eprintln!("[photo] латент {:?}", z.dims());

    let dec = h3::vae::VaeDecoder::load(&wts, cfg, dev, DType::BF16).expect("dec");
    let y = dec.decode(&z).expect("decode");
    eprintln!("[photo] выход {:?}", y.dims());

    let yd = y.dims().to_vec();
    let of = yd[2];
    let first = y
        .narrow(0, 0, 1)
        .unwrap()
        .narrow(2, of / 2, 1)
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape(vec![3, yd[3], yd[4]])
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    synaptix_io::image::png::save_image(&first, out_dir.join("photo_out.png")).expect("save");

    let vv = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let src = x
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let n = vv.len().min(src.len());
    let (mut num, mut da, mut db, mut ma, mut mb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        ma += vv[i] as f64;
        mb += src[i] as f64;
    }
    ma /= n as f64;
    mb /= n as f64;
    for i in 0..n {
        let (u, v) = (vv[i] as f64 - ma, src[i] as f64 - mb);
        num += u * v;
        da += u * u;
        db += v * v;
    }
    eprintln!("[photo] корреляция вход/выход {:.4}", num / (da.sqrt() * db.sqrt()).max(1e-12));
    let ch0: Vec<f32> = vv[..of * yd[3] * yd[4]].to_vec();
    eprintln!(
        "[photo] блочность 16px: {:.3} (1.0 = швов нет)",
        blockiness(&ch0, of, yd[3], yd[4], 16)
    );
}
