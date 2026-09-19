//! Сквозной прогон FLUX на GPU: txt2img из `.syn`-бандла бит в бит равен
//! прогону из каталога, img2img через VAE-энкодер даёт картинку того же
//! размера, отмена прерывает денойз.
//!
//! ```sh
//! FLUX_DIR=…/FLUX.1-dev FLUX_SYN=…/flux.1-dev.syn FLUX_OUT=/tmp/flux \
//!   cargo test --release -p synaptix-image-flux --test bundle_e2e -- --ignored --nocapture --test-threads=1
//! ```

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_image_flux::{FluxError, FluxModel, SampleParams};

fn setup() -> Option<(String, String, std::path::PathBuf)> {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    synaptix_kernels_cpu::ensure_registered();
    let out = std::path::PathBuf::from(std::env::var("FLUX_OUT").unwrap_or_else(|_| "/tmp/flux_e2e".into()));
    std::fs::create_dir_all(&out).ok();
    Some((std::env::var("FLUX_DIR").ok()?, std::env::var("FLUX_SYN").ok()?, out))
}

const PROMPT: &str = "a lighthouse on a rocky coast at sunset, dramatic clouds, photo";

fn params(seed: u64) -> SampleParams {
    SampleParams { width: 512, height: 384, steps: 8, guidance: 3.5, seed, denoise: 1.0 }
}

fn txt2img(path: &str, p: &SampleParams) -> (FluxModel, Tensor, Tensor) {
    let m = FluxModel::open(path, Device::Cuda(0), DType::F16, DType::NVFP4).unwrap();
    let seq = m.default_max_seq_len();
    let cond = m.encode_prompt(PROMPT, seq).unwrap();
    let tr = m.load_transformer(FluxModel::tokens_for(p.width, p.height, seq)).unwrap();
    let lat = m.sample(&tr, &cond, None, p, &mut |_, _| true).unwrap();
    drop(tr);
    let img = m.decode(&lat).unwrap();
    (m, lat, img)
}

fn bits(t: &Tensor) -> Vec<u32> {
    let n: usize = t.dims().iter().product();
    t.to_device(Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape(vec![n])
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[test]
#[ignore]
fn bundle_and_directory_give_identical_image() {
    let Some((dir, syn, out)) = setup() else {
        eprintln!("SKIP: задайте FLUX_DIR и FLUX_SYN");
        return;
    };
    let p = params(11);
    let t = std::time::Instant::now();
    let (_, lat_dir, img_dir) = txt2img(&dir, &p);
    eprintln!("каталог: {:.1} с", t.elapsed().as_secs_f32());
    let t = std::time::Instant::now();
    let (_, lat_syn, img_syn) = txt2img(&syn, &p);
    eprintln!("бандл: {:.1} с", t.elapsed().as_secs_f32());
    assert_eq!(img_syn.dims(), &[3, 384, 512]);
    assert!(bits(&lat_dir) == bits(&lat_syn), "латенты расходятся");
    assert!(bits(&img_dir) == bits(&img_syn), "картинки расходятся");
    synaptix_io::image::save_image(&img_syn, out.join("bundle_txt2img.png")).unwrap();
}

#[test]
#[ignore]
fn img2img_keeps_size_and_cancel_stops() {
    let Some((_, syn, out)) = setup() else {
        eprintln!("SKIP: задайте FLUX_SYN");
        return;
    };
    let p = params(5);
    let (m, _, img) = txt2img(&syn, &p);

    // Латент картинки: VAE-энкодер → тот же размер латента, что и у генерации.
    let x0 = m.encode_image(&img).unwrap();
    assert_eq!(x0.dims(), &[1, 16, 48, 64]);
    // Кодирование + декодирование без денойза — почти та же картинка.
    let round = m.decode(&x0).unwrap();
    let (a, b) = (bits(&img), bits(&round));
    let mae: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs() as f64)
        .sum::<f64>()
        / a.len() as f64;
    eprintln!("VAE round-trip MAE = {mae:.4}");
    assert!(mae < 0.03, "VAE encode→decode слишком далеко: {mae}");

    let seq = m.default_max_seq_len();
    let cond = m
        .encode_prompt("the same lighthouse in a snowstorm at night, photo", seq)
        .unwrap();
    let tr = m.load_transformer(FluxModel::tokens_for(p.width, p.height, seq)).unwrap();
    let mut steps = 0;
    let p2 = SampleParams { denoise: 0.6, seed: 9, ..p.clone() };
    let lat = m
        .sample(&tr, &cond, Some(&x0), &p2, &mut |i, n| {
            steps = n;
            let _ = i;
            true
        })
        .unwrap();
    // 8 шагов × 0.6 → int(8 − 4.8) = 3 → денойз с 3-го шага, 5 шагов.
    assert_eq!(steps, 5);
    let edited = m.decode(&lat).unwrap();
    assert_eq!(edited.dims(), &[3, 384, 512]);
    synaptix_io::image::save_image(&edited, out.join("bundle_img2img.png")).unwrap();

    let mut seen = 0;
    let r = m.sample(&tr, &cond, None, &p, &mut |i, _| {
        seen = i;
        i < 2
    });
    assert!(matches!(r, Err(FluxError::Cancelled)), "ожидалась отмена");
    assert_eq!(seen, 2);
}
