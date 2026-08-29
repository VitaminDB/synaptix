use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

const CAP: usize = 256;
const RATIO: usize = 4;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.5
        })
        .collect()
}

fn ready() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .and_then(|x| x.to_dtype(DType::F32))
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .expect("на хост")
}

fn quantized(src: &Tensor, nkv: usize, d: usize, device: Device) -> (Tensor, Tensor) {
    let mut packed = Tensor::zeros(vec![1, nkv, CAP, d], DType::MXFP8, device).expect("packed");
    let mut scale = Tensor::zeros(vec![1, nkv, CAP, d / 32], DType::U8, device).expect("scale");
    let src16 = src.to_dtype(DType::F16).expect("f16 для кванта");
    packed.kv_append_quant_mxfp8_inplace(&mut scale, &src16, 0).expect("квант-append");
    (packed, scale)
}

fn check(dtype: DType, nh: usize, nkv: usize, d: usize, tol: f32) {
    let device = Device::Cuda(0);
    let b = 5usize;
    let nb = 6usize;

    let q = Tensor::from_vec::<_, f32>(noise(1, b * nh * d), vec![b, nh, d], Device::Cpu)
        .and_then(|t| t.to_dtype(dtype))
        .and_then(|t| t.to_device(device))
        .expect("q");
    let dense = |seed: u64| {
        Tensor::from_vec::<_, f32>(noise(seed, nkv * CAP * d), vec![1, nkv, CAP, d], Device::Cpu)
            .and_then(|t| t.to_dtype(dtype))
            .and_then(|t| t.to_device(device))
            .expect("kv")
    };
    let k = dense(2);
    let v = dense(3);

    let (k8, ks) = quantized(&k, nkv, d, device);
    let (v8, vs) = quantized(&v, nkv, d, device);

    let flat = |t: &Tensor, last: usize| t.reshape(vec![nkv, CAP, last]).expect("reshape");
    let k_ref = k8.mxfp8_dequant(&ks).and_then(|t| t.to_dtype(dtype)).expect("деквант k");
    let v_ref = v8.mxfp8_dequant(&vs).and_then(|t| t.to_dtype(dtype)).expect("деквант v");

    let rows: Vec<(Vec<u32>, (u32, u32))> = (0..b)
        .map(|i| {
            let mut blocks: Vec<u32> =
                (0..nb).map(|j| ((i * 7 + j * 5) % (CAP / RATIO)) as u32).collect();
            blocks.sort_unstable();
            blocks.dedup();
            while blocks.len() < nb {
                let next = (blocks.last().copied().unwrap_or(0) + 1) % (CAP / RATIO) as u32;
                if !blocks.contains(&next) {
                    blocks.push(next);
                }
                blocks.sort_unstable();
            }
            (blocks, ((i as u32 % 3) * RATIO as u32 + 40, 3))
        })
        .collect();

    let table: Vec<u32> = rows.iter().flat_map(|(b, _)| b.iter().copied()).collect();
    let table = Tensor::from_vec(table, vec![b, nb], device).expect("таблица");
    let tail_from: Vec<u32> = rows.iter().map(|(_, (f, _))| *f).collect();
    let tail_len: Vec<u32> = rows.iter().map(|(_, (_, l))| *l).collect();
    let tail_from = Tensor::from_vec(tail_from, vec![b], device).expect("хвост");
    let tail_len = Tensor::from_vec(tail_len, vec![b], device).expect("длина хвоста");
    let scale = 1.0 / (d as f32).sqrt();

    let want = q
        .flash_attention_blocks(
            &flat(&k_ref, d),
            &flat(&v_ref, d),
            &table,
            &tail_from,
            &tail_len,
            RATIO,
            scale,
            0,
        )
        .expect("эталон по деквантованному KV");
    let got = q
        .flash_attention_blocks_mxfp8(
            &flat(&k8, d),
            &flat(&v8, d),
            &flat(&ks, d / 32),
            &flat(&vs, d / 32),
            &table,
            &tail_from,
            &tail_len,
            RATIO,
            scale,
            0,
        )
        .expect("ядро mxfp8 по блокам");

    let want = host(&want);
    let got = host(&got);
    assert_eq!(got.len(), want.len());
    let num: f64 = got.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|x| (*x as f64).powi(2)).sum();
    let rel = (num / den.max(1e-12)).sqrt() as f32;
    eprintln!("{dtype:?} nh={nh} d={d}: mxfp8-блоки против деквантованных rel_l2={rel:.3e}");
    assert!(rel < tol, "{dtype:?}: разошлось на {rel:.3e}");
}

#[test]
fn mxfp8_blocks_match_dequantized_f16() {
    if !ready() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    check(DType::F16, 8, 2, 128, 4e-3);
}

#[test]
fn mxfp8_blocks_match_dequantized_wide_heads() {
    if !ready() {
        return;
    }
    check(DType::F16, 24, 2, 256, 4e-3);
}

#[test]
fn mxfp8_blocks_match_dequantized_f32() {
    if !ready() {
        return;
    }
    check(DType::F32, 8, 2, 128, 1e-5);
}
