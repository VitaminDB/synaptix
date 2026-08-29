use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

const NH: usize = 24;
const NKV: usize = 2;
const D: usize = 256;
const CAP: usize = 8192;
const RATIO: usize = 4;
const NB: usize = 512;
const B: usize = 512;
const ITERS: usize = 20;

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

#[test]
#[ignore]
fn flash_blocks_dense_vs_mxfp8() {
    if !ready() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    let device = Device::Cuda(0);
    let dtype = DType::F16;
    let q = Tensor::from_vec::<_, f32>(noise(1, B * NH * D), vec![B, NH, D], Device::Cpu)
        .and_then(|t| t.to_dtype(dtype))
        .and_then(|t| t.to_device(device))
        .expect("q");
    let dense = |seed: u64| {
        Tensor::from_vec::<_, f32>(noise(seed, NKV * CAP * D), vec![1, NKV, CAP, D], Device::Cpu)
            .and_then(|t| t.to_dtype(dtype))
            .and_then(|t| t.to_device(device))
            .expect("kv")
    };
    let k = dense(2);
    let v = dense(3);
    let quantized = |src: &Tensor| {
        let mut packed = Tensor::zeros(vec![1, NKV, CAP, D], DType::MXFP8, device).expect("packed");
        let mut scale = Tensor::zeros(vec![1, NKV, CAP, D / 32], DType::U8, device).expect("scale");
        packed.kv_append_quant_mxfp8_inplace(&mut scale, src, 0).expect("квант");
        (packed, scale)
    };
    let (k8, ks) = quantized(&k);
    let (v8, vs) = quantized(&v);
    let flat = |t: &Tensor, last: usize| t.reshape(vec![NKV, CAP, last]).expect("reshape");

    let table: Vec<u32> = (0..B)
        .flat_map(|i| (0..NB).map(move |j| (((i * 7 + j * 13) % (CAP / RATIO)) as u32)))
        .collect();
    let table = Tensor::from_vec(table, vec![B, NB], device).expect("таблица");
    let tail_from = Tensor::from_vec(vec![0u32; B], vec![B], device).expect("хвост");
    let tail_len = Tensor::from_vec(vec![0u32; B], vec![B], device).expect("длина хвоста");
    let scale = 1.0 / (D as f32).sqrt();
    let stream = synaptix_core::device::cuda::default_stream(0).expect("поток");

    let bytes_dense = (B * NKV * NB * RATIO * D * 2 * 2) as f64;
    let mut run = |name: &str, quant: bool, bytes: f64| {
        for _ in 0..3 {
            let _ = if quant {
                q.flash_attention_blocks_mxfp8(
                    &flat(&k8, D), &flat(&v8, D), &flat(&ks, D / 32), &flat(&vs, D / 32),
                    &table, &tail_from, &tail_len, RATIO, scale, 0,
                )
            } else {
                q.flash_attention_blocks(
                    &flat(&k, D), &flat(&v, D), &table, &tail_from, &tail_len, RATIO, scale, 0,
                )
            }
            .expect("прогон");
        }
        stream.synchronize().expect("синк");
        let t = std::time::Instant::now();
        for _ in 0..ITERS {
            let out = if quant {
                q.flash_attention_blocks_mxfp8(
                    &flat(&k8, D), &flat(&v8, D), &flat(&ks, D / 32), &flat(&vs, D / 32),
                    &table, &tail_from, &tail_len, RATIO, scale, 0,
                )
            } else {
                q.flash_attention_blocks(
                    &flat(&k, D), &flat(&v, D), &table, &tail_from, &tail_len, RATIO, scale, 0,
                )
            }
            .expect("прогон");
            std::hint::black_box(&out);
        }
        stream.synchronize().expect("синк");
        let el = t.elapsed().as_secs_f64() / ITERS as f64;
        eprintln!("{name}: {:.2} мс, {:.0} ГБ/с по KV", el * 1e3, bytes / el / 1e9);
    };
    run("плотный f16", false, bytes_dense);
    run("mxfp8", true, bytes_dense / 2.0);
}
