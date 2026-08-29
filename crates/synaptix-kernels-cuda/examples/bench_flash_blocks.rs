//! Разреженное внимание: сборка позиций гатером против чтения по таблице блоков.
//!
//! Формы взяты у QSA Qwen4Exp на длинном контексте: 16 голов запроса на 2
//! kv-головы, голова 128, бюджет индексатора 512 блоков по 4 позиции.

use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

const NH: usize = 24;
const NKV: usize = 2;
const D: usize = 256;
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

fn sync() {
    let s = synaptix_core::device::cuda::default_stream(0).expect("stream");
    s.synchronize().expect("sync");
}

fn bench(cap: usize, g: usize, nb: usize, iters: usize) {
    let device = Device::Cuda(0);
    let dtype = DType::F16;
    let q = Tensor::from_vec::<_, f32>(noise(1, g * NH * D), vec![g, NH, D], Device::Cpu)
        .and_then(|t| t.to_dtype(dtype))
        .and_then(|t| t.to_device(device))
        .expect("q");
    let k = Tensor::from_vec::<_, f32>(noise(2, NKV * cap * D), vec![NKV, cap, D], Device::Cpu)
        .and_then(|t| t.to_dtype(dtype))
        .and_then(|t| t.to_device(device))
        .expect("k");
    let v = Tensor::from_vec::<_, f32>(noise(3, NKV * cap * D), vec![NKV, cap, D], Device::Cpu)
        .and_then(|t| t.to_dtype(dtype))
        .and_then(|t| t.to_device(device))
        .expect("v");

    let blocks_total = cap / RATIO;
    let table_host: Vec<u32> = (0..g)
        .flat_map(|i| (0..nb).map(move |j| ((i * 131 + j * 7919) % blocks_total) as u32))
        .collect();
    let table = Tensor::from_vec(table_host.clone(), vec![g, nb], device).expect("таблица");
    let tail_from = Tensor::from_vec(vec![0u32; g], vec![g], device).expect("хвост");
    let tail_len = Tensor::from_vec(vec![0u32; g], vec![g], device).expect("длина");
    let scale = 1.0 / (D as f32).sqrt();

    // Путь через сборку: индексы блоков → embed-гатер → flash.
    let block_rows = cap / RATIO;
    let mut idx = Vec::with_capacity(g * NKV * nb);
    for i in 0..g {
        for head in 0..NKV {
            let base = (head * block_rows) as u32;
            idx.extend(table_host[i * nb..(i + 1) * nb].iter().map(|b| base + *b));
        }
    }
    let idx = Tensor::from_vec(idx, vec![g * NKV * nb], device).expect("индексы");

    let gathered = || {
        let gather = |src: &Tensor| {
            src.reshape(vec![NKV * block_rows, RATIO * D])
                .and_then(|t| t.embed_gather(&idx))
                .and_then(|t| t.reshape(vec![g, NKV, nb * RATIO, D]))
                .expect("гатер")
        };
        let k_sel = gather(&k);
        let v_sel = gather(&v);
        let q4 = q.reshape(vec![g, NH, 1, D]).expect("q");
        q4.flash_attention(&k_sel, &v_sel, scale, false).expect("flash")
    };
    let by_blocks = || {
        q.flash_attention_blocks(&k, &v, &table, &tail_from, &tail_len, RATIO, scale, 0)
            .expect("ядро по блокам")
    };

    for _ in 0..3 {
        let _ = gathered();
        let _ = by_blocks();
    }
    sync();

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = gathered();
    }
    sync();
    let gather_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = by_blocks();
    }
    sync();
    let blocks_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let gather_only = || {
        let gather = |src: &Tensor| {
            src.reshape(vec![NKV * block_rows, RATIO * D])
                .and_then(|t| t.embed_gather(&idx))
                .expect("гатер")
        };
        (gather(&k), gather(&v))
    };
    for _ in 0..3 {
        let _ = gather_only();
    }
    sync();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = gather_only();
    }
    sync();
    let only_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let bytes = (g * NKV * nb * RATIO * D * 2 * 2) as f64;
    println!(
        "cap={cap:6} g={g:4} блоков={nb:4}: сборка+flash {gather_ms:7.3} мс (из них гатер {only_ms:6.3}), ядро {blocks_ms:7.3} мс ({:5.1} ГБ/с), выигрыш {:.2}×",
        bytes / blocks_ms * 1e-6,
        gather_ms / blocks_ms
    );
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        println!("CUDA-устройств нет");
        return;
    }
    bench(36864, 256, 512, 20);
    bench(36864, 64, 512, 20);
    bench(8192, 256, 512, 20);
    bench(36864, 256, 128, 20);
    bench(36864, 16, 512, 50);
    bench(36864, 4, 512, 50);
    bench(36864, 1, 512, 50);
}
