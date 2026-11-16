//! Скорость чтения ядром закреплённой RAM (zero-copy хвост модели) против
//! видеопамяти и против DMA-копии: ширина загрузки и число потоков.
//!
//! `cargo run --profile fast-release -p synaptix-kernels-cuda --example bench_host_read`

use std::time::Instant;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use synaptix_kernels_cuda::kernels::compile::{compile_module, load_fn};

const SRC: &str = r#"
#define RD(NAME, T, XOR)                                                          \
extern "C" __global__ void NAME(const T* p, unsigned long long n, unsigned* out) { \
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x; \
    unsigned acc = 0;                                                             \
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) { \
        const T v = p[i];                                                         \
        acc += XOR;                                                               \
    }                                                                             \
    if (acc == 0x12345678u) out[0] = acc;                                         \
}
RD(rd16, uint4, (v.x ^ v.y ^ v.z ^ v.w))
RD(rd8, unsigned long long, (unsigned)(v ^ (v >> 32)))
RD(rd4, unsigned, v)
RD(rd1, unsigned char, v)
// Строки, как у GEMV: блок читает свой непрерывный кусок, поток — свою полосу.
extern "C" __global__ void rows16(const uint4* p, unsigned long long n, unsigned* out) {
    const unsigned long long per = n / gridDim.x;
    const uint4* q = p + per * blockIdx.x;
    unsigned acc = 0;
    for (unsigned long long i = threadIdx.x; i < per; i += blockDim.x) {
        const uint4 v = q[i];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;
}
"#;

fn main() {
    let gb: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(2);
    let n = gb << 30;
    let ctx = synaptix_core::device::cuda::get(0).expect("cuda");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    let host: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(31)).collect();
    let mapped = synaptix_core::memory::host_mapped::htod(&stream, &host).expect("закреплённая RAM");
    let mut vram = stream.alloc_zeros::<u8>(n).expect("vram");
    let mut out = stream.alloc_zeros::<u32>(4).expect("out");
    let module = compile_module(&ctx, SRC, "bench_host_read.cu").expect("compile");

    // DMA-копия из закреплённых страниц.
    let pinned = unsafe { std::slice::from_raw_parts({
        use cudarc::driver::DevicePtr;
        mapped.device_ptr(&stream).0 as *const u8
    }, n) };
    for _ in 0..2 {
        let t = Instant::now();
        stream.memcpy_htod(pinned, &mut vram).expect("htod");
        stream.synchronize().expect("sync");
        println!("DMA-копия RAM → VRAM: {:.1} ГБ/с", gb as f64 / t.elapsed().as_secs_f64());
    }

    for (name, width) in [("rd16", 16usize), ("rows16", 16), ("rd8", 8), ("rd4", 4), ("rd1", 1)] {
        let f = load_fn(&module, name).expect("fn");
        for blocks in [64u32, 256, 1024, 4096, 16384] {
            let mut line = format!("{name:>6} блоков {blocks:>5}:");
            for (label, buf) in [("RAM", &mapped), ("VRAM", &vram)] {
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let count = (n / width) as u64;
                    let t = Instant::now();
                    let mut b = stream.launch_builder(&f);
                    b.arg(buf);
                    b.arg(&count);
                    b.arg(&mut out);
                    unsafe {
                        b.launch(LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                    }
                    .expect("launch");
                    stream.synchronize().expect("sync");
                    best = best.min(t.elapsed().as_secs_f64());
                }
                line += &format!("  {label} {:>6.1} ГБ/с", gb as f64 / best);
            }
            println!("{line}");
        }
    }
}
