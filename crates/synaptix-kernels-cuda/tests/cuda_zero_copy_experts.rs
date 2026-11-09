//! Эксперты прямо из закреплённой RAM: индексный GEMV читает вес по адресу
//! хостовой памяти (единая адресация, cuMemHostAlloc) — без копии на карту и
//! без синхронизации с хостом. Результат обязан совпасть с весом в VRAM.
use std::sync::Arc;

use half::f16;
use synaptix_core::dtype::DType;
use synaptix_core::quant::ggml::{quantize_iq3_s128, GgmlType};
use synaptix_kernels_cuda::elementwise::blockq::{blockq_gemv_indexed, BlockqGemvKernels};

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05
        })
        .collect()
}

#[test]
fn indexed_gemv_reads_pinned_host_weights() {
    let Ok(ctx) = synaptix_core::device::cuda::get(0) else { return };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let (n, k, e) = (1280usize, 2560usize, 10usize);
    let dt = DType::Ggml(GgmlType::Iq3S128);
    let blob = quantize_iq3_s128(&noise(3, n * k), n, k, None);
    let bytes = blob.len();
    // e экспертов подряд: на карте и в закреплённой RAM.
    let mut host = synaptix_core::memory::pinned::PinnedBuf::new_uninit(bytes * e);
    assert!(host.is_pinned(), "нужна page-locked память");
    for i in 0..e {
        host.as_mut_slice()[i * bytes..(i + 1) * bytes].copy_from_slice(&blob);
    }
    let dev: cudarc::driver::CudaSlice<u8> = stream.clone_htod(host.as_slice()).unwrap();
    let dev_base = {
        use cudarc::driver::DevicePtr;
        let (p, _r) = dev.device_ptr(&stream);
        p
    };
    let host_base = host.as_ptr() as u64;
    let x: Vec<f16> = noise(9, k).iter().map(|v| f16::from_f32(*v * 20.0)).collect();
    let xd = stream.clone_htod(&x).unwrap();
    let idx: Vec<u32> = (0..e as u32).collect();
    let idxd = stream.clone_htod(&idx).unwrap();
    let kernels: Arc<BlockqGemvKernels> = BlockqGemvKernels::for_context(&ctx).unwrap();
    let run = |base: u64| -> (Vec<f16>, f64) {
        let tab: Vec<u64> = (0..e).map(|i| base + (i * bytes) as u64).collect();
        let tabd = stream.clone_htod(&tab).unwrap();
        let mut out = stream.alloc_zeros::<u8>(e * n * 2).unwrap();
        let xv = unsafe { xd.transmute::<u8>(k * 2).unwrap() };
        let go = |out: &mut cudarc::driver::CudaSlice<u8>| {
            let mut ov = out.as_view_mut();
            blockq_gemv_indexed(&kernels, &stream, dt, &tabd.as_view(), &tabd.as_view(), &idxd.as_view(), &xv, &mut ov, e as u32, e as u32, n as u32, k as u32, false).unwrap();
        };
        go(&mut out);
        stream.synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..20 {
            go(&mut out);
        }
        stream.synchronize().unwrap();
        let dt = t.elapsed().as_secs_f64() / 20.0;
        let o: Vec<u8> = stream.clone_dtoh(&out).unwrap();
        (bytemuck::cast_slice::<u8, f16>(&o).to_vec(), dt)
    };
    let (vram, t_v) = run(dev_base);
    let (pinned, t_h) = run(host_base);
    assert_eq!(vram, pinned, "из RAM и из VRAM — разные результаты");
    let gb = (bytes * e) as f64 / 1e9;
    eprintln!(
        "10 экспертов gate_up ({:.1} МБ): VRAM {:.0} мкс ({:.0} ГБ/с), закреплённая RAM {:.0} мкс ({:.1} ГБ/с)",
        gb * 1e3,
        t_v * 1e6,
        gb / t_v,
        t_h * 1e6,
        gb / t_h
    );
}

/// Потолок чтения закреплённой RAM ядром: копия широкими (16 байт)
/// коалесцированными загрузками в VRAM.
#[test]
fn pinned_read_bandwidth_wide_loads() {
    let Ok(ctx) = synaptix_core::device::cuda::get(0) else { return };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let src = r#"
extern "C" __global__ void copy16(const uint4* __restrict__ s, uint4* __restrict__ d, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x) d[i] = s[i];
}"#;
    let module = synaptix_kernels_cuda::kernels::compile::compile_module(&ctx, src, "copy16.cu").unwrap();
    let f = synaptix_kernels_cuda::kernels::compile::load_fn(&module, "copy16").unwrap();
    let bytes = 64usize << 20;
    let mut host = synaptix_core::memory::pinned::PinnedBuf::new(bytes);
    host.as_mut_slice()[123] = 7;
    let dev = stream.alloc_zeros::<u8>(bytes).unwrap();
    let n = (bytes / 16) as u64;
    let hp = host.as_ptr() as u64;
    let dp = {
        use cudarc::driver::DevicePtr;
        dev.device_ptr(&stream).0
    };
    use cudarc::driver::PushKernelArg;
    for blocks in [256u32, 1024, 4096] {
        let go = || {
            let mut b = stream.launch_builder(&f);
            b.arg(&hp).arg(&dp).arg(&n);
            unsafe { b.launch(cudarc::driver::LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }.unwrap();
        };
        go();
        stream.synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..10 {
            go();
        }
        stream.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64() / 10.0;
        eprintln!("ядро-копия RAM→VRAM, {blocks} блоков: {:.1} ГБ/с", bytes as f64 / s / 1e9);
    }
    let t = std::time::Instant::now();
    for _ in 0..10 {
        unsafe { cudarc::driver::result::memcpy_htod_async(dp, host.as_slice(), stream.cu_stream()) }.unwrap();
    }
    stream.synchronize().unwrap();
    eprintln!("DMA cuMemcpyHtoD: {:.1} ГБ/с", bytes as f64 * 10.0 / t.elapsed().as_secs_f64() / 1e9);
    let back: Vec<u8> = stream.clone_dtoh(&dev).unwrap();
    assert_eq!(back[123], 7);
}
