#![cfg(feature = "cuda")]

use half::{bf16, f16};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::elementwise::kv_append::{
    append_bf16, append_bf16_dev, append_f16, append_f16_dev, append_f32, append_f32_dev,
    KvAppendKernels,
};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f16(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f16::from_f32(f * scale)
        })
        .collect()
}

fn det_bf16(seed: u64, n: usize, scale: f32) -> Vec<bf16> {
    det_f16(seed, n, scale)
        .iter()
        .map(|v| bf16::from_f32(v.to_f32()))
        .collect()
}

fn det_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    det_f16(seed, n, scale).iter().map(|v| v.to_f32()).collect()
}

#[test]
fn kv_append_f16_basic() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = KvAppendKernels::for_context(&ctx).expect("compile kv_append");

    let b = 2u32;
    let kv = 4u32;
    let t_new = 3u32;
    let hd = 64u32;
    let max_seq = 32u32;
    let seq_pos = 5u32;
    let src_host = det_f16(0xA110_C8E1, (b * kv * t_new * hd) as usize, 0.5);
    let dst_init = det_f16(0xDEAD_BEEF, (b * kv * max_seq * hd) as usize, 0.3);

    let src_dev: CudaSlice<f16> = stream.clone_htod(&src_host).unwrap();
    let mut dst_dev: CudaSlice<f16> = stream.clone_htod(&dst_init).unwrap();

    append_f16(
        &kernels,
        &stream,
        &src_dev,
        &mut dst_dev,
        b,
        kv,
        t_new,
        hd,
        max_seq,
        seq_pos,
    )
    .expect("append f16");
    stream.synchronize().unwrap();

    let dst_host: Vec<f16> = stream.clone_dtoh(&dst_dev).unwrap();
    // Проверяем, что src попал точно в dst[..][..][seq_pos..seq_pos+t_new][..]
    for bi in 0..b as usize {
        for ki in 0..kv as usize {
            for ti in 0..t_new as usize {
                for di in 0..hd as usize {
                    let src_off =
                        ((bi * kv as usize + ki) * t_new as usize + ti) * hd as usize + di;
                    let dst_off =
                        ((bi * kv as usize + ki) * max_seq as usize + seq_pos as usize + ti)
                            * hd as usize
                            + di;
                    assert_eq!(
                        dst_host[dst_off].to_bits(),
                        src_host[src_off].to_bits(),
                        "mismatch at b={bi} k={ki} t={ti} d={di}"
                    );
                }
            }
        }
    }
    // Проверяем, что bytes вне scatter не изменились.
    for bi in 0..b as usize {
        for ki in 0..kv as usize {
            for ti in 0..max_seq as usize {
                if ti >= seq_pos as usize && ti < seq_pos as usize + t_new as usize {
                    continue;
                }
                for di in 0..hd as usize {
                    let off = ((bi * kv as usize + ki) * max_seq as usize + ti) * hd as usize + di;
                    assert_eq!(
                        dst_host[off].to_bits(),
                        dst_init[off].to_bits(),
                        "modified outside slice at b={bi} k={ki} t={ti}"
                    );
                }
            }
        }
    }
}

#[test]
fn kv_append_bf16_basic() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = KvAppendKernels::for_context(&ctx).expect("compile kv_append");
    let b = 1u32;
    let kv = 2u32;
    let t_new = 4u32;
    let hd = 128u32;
    let max_seq = 16u32;
    let seq_pos = 6u32;
    let src_host = det_bf16(0xC0DE, (b * kv * t_new * hd) as usize, 0.5);
    let dst_init = det_bf16(0xBABE, (b * kv * max_seq * hd) as usize, 0.3);

    let src_dev: CudaSlice<bf16> = stream.clone_htod(&src_host).unwrap();
    let mut dst_dev: CudaSlice<bf16> = stream.clone_htod(&dst_init).unwrap();
    append_bf16(
        &kernels,
        &stream,
        &src_dev,
        &mut dst_dev,
        b,
        kv,
        t_new,
        hd,
        max_seq,
        seq_pos,
    )
    .expect("append bf16");
    stream.synchronize().unwrap();
    let dst_host: Vec<bf16> = stream.clone_dtoh(&dst_dev).unwrap();
    for bi in 0..b as usize {
        for ki in 0..kv as usize {
            for ti in 0..t_new as usize {
                for di in 0..hd as usize {
                    let src_off =
                        ((bi * kv as usize + ki) * t_new as usize + ti) * hd as usize + di;
                    let dst_off =
                        ((bi * kv as usize + ki) * max_seq as usize + seq_pos as usize + ti)
                            * hd as usize
                            + di;
                    assert_eq!(dst_host[dst_off].to_bits(), src_host[src_off].to_bits());
                }
            }
        }
    }
}

#[test]
fn kv_append_f32_basic() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = KvAppendKernels::for_context(&ctx).expect("compile kv_append");
    let b = 1u32;
    let kv = 1u32;
    let t_new = 2u32;
    let hd = 33u32;
    let max_seq = 8u32;
    let seq_pos = 3u32;
    let src_host = det_f32(0x1234, (b * kv * t_new * hd) as usize, 1.0);
    let dst_init = det_f32(0x5678, (b * kv * max_seq * hd) as usize, 1.0);

    let src_dev: CudaSlice<f32> = stream.clone_htod(&src_host).unwrap();
    let mut dst_dev: CudaSlice<f32> = stream.clone_htod(&dst_init).unwrap();
    append_f32(
        &kernels,
        &stream,
        &src_dev,
        &mut dst_dev,
        b,
        kv,
        t_new,
        hd,
        max_seq,
        seq_pos,
    )
    .expect("append f32");
    stream.synchronize().unwrap();
    let dst_host: Vec<f32> = stream.clone_dtoh(&dst_dev).unwrap();
    for ti in 0..t_new as usize {
        for di in 0..hd as usize {
            let src_off = ti * hd as usize + di;
            let dst_off = (seq_pos as usize + ti) * hd as usize + di;
            assert_eq!(dst_host[dst_off].to_bits(), src_host[src_off].to_bits());
        }
    }
}

#[test]
fn kv_append_f16_dev_pos() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = KvAppendKernels::for_context(&ctx).expect("compile kv_append");
    let b = 1u32;
    let kv = 2u32;
    let t_new = 2u32;
    let hd = 64u32;
    let max_seq = 16u32;
    let seq_pos = 7u32;
    let src_host = det_f16(0xA110, (b * kv * t_new * hd) as usize, 0.4);
    let dst_init = vec![f16::ZERO; (b * kv * max_seq * hd) as usize];

    let src_dev: CudaSlice<f16> = stream.clone_htod(&src_host).unwrap();
    let mut dst_dev: CudaSlice<f16> = stream.clone_htod(&dst_init).unwrap();
    let pos_dev: CudaSlice<u32> = stream.clone_htod(&[seq_pos]).unwrap();
    append_f16_dev(
        &kernels,
        &stream,
        &src_dev,
        &mut dst_dev,
        b,
        kv,
        t_new,
        hd,
        max_seq,
        &pos_dev,
    )
    .expect("append f16 dev");
    stream.synchronize().unwrap();
    let dst_host: Vec<f16> = stream.clone_dtoh(&dst_dev).unwrap();
    for bi in 0..b as usize {
        for ki in 0..kv as usize {
            for ti in 0..t_new as usize {
                for di in 0..hd as usize {
                    let src_off =
                        ((bi * kv as usize + ki) * t_new as usize + ti) * hd as usize + di;
                    let dst_off =
                        ((bi * kv as usize + ki) * max_seq as usize + seq_pos as usize + ti)
                            * hd as usize
                            + di;
                    assert_eq!(dst_host[dst_off].to_bits(), src_host[src_off].to_bits());
                }
            }
        }
    }
}

#[test]
fn kv_append_bf16_dev_oob_guard() {
    // seq_pos такой, что часть строк попадёт за max_seq_len → kernel должен
    // silently skip эти rows (Phase E.7 bounds-check).
    let Some((ctx, stream)) = setup() else { return };
    let kernels = KvAppendKernels::for_context(&ctx).expect("compile kv_append");
    let b = 1u32;
    let kv = 1u32;
    let t_new = 4u32;
    let hd = 64u32;
    let max_seq = 8u32;
    let seq_pos = 6u32; // 6+0=6, 6+1=7, 6+2=8 (OOB), 6+3=9 (OOB)
    let src_host = det_bf16(0xC0DE, (b * kv * t_new * hd) as usize, 0.4);
    let canary = bf16::from_f32(99.0);
    let dst_init = vec![canary; (b * kv * max_seq * hd) as usize];

    let src_dev: CudaSlice<bf16> = stream.clone_htod(&src_host).unwrap();
    let mut dst_dev: CudaSlice<bf16> = stream.clone_htod(&dst_init).unwrap();
    let pos_dev: CudaSlice<u32> = stream.clone_htod(&[seq_pos]).unwrap();
    append_bf16_dev(
        &kernels,
        &stream,
        &src_dev,
        &mut dst_dev,
        b,
        kv,
        t_new,
        hd,
        max_seq,
        &pos_dev,
    )
    .expect("append bf16 dev");
    stream.synchronize().unwrap();
    let dst_host: Vec<bf16> = stream.clone_dtoh(&dst_dev).unwrap();
    // ti=0 (pos=6) и ti=1 (pos=7) скопированы.
    for ti in 0..2usize {
        for di in 0..hd as usize {
            let src_off = ti * hd as usize + di;
            let dst_off = (seq_pos as usize + ti) * hd as usize + di;
            assert_eq!(dst_host[dst_off].to_bits(), src_host[src_off].to_bits());
        }
    }
    // dst rows pos=8,9,... не существуют — это просто за пределами буфера.
    // Buffer rows [0..6] остались canary (не трогали).
    for ti in 0..(seq_pos as usize) {
        for di in 0..hd as usize {
            let off = ti * hd as usize + di;
            assert_eq!(
                dst_host[off].to_bits(),
                canary.to_bits(),
                "canary clobbered at ti={ti}"
            );
        }
    }
}

#[test]
fn kv_append_f32_dev() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = KvAppendKernels::for_context(&ctx).expect("compile kv_append");
    let b = 1u32;
    let kv = 1u32;
    let t_new = 3u32;
    let hd = 16u32;
    let max_seq = 12u32;
    let seq_pos = 4u32;
    let src_host = det_f32(0x1234, (b * kv * t_new * hd) as usize, 1.0);
    let dst_init = vec![0.0_f32; (b * kv * max_seq * hd) as usize];

    let src_dev: CudaSlice<f32> = stream.clone_htod(&src_host).unwrap();
    let mut dst_dev: CudaSlice<f32> = stream.clone_htod(&dst_init).unwrap();
    let pos_dev: CudaSlice<u32> = stream.clone_htod(&[seq_pos]).unwrap();
    append_f32_dev(
        &kernels,
        &stream,
        &src_dev,
        &mut dst_dev,
        b,
        kv,
        t_new,
        hd,
        max_seq,
        &pos_dev,
    )
    .expect("append f32 dev");
    stream.synchronize().unwrap();
    let dst_host: Vec<f32> = stream.clone_dtoh(&dst_dev).unwrap();
    for ti in 0..t_new as usize {
        for di in 0..hd as usize {
            let src_off = ti * hd as usize + di;
            let dst_off = (seq_pos as usize + ti) * hd as usize + di;
            assert_eq!(dst_host[dst_off].to_bits(), src_host[src_off].to_bits());
        }
    }
}
