#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::embed::{
    embed_gather_bf16, embed_gather_f16, embed_gather_f32, EmbedKernels,
};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f * scale
        })
        .collect()
}

fn det_ids(seed: u64, n: usize, vocab: u32) -> Vec<u32> {
    let mut x = seed.wrapping_add(0x1234_5678_9ABC_DEF0);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as u32) % vocab
        })
        .collect()
}

fn cpu_embed(table: &[f32], ids: &[u32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; ids.len() * dim];
    for (t, &id) in ids.iter().enumerate() {
        let src = id as usize * dim;
        out[t * dim..t * dim + dim].copy_from_slice(&table[src..src + dim]);
    }
    out
}

#[test]
fn embed_gather_f32_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = EmbedKernels::for_context(&ctx).expect("compile embed");
    let vocab = 1000usize;
    let dim = 320usize;
    let n_ids = 64usize;
    let table = det_f32(0xE11, vocab * dim, 1.0);
    let ids = det_ids(0xD22, n_ids, vocab as u32);
    let expected = cpu_embed(&table, &ids, dim);

    let dev_t: CudaSlice<f32> = stream.clone_htod(&table).unwrap();
    let dev_ids: CudaSlice<u32> = stream.clone_htod(&ids).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(n_ids * dim).unwrap();
    embed_gather_f32(
        &kernels,
        &stream,
        &dev_t,
        &dev_ids,
        &mut dev_out,
        n_ids as u32,
        dim as u32,
        vocab as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();
    assert_eq!(got, expected, "embed f32 mismatch");
}

#[test]
fn embed_gather_f16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = EmbedKernels::for_context(&ctx).expect("compile embed");
    let vocab = 512usize;
    let dim = 128usize;
    let n_ids = 48usize;
    let table_f = det_f32(0xE33, vocab * dim, 1.0);
    let table: Vec<f16> = table_f.iter().map(|v| f16::from_f32(*v)).collect();
    let table_back: Vec<f32> = table.iter().map(|v| v.to_f32()).collect();
    let ids = det_ids(0xD44, n_ids, vocab as u32);
    let expected = cpu_embed(&table_back, &ids, dim);

    let dev_t: CudaSlice<f16> = stream.clone_htod(&table).unwrap();
    let dev_ids: CudaSlice<u32> = stream.clone_htod(&ids).unwrap();
    let mut dev_out: CudaSlice<f16> = stream.alloc_zeros(n_ids * dim).unwrap();
    embed_gather_f16(
        &kernels,
        &stream,
        &dev_t,
        &dev_ids,
        &mut dev_out,
        n_ids as u32,
        dim as u32,
        vocab as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_h: Vec<f16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();
    // gather = чистое копирование, должно быть бит-точно.
    assert_eq!(got, expected, "embed f16 must be exact (pure gather)");
}

#[test]
fn embed_gather_bf16_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = EmbedKernels::for_context(&ctx).expect("compile embed");
    let vocab = 512usize;
    let dim = 96usize;
    let n_ids = 40usize;
    let table_f = det_f32(0xE55, vocab * dim, 1.0);
    let table: Vec<bf16> = table_f.iter().map(|v| bf16::from_f32(*v)).collect();
    let table_back: Vec<f32> = table.iter().map(|v| v.to_f32()).collect();
    let ids = det_ids(0xD66, n_ids, vocab as u32);
    let expected = cpu_embed(&table_back, &ids, dim);

    let dev_t: CudaSlice<bf16> = stream.clone_htod(&table).unwrap();
    let dev_ids: CudaSlice<u32> = stream.clone_htod(&ids).unwrap();
    let mut dev_out: CudaSlice<bf16> = stream.alloc_zeros(n_ids * dim).unwrap();
    embed_gather_bf16(
        &kernels,
        &stream,
        &dev_t,
        &dev_ids,
        &mut dev_out,
        n_ids as u32,
        dim as u32,
        vocab as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_b: Vec<bf16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_b.iter().map(|v| v.to_f32()).collect();
    assert_eq!(got, expected, "embed bf16 must be exact (pure gather)");
}
