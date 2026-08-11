use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

#[test]
fn mxfp8_embed_gather_matches_dense() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let vocab = 512usize;
    let dim = 128usize;
    let mut host = vec![0f32; vocab * dim];
    for (i, x) in host.iter_mut().enumerate() {
        *x = ((i * 37) % 251) as f32 / 251.0 - 0.5;
    }
    let table = Tensor::from_vec(host.clone(), vec![vocab, dim], dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let qw = table.quantize_to_mxfp8().unwrap();

    let ids: Vec<u32> = vec![0, 1, 7, 63, 64, 65, 200, 511];
    let ids_t = Tensor::from_vec(ids.clone(), vec![ids.len()], dev).unwrap();

    let got = qw
        .embed_gather(&ids_t)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut max_rel = 0f32;
    let mut worst = (0usize, 0usize, 0f32, 0f32);
    for (row, id) in ids.iter().enumerate() {
        for c in 0..dim {
            let expect = host[(*id as usize) * dim + c];
            let actual = got[row * dim + c];
            let rel = (actual - expect).abs() / expect.abs().max(1e-3);
            if rel > max_rel {
                max_rel = rel;
                worst = (row, c, expect, actual);
            }
        }
    }
    println!(
        "max_rel={max_rel} worst row={} col={} expect={} actual={}",
        worst.0, worst.1, worst.2, worst.3
    );
    assert!(max_rel < 0.15, "MXFP8-embed расходится: max_rel={max_rel}");
}
