use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_infer::graph_capture::GraphCapturer;
use synaptix_infer::InferError;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");

    let a = Tensor::ones(vec![4usize, 8], DType::BF16, dev).unwrap();
    let b = Tensor::ones(vec![4usize, 8], DType::BF16, dev).unwrap();
    let mut out = Tensor::zeros(vec![4usize, 8], DType::BF16, dev).unwrap();

    for name in ["add only", "many contig", "huge alloc"] {
        let mut cap = GraphCapturer::new(1);
        let dst = &mut out;
        let res = cap.capture_with(&stream, |_s| {
            let t = match name {
                "with cat" => {
                    let c = Tensor::cat(&[&a, &b], 0)
                        .map_err(|e| InferError::Other(e.to_string()))?;
                    let lo = c.narrow(0, 0, 4).map_err(|e| InferError::Other(e.to_string()))?;
                    let hi = c.narrow(0, 4, 4).map_err(|e| InferError::Other(e.to_string()))?;
                    lo.add(&hi).map_err(|e| InferError::Other(e.to_string()))?
                }
                "many contig" => {
                    let mut acc = a.add(&b).map_err(|e| InferError::Other(e.to_string()))?;
                    for i in 0..300 {
                        let c = 64usize + (i % 16) * 64;
                        let t = Tensor::ones(vec![1usize, c, 8], DType::BF16, dev)
                            .map_err(|e| InferError::Other(e.to_string()))?;
                        let tr = t
                            .transpose(1, 2)
                            .and_then(|x| x.contiguous())
                            .map_err(|e| InferError::Other(e.to_string()))?;
                        let s = tr
                            .reshape(vec![c * 8])
                            .and_then(|x| x.narrow(0, 0, 32))
                            .and_then(|x| x.reshape(vec![4usize, 8]))
                            .and_then(|x| x.contiguous())
                            .map_err(|e| InferError::Other(e.to_string()))?;
                        acc = acc.add(&s).map_err(|e| InferError::Other(e.to_string()))?;
                    }
                    acc
                }
                "huge alloc" => {
                    let z = Tensor::ones(vec![2048usize, 4096], DType::BF16, dev)
                        .map_err(|e| InferError::Other(e.to_string()))?;
                    let s = z.narrow(0, 0, 4).and_then(|t| t.narrow(1, 0, 8))
                        .and_then(|t| t.contiguous())
                        .map_err(|e| InferError::Other(e.to_string()))?;
                    a.add(&b).and_then(|t| t.add(&s)).map_err(|e| InferError::Other(e.to_string()))?
                }
                "big zeros" => {
                    let z = Tensor::zeros(vec![64usize, 4096], DType::BF16, dev)
                        .map_err(|e| InferError::Other(e.to_string()))?;
                    let s = z.narrow(0, 0, 4).and_then(|t| t.narrow(1, 0, 8))
                        .and_then(|t| t.contiguous())
                        .map_err(|e| InferError::Other(e.to_string()))?;
                    a.add(&b).and_then(|t| t.add(&s)).map_err(|e| InferError::Other(e.to_string()))?
                }
                "many zeros" => {
                    let mut acc = a.add(&b).map_err(|e| InferError::Other(e.to_string()))?;
                    for i in 0..40 {
                        let z = Tensor::zeros(vec![8usize, 1024 + i], DType::BF16, dev)
                            .map_err(|e| InferError::Other(e.to_string()))?;
                        let s = z.narrow(0, 0, 4).and_then(|t| t.narrow(1, 0, 8))
                            .and_then(|t| t.contiguous())
                            .map_err(|e| InferError::Other(e.to_string()))?;
                        acc = acc.add(&s).map_err(|e| InferError::Other(e.to_string()))?;
                    }
                    acc
                }
                "with zeros" => {
                    let z = Tensor::zeros(vec![4usize, 8], DType::BF16, dev)
                        .map_err(|e| InferError::Other(e.to_string()))?;
                    a.add(&b)
                        .and_then(|t| t.add(&z))
                        .map_err(|e| InferError::Other(e.to_string()))?
                }
                _ => a.add(&b).map_err(|e| InferError::Other(e.to_string()))?,
            };
            dst.copy_from(&t).map_err(|e| InferError::Other(e.to_string()))?;
            Ok(())
        });
        match res {
            Ok(g) => {
                g.launch().expect("launch");
                stream.synchronize().expect("sync");
                let v = out
                    .to_dtype(DType::F32)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .unwrap();
                println!("{name}: ok, out[0]={}", v[0]);
            }
            Err(e) => println!("{name}: FAILED {e}"),
        }
    }
}
