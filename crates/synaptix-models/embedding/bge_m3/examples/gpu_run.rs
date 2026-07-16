use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_embedding_bge_m3::pipeline::BgeM3;

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-9)
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "tmp/bge_unpack".to_string());
    let texts = ["Привет, это тест эмбеддинга BGE-M3.", "Second sentence for batch.", "Третье предложение."];
    let tv: Vec<&str> = texts.to_vec();

    synaptix_kernels_cpu::ensure_registered();
    let cpu = BgeM3::from_unpacked(&dir, &Device::Cpu, DType::F32).expect("cpu load");
    let t = std::time::Instant::now();
    let cpu_emb = cpu.encode(&tv).expect("cpu encode");
    eprintln!("[bge_gpu] CPU encode {} texts in {:?}", tv.len(), t.elapsed());

    synaptix_kernels_cuda::ensure_registered();
    let t = std::time::Instant::now();
    let gpu = match BgeM3::from_unpacked(&dir, &Device::Cuda(0), DType::F32) {
        Ok(g) => g, Err(e) => { eprintln!("GPU load FAILED: {e}"); std::process::exit(1); }
    };
    eprintln!("[bge_gpu] GPU load {:?}", t.elapsed());
    let t = std::time::Instant::now();
    let gpu_emb = match gpu.encode(&tv) { Ok(e) => e, Err(e) => { eprintln!("GPU encode FAILED: {e}"); std::process::exit(1); } };
    eprintln!("[bge_gpu] GPU encode {} texts in {:?}", tv.len(), t.elapsed());

    for (i, (c, g)) in cpu_emb.iter().zip(&gpu_emb).enumerate() {
        eprintln!("[bge_gpu] text{i}: dim={} cos(cpu,gpu)={:.6} finite={}", g.len(), cos(c, g), g.iter().all(|x| x.is_finite()));
    }
}
