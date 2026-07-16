use synaptix_asr_gigaam::GigaAm;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

fn load_wav(path: &str) -> (Vec<f32>, u32) {
    let r = hound::WavReader::open(path).expect("wav");
    let spec = r.spec();
    let ch = (spec.channels as usize).max(1);
    let inter: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.into_samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let mx = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / mx).collect()
        }
    };
    let mono: Vec<f32> = if ch <= 1 { inter } else { inter.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect() };
    (mono, spec.sample_rate)
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "tmp/gigaam_unpack".to_string());
    let wav = std::env::args().nth(2).unwrap_or_else(|| "extr.wav".to_string());
    let (pcm, sr) = load_wav(&wav);

    synaptix_kernels_cpu::ensure_registered();
    let cpu = GigaAm::from_unpacked(&dir, &Device::Cpu, DType::F32).expect("cpu load");
    let t = std::time::Instant::now();
    let cpu_text = cpu.transcribe_pcm(&pcm, sr).expect("cpu transcribe");
    eprintln!("[gigaam_gpu] CPU {:?}: {}", t.elapsed(), cpu_text);

    synaptix_kernels_cuda::ensure_registered();
    let t = std::time::Instant::now();
    let gpu = match GigaAm::from_unpacked(&dir, &Device::Cuda(0), DType::F32) {
        Ok(g) => g, Err(e) => { eprintln!("GPU load FAILED: {e}"); std::process::exit(1); }
    };
    eprintln!("[gigaam_gpu] GPU load {:?}", t.elapsed());
    let t = std::time::Instant::now();
    let gpu_text = match gpu.transcribe_pcm(&pcm, sr) { Ok(s) => s, Err(e) => { eprintln!("GPU transcribe FAILED: {e}"); std::process::exit(1); } };
    eprintln!("[gigaam_gpu] GPU {:?}: {}", t.elapsed(), gpu_text);
    eprintln!("[gigaam_gpu] CPU==GPU text: {}", cpu_text.trim() == gpu_text.trim());
}
