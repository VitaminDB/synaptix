use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_omnivoice::pipeline::OmniVoicePipeline;
fn main() {
    synaptix_kernels_cpu::ensure_registered();
    let p = std::env::args().nth(1).expect("usage: from_syn_smoke <bundle.syn>");
    let t0 = std::time::Instant::now();
    match OmniVoicePipeline::from_syn(&p, Device::Cpu, DType::F32) {
        Ok(pl) => println!("from_syn OK in {:?}; frame_rate={}", t0.elapsed(), pl.frame_rate()),
        Err(e) => { eprintln!("from_syn FAILED: {e}"); std::process::exit(1); }
    }
}
