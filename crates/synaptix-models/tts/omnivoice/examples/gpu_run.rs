use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_omnivoice::config::OmniVoiceGenerationConfig;
use synaptix_tts_omnivoice::pipeline::OmniVoicePipeline;

fn main() {
    synaptix_kernels_cuda::ensure_registered();

    let bundle = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "storage/syn_models/omnivoice.syn".to_string());
    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Привет! Это полный прогон OmniVoice на видеокарте.".to_string());

    let device = Device::Cuda(0);
    let dtype = DType::F32;

    let t_load = std::time::Instant::now();
    let pl = match OmniVoicePipeline::from_syn(&bundle, device, dtype) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("from_syn FAILED: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("[gpu_run] from_syn OK in {:?}", t_load.elapsed());

    // Full quality: num_step=32. position/class_temperature=0 → greedy (реализованный
    // сверенный путь; gumbel-сэмплинг для non-greedy пока не портирован).
    let gen = OmniVoiceGenerationConfig {
        num_step: 32,
        guidance_scale: 2.0,
        t_shift: 0.1,
        position_temperature: 0.0,
        class_temperature: 0.0,
        ..OmniVoiceGenerationConfig::default()
    };

    let target = pl.estimate_target_tokens(&text);
    eprintln!("[gpu_run] target_tokens={target} num_step={}", gen.num_step);

    let t_gen = std::time::Instant::now();
    let wav = match pl.generate(&text, &gen) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("generate FAILED: {e}");
            std::process::exit(1);
        }
    };
    let dt = t_gen.elapsed();

    let n = wav.len();
    let rms = (wav.iter().map(|x| x * x).sum::<f32>() / n.max(1) as f32).sqrt();
    let peak = wav.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let finite = wav.iter().all(|x| x.is_finite());
    let dur = n as f32 / 24000.0;
    eprintln!(
        "[gpu_run] generate {:?} | samples={n} dur={dur:.2}s rms={rms:.4} peak={peak:.4} finite={finite} | RTF={:.3}",
        dt,
        dt.as_secs_f32() / dur.max(1e-6)
    );

    let out = "/tmp/omni_gpu_q32.wav";
    if let Err(e) = synaptix_audio::write_wav_mono_f32(out, &wav, 24000) {
        eprintln!("write_wav FAILED: {e}");
    } else {
        eprintln!("[gpu_run] wrote {out}");
    }
}
