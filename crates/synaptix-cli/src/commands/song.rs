//! `synaptix song` — песня целиком: стиль и лирика → партитура → музыка →
//! 48 кГц стерео WAV.

use std::io::Write;
use std::path::PathBuf;

use synaptix_audio::io::write_wav_interleaved_f32;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_music_yue2::ar::Phase;
use synaptix_music_yue2::pipeline::{
    find_bundle, Callbacks, Yue2Options, Yue2Paths, Yue2Pipeline, MODEL_NAMES, VAE_NAMES,
};
use synaptix_music_yue2::protocol::{
    frames_to_seconds, seconds_to_frames, Cot, GenerationConfig, Sampling, SongRequest, CONTEXT,
};
use synaptix_music_yue2::vae::Yue2Vae;

use crate::commands::device;

pub struct SongArgs {
    pub style: String,
    pub lyrics: String,
    pub lyrics_file: Option<PathBuf>,
    pub output: PathBuf,
    pub models: PathBuf,
    pub model: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    /// `full` | `melody` | `off`.
    pub cot: String,
    /// Готовая партитура ABC вместо сгенерированной.
    pub abc_file: Option<PathBuf>,
    /// Кавер записи: SheetSage2 из того же бандла → мелодия → `cot = melody`.
    pub cover: Option<PathBuf>,
    /// Какие мелодии записи взять в кавер.
    pub cover_voices: String,
    pub cover_full: bool,
    pub cover_max_seconds: Option<f64>,
    pub seed: u64,
    pub cfg: Option<f32>,
    pub steps: usize,
    /// Потолок семантических токенов (25 на секунду музыки).
    pub max_tokens: Option<usize>,
    pub seconds: Option<f32>,
    pub min_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub penalty_window: Option<usize>,
    pub abc_max_tokens: Option<usize>,
    pub abc_min_tokens: Option<usize>,
    pub abc_temperature: Option<f32>,
    pub abc_top_p: Option<f32>,
    pub abc_top_k: Option<usize>,
    pub abc_repetition_penalty: Option<f32>,
    pub abc_penalty_window: Option<usize>,
    pub context: Option<usize>,
    pub device: String,
    pub compute_dtype: Option<String>,
    pub quant: Option<String>,
    pub vae_dtype: Option<String>,
    pub vae_core_frames: usize,
    pub vae_halo_frames: usize,
    /// Куда сохранить партитуру (`score.abc`).
    pub save_abc: Option<PathBuf>,
    pub save_latent: Option<PathBuf>,
}

pub struct SongDecodeArgs {
    pub latent: PathBuf,
    pub output: PathBuf,
    pub models: PathBuf,
    pub vae: Option<PathBuf>,
    pub device: String,
    pub vae_dtype: Option<String>,
    pub vae_core_frames: usize,
    pub vae_halo_frames: usize,
}

struct PhaseOverrides {
    max_tokens: Option<usize>,
    min_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    penalty_window: Option<usize>,
}

fn apply_sampling(s: &mut Sampling, o: PhaseOverrides, phase: &str) -> Result<(), String> {
    if let Some(v) = o.max_tokens {
        s.max_tokens = v;
        s.min_tokens = s.min_tokens.min(v);
    }
    if let Some(v) = o.min_tokens {
        s.min_tokens = v;
    }
    if let Some(v) = o.temperature {
        s.temperature = v;
    }
    if let Some(v) = o.top_p {
        s.top_p = v;
    }
    if let Some(v) = o.top_k {
        s.top_k = v;
    }
    if let Some(v) = o.repetition_penalty {
        s.repetition_penalty = v;
    }
    if let Some(v) = o.penalty_window {
        s.penalty_window = v;
    }
    s.validate().map_err(|e| format!("{phase}: {e}"))
}

pub fn save_latent(path: &std::path::Path, latents: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let dims = latents.dims().to_vec();
    let data: Vec<f32> = latents.to_dtype(DType::F32)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1()?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, dims, &bytes)?;
    safetensors::serialize_to_file([("latents", view)], None, path)?;
    Ok(())
}

fn load_latent(path: &std::path::Path) -> Result<Tensor, Box<dyn std::error::Error>> {
    let raw = std::fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&raw)?;
    let view = st.tensor("latents")?;
    if view.dtype() != safetensors::Dtype::F32 || view.shape().len() != 2 {
        return Err(format!("{}: ожидался F32-тензор latents [кадры, 64]", path.display()).into());
    }
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(Tensor::from_vec(data, view.shape().to_vec(), Device::Cpu)?)
}

fn resolve_vae(models: &std::path::Path, vae: Option<PathBuf>) -> Result<PathBuf, String> {
    match vae {
        Some(p) => Ok(p),
        None => find_bundle(models, VAE_NAMES).ok_or_else(|| {
            format!("в {} нет ни одного из {VAE_NAMES:?} — укажите --vae", models.display())
        }),
    }
}

fn parse_dtype(name: Option<&str>, default: DType) -> DType {
    match name {
        Some("f32") | Some("fp32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some("f16") | Some("fp16") => DType::F16,
        _ => default,
    }
}

fn parse_quant(name: Option<&str>) -> Option<DType> {
    match name {
        Some("nvfp4") => Some(DType::NVFP4),
        Some("mxfp8") => Some(DType::MXFP8),
        _ => None,
    }
}

pub fn run(args: SongArgs) -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = device::resolve(&args.device);

    let model = match args.model {
        Some(p) => p,
        None => find_bundle(&args.models, MODEL_NAMES).ok_or_else(|| {
            format!(
                "в {} нет ни одного из {MODEL_NAMES:?} — укажите --model",
                args.models.display()
            )
        })?,
    };
    let vae = resolve_vae(&args.models, args.vae.clone())?;

    let mut cot = Cot::parse(&args.cot).ok_or("--cot ожидает full | melody | off")?;
    let lyrics = match &args.lyrics_file {
        Some(p) => std::fs::read_to_string(p)?,
        None => args.lyrics.clone(),
    };
    let mut abc = match &args.abc_file {
        Some(p) => Some(std::fs::read_to_string(p)?),
        None => None,
    };
    if let Some(source) = &args.cover {
        // Мелодия без аккордов — аккомпанемент подстраивается под новый стиль.
        // SheetSage2 выгружается до загрузки YuE2: двум моделям ни к чему
        // делить карту.
        let options = synaptix_music_sheetsage2::TranscribeOptions {
            melody_only: !args.cover_full,
            voices: crate::commands::sheet::parse_voices(&args.cover_voices)?,
            max_seconds: args.cover_max_seconds,
            ..Default::default()
        };
        let compute = parse_dtype(args.compute_dtype.as_deref(), DType::BF16);
        let result = crate::commands::sheet::transcribe_file(&model, source, &args.device, compute, &options)?;
        let score = result
            .abc
            .ok_or_else(|| format!("мелодия записи не собрана: {}", result.abc_error.unwrap_or_default()))?;
        cot = if args.cover_full { Cot::Full } else { Cot::Melody };
        eprintln!(
            "[yue2] кавер: партитура записи ({} тактов), cot = {}",
            result.export.measures,
            cot.as_str()
        );
        abc = Some(score);
    }

    let mut generation = GenerationConfig::default();
    generation.ode_steps = args.steps.max(1);
    if let Some(v) = args.context {
        generation.context = v;
    }
    let semantic_max = args
        .max_tokens
        .or(args.seconds.map(|s| seconds_to_frames(s).clamp(1, CONTEXT - 2)));
    apply_sampling(
        &mut generation.semantic,
        PhaseOverrides {
            max_tokens: semantic_max,
            min_tokens: args.min_tokens,
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            repetition_penalty: args.repetition_penalty,
            penalty_window: args.penalty_window,
        },
        "музыка",
    )?;
    apply_sampling(
        &mut generation.abc,
        PhaseOverrides {
            max_tokens: args.abc_max_tokens,
            min_tokens: args.abc_min_tokens,
            temperature: args.abc_temperature,
            top_p: args.abc_top_p,
            top_k: args.abc_top_k,
            repetition_penalty: args.abc_repetition_penalty,
            penalty_window: args.abc_penalty_window,
        },
        "партитура",
    )?;

    let options = Yue2Options {
        device: dev,
        compute: parse_dtype(args.compute_dtype.as_deref(), DType::BF16),
        quant: parse_quant(args.quant.as_deref()),
        vae_dtype: parse_dtype(args.vae_dtype.as_deref(), DType::F32),
        vae_core_frames: args.vae_core_frames,
        vae_halo_frames: args.vae_halo_frames,
        generation,
    };
    eprintln!(
        "[yue2] модель {} | декодер {} | {:?} {:?} quant={:?}",
        model.display(),
        vae.display(),
        options.device,
        options.compute,
        options.quant
    );

    let paths = Yue2Paths { model, vae };
    let mut pipe = Yue2Pipeline::open(&paths, options)?;
    eprintln!("[yue2] загрузка: {:.1} с", pipe.load_seconds);

    let request = SongRequest {
        style: args.style,
        lyrics,
        cot,
        seed: args.seed,
        abc,
        cfg_scale: args.cfg,
    };

    // Прогресс по фазам: партитура печатается текстом по мере появления,
    // музыка — счётчиком секунд.
    let on_token = |phase: Phase, _token: u32, step: usize| {
        if step % 25 == 0 {
            let label = match phase {
                Phase::Abc => "партитура",
                Phase::Semantic => "музыка",
            };
            let extra = if phase == Phase::Semantic {
                format!(" ({:.1} с)", frames_to_seconds(step))
            } else {
                String::new()
            };
            eprint!("\r[yue2] {label}: {step} токенов{extra}    ");
            let _ = std::io::stderr().flush();
        }
    };
    let on_ode = |done: usize, total: usize| {
        eprint!("\r[yue2] flow matching: {done}/{total} шагов    ");
        let _ = std::io::stderr().flush();
    };
    let on_vae = |done: usize, total: usize| {
        eprint!("\r[yue2] декод: тайл {done}/{total}    ");
        let _ = std::io::stderr().flush();
    };
    let cb = Callbacks {
        cancel: None,
        on_token: Some(&on_token),
        on_ode: Some(&on_ode),
        on_vae: Some(&on_vae),
    };

    let song = pipe.run(&request, cb)?;
    eprintln!();

    if let Some(path) = &args.save_abc {
        if let Some(abc) = &song.semantic.plan.abc {
            std::fs::write(path, abc)?;
            eprintln!("[yue2] партитура → {}", path.display());
        }
    }
    if let Some(abc) = &song.semantic.plan.abc {
        eprintln!("[yue2] партитура ({} токенов):\n{abc}", song.semantic.plan.abc_ids.len());
    }
    if let Some(path) = &args.save_latent {
        save_latent(path, &song.latents)?;
        eprintln!("[yue2] латенты {:?} → {}", song.latents.dims(), path.display());
    }
    write_wav_interleaved_f32(&args.output, &song.audio, song.sample_rate, song.channels as u16)?;

    let t = &song.timing;
    eprintln!(
        "[yue2] партитура {:.1} с ({} ток., {:.1} ток/с) | музыка {:.1} с ({} ток., {:.1} ток/с{}) | flow {:.1} с | декод {:.1} с | всего {:.1} с",
        t.abc.seconds,
        t.abc.output_tokens,
        t.abc.tokens_per_second,
        t.semantic.seconds,
        t.semantic.output_tokens,
        t.semantic.tokens_per_second,
        if t.semantic.cuda_graph { ", cuda-граф" } else { "" },
        t.nar_seconds,
        t.vae_seconds,
        t.total_seconds
    );
    let truncated = song.semantic.plan.truncated || song.semantic.truncated;
    eprintln!(
        "[yue2] {} → {:.1} с звука{}",
        args.output.display(),
        song.seconds(),
        if truncated { " (упёрлось в лимит токенов)" } else { "" }
    );
    Ok(())
}

pub fn run_decode(args: SongDecodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = device::resolve(&args.device);
    let vae_path = resolve_vae(&args.models, args.vae)?;
    let vae_dtype = parse_dtype(args.vae_dtype.as_deref(), DType::F32);
    let latents = load_latent(&args.latent)?;
    let started = std::time::Instant::now();
    let vae = Yue2Vae::open(&vae_path, dev, vae_dtype, true)?;
    eprintln!(
        "[yue2] декодер {} ({:?} {:?}) | латенты {:?} | загрузка {:.1} с",
        vae_path.display(),
        dev,
        vae_dtype,
        latents.dims(),
        started.elapsed().as_secs_f64()
    );
    let z = latents.transpose(0, 1)?.contiguous()?.unsqueeze(0)?.to_device(dev)?;
    let on_vae = |done: usize, total: usize| {
        eprint!("\r[yue2] декод: тайл {done}/{total}    ");
        let _ = std::io::stderr().flush();
    };
    let audio = vae.decode_tiled(&z, args.vae_core_frames.max(16), args.vae_halo_frames, &|| false, &on_vae)?;
    eprintln!();
    let samples = synaptix_music_yue2::pipeline::interleave(&audio)?;
    write_wav_interleaved_f32(&args.output, &samples, synaptix_music_yue2::protocol::SAMPLE_RATE, 2)?;
    eprintln!(
        "[yue2] {} → {:.1} с звука за {:.1} с",
        args.output.display(),
        samples.len() as f32 / (2.0 * synaptix_music_yue2::protocol::SAMPLE_RATE as f32),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
