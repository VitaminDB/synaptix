//! `synaptix song` — песня целиком: стиль и лирика → партитура → музыка →
//! 48 кГц стерео WAV.

use std::io::Write;
use std::path::PathBuf;

use synaptix_audio::io::write_wav_interleaved_f32;
use synaptix_core::dtype::DType;
use synaptix_music_yue2::ar::Phase;
use synaptix_music_yue2::pipeline::{
    find_bundle, Callbacks, Yue2Options, Yue2Paths, Yue2Pipeline, MODEL_NAMES, VAE_NAMES,
};
use synaptix_music_yue2::protocol::{frames_to_seconds, Cot, GenerationConfig, SongRequest};

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
    pub seed: u64,
    pub cfg: Option<f32>,
    pub steps: usize,
    /// Потолок семантических токенов (25 на секунду музыки).
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub device: String,
    pub compute_dtype: Option<String>,
    pub quant: Option<String>,
    pub vae_dtype: Option<String>,
    pub vae_core_frames: usize,
    /// Куда сохранить партитуру (`score.abc`).
    pub save_abc: Option<PathBuf>,
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
    let vae = match args.vae {
        Some(p) => p,
        None => find_bundle(&args.models, VAE_NAMES).ok_or_else(|| {
            format!(
                "в {} нет ни одного из {VAE_NAMES:?} — укажите --vae",
                args.models.display()
            )
        })?,
    };

    let cot = Cot::parse(&args.cot).ok_or("--cot ожидает full | melody | off")?;
    let lyrics = match &args.lyrics_file {
        Some(p) => std::fs::read_to_string(p)?,
        None => args.lyrics.clone(),
    };
    let abc = match &args.abc_file {
        Some(p) => Some(std::fs::read_to_string(p)?),
        None => None,
    };

    let mut generation = GenerationConfig::default();
    generation.ode_steps = args.steps;
    if let Some(v) = args.max_tokens {
        generation.semantic.max_tokens = v;
        generation.semantic.min_tokens = generation.semantic.min_tokens.min(v);
    }
    if let Some(v) = args.temperature {
        generation.semantic.temperature = v;
    }
    if let Some(v) = args.top_p {
        generation.semantic.top_p = v;
    }
    if let Some(v) = args.top_k {
        generation.semantic.top_k = v;
    }
    if let Some(v) = args.repetition_penalty {
        generation.semantic.repetition_penalty = v;
    }

    let options = Yue2Options {
        device: dev,
        compute: parse_dtype(args.compute_dtype.as_deref(), DType::BF16),
        quant: parse_quant(args.quant.as_deref()),
        vae_dtype: parse_dtype(args.vae_dtype.as_deref(), DType::F32),
        vae_core_frames: args.vae_core_frames,
        vae_halo_frames: 16,
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
