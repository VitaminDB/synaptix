//! Музыка одним вызовом: ACE-Step v1.5 (музыка по описанию и тексту) и YuE2
//! (песня с вокалом). Умолчания — как у `synaptix music` / `synaptix song`.
//! Отмена — флагом `cancel`; прогресс — `progress(готово, всего)`, где он есть.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

type R<T> = Result<T, String>;

#[derive(Debug, Clone, Default)]
pub struct MusicRequest {
    /// Описание: жанр, инструменты, настроение (ACE-Step — caption, YuE2 — style).
    pub prompt: String,
    /// Текст песни (с метками [verse]/[chorus]); пусто — инструментал.
    pub lyrics: String,
    /// Длительность, секунды; 0 — модель решает сама.
    pub seconds: u32,
    pub seed: u64,
    pub steps: Option<usize>,
    /// CFG (ACE-Step); `None` — умолчание варианта.
    pub guidance: Option<f32>,
    /// Квант: `nvfp4`, `mxfp8`, `bf16`; пусто/`auto` — плотный.
    pub quant: Option<String>,
    /// Родные настройки YuE2 (остальные модели их не смотрят).
    pub yue2: Yue2Settings,
}

/// Сэмплинг одной фазы YuE2 поверх умолчаний чекпойнта
/// (`yue2_generation_config.json`): `None` — как в оригинале.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingOverride {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub penalty_window: Option<usize>,
    pub min_tokens: Option<usize>,
    pub max_tokens: Option<usize>,
}

impl SamplingOverride {
    fn apply(&self, s: &mut synaptix_music_yue2::Sampling) {
        if let Some(v) = self.temperature {
            s.temperature = v;
        }
        if let Some(v) = self.top_p {
            s.top_p = v;
        }
        if let Some(v) = self.top_k {
            s.top_k = v;
        }
        if let Some(v) = self.repetition_penalty {
            s.repetition_penalty = v;
        }
        if let Some(v) = self.penalty_window {
            s.penalty_window = v;
        }
        if let Some(v) = self.min_tokens {
            s.min_tokens = v;
        }
        if let Some(v) = self.max_tokens {
            s.max_tokens = v;
        }
    }
}

/// Родные режимы и настройки YuE2 (как у `YuE2Pipeline`, `yue2 generate` и
/// официального навыка инструментала). Пусто — умолчания оригинала.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Yue2Settings {
    /// Вокал: `Some(false)` — инструментал родным процессом (YuE2 пишет
    /// партитуру, мелодия `Vocal` переносится в `Ins`, рендер по ней без слов);
    /// `None` — по тексту: нет слов (пусто или только метки разделов) — без вокала.
    pub vocals: Option<bool>,
    /// Партитура перед музыкой: `full` (мелодия и аккорды), `melody`, `off`.
    pub cot: Option<String>,
    /// Своя партитура в родной ABC (вместо сочинённой моделью).
    pub abc: Option<String>,
    /// Мелодия из MIDI-файла (оригинал MIDI не читает — перевод в родную ABC).
    pub midi: Option<PathBuf>,
    /// Дорожка MIDI (с 0); `None` — с наибольшим числом нот.
    pub midi_track: Option<usize>,
    /// Кавер: запись, с которой SheetSage2 снимает партитуру.
    pub cover: Option<PathBuf>,
    /// Кавер с исходной гармонией (`full`), иначе — только мелодия.
    pub cover_harmony: bool,
    /// Text guidance (CFG); `None` — умолчание режима (1.0, у `off` — 1.01).
    pub cfg_scale: Option<f32>,
    pub abc_sampling: SamplingOverride,
    pub semantic_sampling: SamplingOverride,
    /// Не короче, секунды (min_tokens семантической фазы).
    pub min_seconds: Option<f32>,
    /// Окно модели для нарезки на чанки (≤ 24576).
    pub context: Option<usize>,
    /// Только партитура, без звука (план для правки).
    pub plan_only: bool,
}

pub struct MusicOutput {
    /// Отсчёты, перемежённые по каналам.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub label: String,
    /// Партитура, по которой звучит результат (YuE2).
    pub score: Option<String>,
    /// Что сделано по ходу: откуда партитура, перенос голосов, итоговые стиль и лирика.
    pub notes: Vec<String>,
}

/// Компоненты ACE-Step: DiT (сама модель), LM кодов, текстовый энкодер, VAE.
#[derive(Debug, Clone)]
pub struct AcePaths {
    pub dit: PathBuf,
    pub lm: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

pub fn acestep(paths: &AcePaths, device: Device, req: &MusicRequest, cancel: Arc<AtomicBool>) -> R<MusicOutput> {
    use synaptix_music_acestep::ar::CodesGenOptions;
    use synaptix_music_acestep::pipeline::{generate_music, EditOptions, GenExtras, MusicPaths, SamplerOptions};
    use synaptix_music_acestep::DitVariant;
    for (label, p) in [("DiT", &paths.dit), ("LM", &paths.lm), ("текстовый энкодер", &paths.text_encoder), ("VAE", &paths.vae)] {
        if !p.exists() {
            return Err(format!("ACE-Step: нет компонента {label}: {}", p.display()));
        }
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let variant = DitVariant::detect(&paths.dit);
    let steps = req.steps.filter(|&s| s > 0).unwrap_or(variant.default_steps());
    let opts = SamplerOptions { steps, shift: variant.default_shift(), guidance_scale: req.guidance.unwrap_or(variant.default_cfg()), ..Default::default() };
    let copts = CodesGenOptions { seed: req.seed.max(1), ..CodesGenOptions::default() };
    let extras = GenExtras::ar_on();
    let mp = MusicPaths { lm: &paths.lm, text_encoder: &paths.text_encoder, dit: &paths.dit, vae: &paths.vae };
    let compute = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let quant = match req.quant.as_deref().unwrap_or("").to_ascii_lowercase().as_str() {
        "nvfp4" if device.is_cuda() => DType::NVFP4,
        "mxfp8" | "fp8" if device.is_cuda() => DType::MXFP8,
        _ => compute,
    };
    let (samples, sample_rate, _latent) = synaptix_music_acestep::with_cancel(Some(cancel), || {
        generate_music(&mp, &req.prompt, &req.lyrics, req.seconds, device, compute, quant, quant, &opts, &copts, true, &EditOptions::default(), &extras, None)
    })
    .map_err(|e| e.to_string())?;
    let secs = samples.len() as f32 / 2.0 / sample_rate.max(1) as f32;
    Ok(MusicOutput { samples, sample_rate, channels: 2, label: format!("ACE-Step {variant:?}, {secs:.0} с, {steps} шагов"), score: None, notes: Vec::new() })
}

/// YuE2: песня (стиль + текст). `model` и `vae` — бандлы.
/// Есть ли в лирике слова (не только метки разделов `[Verse]`).
pub fn lyrics_have_words(lyrics: &str) -> bool {
    lyrics.lines().map(str::trim).any(|l| !l.is_empty() && !(l.starts_with('[') && l.ends_with(']')))
}

/// Только метки разделов из лирики (`[Verse]\n\n[Chorus]\n`).
fn lyric_section_tags(lyrics: &str) -> String {
    let tags: Vec<&str> = lyrics.lines().map(str::trim).filter(|l| l.starts_with('[') && l.ends_with(']')).collect();
    if tags.is_empty() {
        String::new()
    } else {
        tags.join("\n\n") + "\n"
    }
}

/// YuE2 со всеми родными режимами: песня, своя партитура, MIDI-мелодия, кавер
/// по записи (SheetSage2), инструментал родным процессом, только партитура.
pub fn yue2(model: &Path, vae: &Path, device: Device, req: &MusicRequest, cancel: Arc<AtomicBool>, progress: &mut dyn FnMut(usize, usize)) -> R<MusicOutput> {
    use synaptix_music_yue2::pipeline::{Callbacks, Yue2Options, Yue2Paths, Yue2Pipeline};
    use synaptix_music_yue2::protocol::{seconds_to_frames, Cot, GenerationConfig, SongRequest, CONTEXT};
    use synaptix_music_yue2::score;
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let y = &req.yue2;
    let mut notes = Vec::new();
    let mut generation = GenerationConfig::default();
    if let Some(s) = req.steps.filter(|&s| s > 0) {
        generation.ode_steps = s;
    }
    if let Some(c) = y.context {
        generation.context = c.clamp(1024, CONTEXT);
    }
    let seconds = if req.seconds > 0 { req.seconds } else { 60 };
    generation.semantic.max_tokens = seconds_to_frames(seconds as f32).clamp(1, CONTEXT - 2);
    if let Some(m) = y.min_seconds {
        generation.semantic.min_tokens = seconds_to_frames(m).min(generation.semantic.max_tokens);
    }
    y.abc_sampling.apply(&mut generation.abc);
    y.semantic_sampling.apply(&mut generation.semantic);
    generation.semantic.min_tokens = generation.semantic.min_tokens.min(generation.semantic.max_tokens);
    let mut cot = match y.cot.as_deref() {
        None | Some("") => None,
        Some(c) => Some(Cot::parse(c).ok_or_else(|| format!("YuE2: cot «{c}» — нужно full, melody или off"))?),
    };
    let vocals = y.vocals.unwrap_or_else(|| lyrics_have_words(&req.lyrics));
    let is_cancelled = || cancel.load(std::sync::atomic::Ordering::Relaxed);

    // Своя партитура: ABC, MIDI или снятая с записи (до загрузки костяка — карта свободнее).
    let mut abc = y.abc.clone().filter(|a| !a.trim().is_empty());
    let mut from_melody_only = false;
    if abc.is_none() {
        if let Some(path) = &y.midi {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let pick = y.midi_track.map(synaptix_music_yue2::midi::MidiPick::Track).unwrap_or_default();
            let (text, rep) = synaptix_music_yue2::midi::midi_to_abc(&bytes, pick).map_err(|e| e.to_string())?;
            notes.push(format!(
                "партитура из MIDI: дорожка {:?}, нот {} → {}, нижних голосов отброшено {}, укорочено {}, сдвинуто сеткой 1/32 {}, {} BPM, {}",
                rep.track, rep.notes_in, rep.notes_out, rep.dropped_polyphony, rep.truncated, rep.quantized, rep.bpm, rep.key
            ));
            abc = Some(if vocals { score::melody_to_vocal(&text).map_err(|e| e.to_string())? } else { text });
            from_melody_only = true;
        } else if let Some(path) = &y.cover {
            let mode = if y.cover_harmony { "full" } else { "melody" };
            let text = crate::music_stages::sheetsage_abc(model, device, path, mode, "both", &is_cancelled, &|_| {})?;
            notes.push(format!("партитура снята SheetSage2 с {} ({})", path.display(), if y.cover_harmony { "мелодия и аккорды" } else { "только мелодия" }));
            abc = Some(text);
            from_melody_only = !y.cover_harmony;
        }
    }
    let options = Yue2Options {
        device,
        compute: DType::BF16,
        quant: match req.quant.as_deref().unwrap_or("").to_ascii_lowercase().as_str() {
            "nvfp4" => Some(DType::NVFP4),
            "mxfp8" | "fp8" => Some(DType::MXFP8),
            _ => None,
        },
        vae_dtype: DType::F32,
        vae_core_frames: 1024,
        vae_halo_frames: 16,
        generation,
    };
    let mut pipe = Yue2Pipeline::open(&Yue2Paths { model: model.to_path_buf(), vae: vae.to_path_buf() }, options).map_err(|e| e.to_string())?;
    let seed = if req.seed == 0 { 831001 } else { req.seed };
    let progress = std::cell::RefCell::new(progress);
    let on_ode = |done: usize, total: usize| (progress.borrow_mut())(done, total);
    let cb = Callbacks { cancel: Some(&is_cancelled), on_token: None, on_ode: Some(&on_ode), on_vae: None };

    let (style, lyrics) = if vocals {
        let c = cot.unwrap_or(if from_melody_only { Cot::Melody } else { Cot::Full });
        cot = Some(c);
        (req.prompt.clone(), req.lyrics.clone())
    } else {
        // Родной инструментал: партитура (своя или сочинённая YuE2 по меткам
        // разделов), мелодия Vocal → Ins, стиль с явными запретами голоса,
        // лирика — только метки разделов.
        let plan_mode = cot.filter(|c| *c != Cot::Off).unwrap_or(Cot::Full);
        let source = match abc.take() {
            Some(a) => a,
            None => {
                let tags = lyric_section_tags(&req.lyrics);
                let plan_lyrics = if tags.is_empty() { score::INSTRUMENTAL_PLAN_LYRICS.to_string() } else { tags };
                let plan_style = if req.prompt.trim().is_empty() { "Instrumental, warm lyrical acoustic piano, no vocals, no singing, no choir".to_string() } else { req.prompt.clone() };
                let plan_req = SongRequest { style: plan_style, lyrics: plan_lyrics, cot: plan_mode, seed, abc: None, cfg_scale: None };
                let plan = pipe.plan(&plan_req, cb).map_err(|e| e.to_string())?;
                if plan.truncated {
                    return Err("YuE2: партитура оборвана на пределе abc_max_tokens — увеличьте его или повторите с другим зерном".into());
                }
                notes.push("партитуру сочинила YuE2 (родной путь инструментала)".into());
                plan.abc.ok_or("YuE2: модель не написала партитуру")?
            }
        };
        let keep_chords = !from_melody_only;
        let (converted, check) = score::instrumentalize(&source, score::Overlap::Vocal, keep_chords).map_err(|e| e.to_string())?;
        let checked = score::validate_instrumental(&converted).map_err(|e| e.to_string())?;
        notes.push(format!(
            "мелодия Vocal → Ins: нот голоса {}, нот Ins {} (из них не задето {}, урезано наложением {}), итог {} нот, аккорды {}",
            check.vocal_notes_before,
            check.original_ins_notes,
            check.unaltered_ins_notes,
            check.affected_ins_notes,
            check.output_ins_notes,
            if check.chords_kept { "сохранены" } else { "убраны" }
        ));
        cot = Some(if checked.vocal.chords.is_empty() { Cot::Melody } else { Cot::Full });
        let lyrics = score::lyric_tags(&converted);
        abc = Some(converted);
        (score::instrumental_style(&req.prompt), lyrics)
    };
    let cot = cot.expect("режим выбран");
    if cot == Cot::Off && abc.is_some() {
        return Err("YuE2: своя партитура требует cot = full или melody".into());
    }
    notes.push(format!("стиль: {style}"));
    if !vocals {
        notes.push(format!("лирика (без слов): {}", lyrics.trim().replace('\n', " ")));
    }
    let request = SongRequest { style, lyrics, cot, seed, abc: abc.clone(), cfg_scale: y.cfg_scale };
    request.validate().map_err(|e| e.to_string())?;
    if y.plan_only {
        let score = match abc {
            Some(a) => a,
            None if cot == Cot::Off => return Err("YuE2: при cot = off партитуры нет".into()),
            None => {
                let plan = pipe.plan(&request, cb).map_err(|e| e.to_string())?;
                plan.abc.ok_or("YuE2: модель не написала партитуру")?
            }
        };
        return Ok(MusicOutput { samples: Vec::new(), sample_rate: 48_000, channels: 2, label: format!("YuE2, партитура ({} строк)", score.lines().count()), score: Some(score), notes });
    }
    let song = pipe.run(&request, cb).map_err(|e| e.to_string())?;
    let secs = song.audio.len() as f32 / (song.sample_rate.max(1) as f32 * song.channels.max(1) as f32);
    let score = song.semantic.plan.abc.clone();
    let mode = if vocals { "песня" } else { "инструментал" };
    Ok(MusicOutput { samples: song.audio, sample_rate: song.sample_rate, channels: song.channels as u16, label: format!("YuE2, {mode}, {secs:.0} с"), score, notes })
}

/// WAV 16 бит из перемежённых отсчётов.
/// MiniMax Music 3: песня по описанию (`prompt`) и тексту (`lyrics`), 44,1 кГц
/// стерео. `seconds` — верхняя граница (0 — минута; модель может закончить
/// раньше), предел — 6 минут. `progress(шаг, всего)` — по кадрам сочинения и
/// шагам звука одной шкалой.
pub fn minimax3(model: &Path, device: Device, req: &MusicRequest, cancel: Arc<AtomicBool>, progress: &mut dyn FnMut(usize, usize)) -> R<MusicOutput> {
    use synaptix_music_minimax3::{Music3, Music3Request};
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if req.lyrics.trim().is_empty() {
        return Err("MiniMax Music 3: нужен текст песни (инструментал без текста модель не делает)".into());
    }
    let mut m = Music3::open(model, device).map_err(|e| e.to_string())?;
    m.quant = match req.quant.as_deref() {
        Some("mxfp8") => Some(DType::MXFP8),
        Some("nvfp4") => Some(DType::NVFP4),
        _ => None,
    };
    let seconds = if req.seconds == 0 { 60.0 } else { (req.seconds as f64).min(360.0) };
    let steps = req.steps.filter(|&s| s > 0).unwrap_or(30);
    let r = Music3Request { description: req.prompt.clone(), lyrics: req.lyrics.clone(), seconds, seed: req.seed, steps };
    // Шкала: кадры сочинения (25 в секунду) + шаги звука, приведённые к кадрам.
    let frames = (seconds * 25.0) as usize;
    let out = m
        .generate(&r, &mut |stage, i, n| {
            match stage {
                "сочинение" => progress(i, frames * 2),
                "звук" => progress(frames + frames * i / n.max(1), frames * 2),
                _ => {}
            }
            !cancel.load(std::sync::atomic::Ordering::Relaxed)
        })
        .map_err(|e| e.to_string())?;
    let secs = out.pcm.len() as f64 / 2.0 / out.sample_rate as f64;
    Ok(MusicOutput {
        label: format!("MiniMax Music 3, {secs:.0} с, {steps} шагов (сочинение {:.0} с, звук {:.0} с)", out.ar_seconds, out.flow_seconds),
        samples: out.pcm,
        sample_rate: out.sample_rate,
        channels: out.channels,
        score: None,
        notes: Vec::new(),
    })
}

pub fn wav_bytes(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut w = Vec::with_capacity(44 + samples.len() * 2);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&channels.to_le_bytes());
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&(sample_rate * channels as u32 * 2).to_le_bytes());
    w.extend_from_slice(&(channels * 2).to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for &x in samples {
        w.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    w
}

#[cfg(test)]
mod yue2_tests {
    use super::*;

    #[test]
    fn lyrics_words_and_tags() {
        assert!(!lyrics_have_words(""));
        assert!(!lyrics_have_words("[Intro]\n\n[Verse]\n[Chorus]\n"));
        assert!(!lyrics_have_words("  [Instrumental]  "));
        assert!(lyrics_have_words("[Verse]\nВечер тихий за окном\n"));
        assert_eq!(lyric_section_tags("[Verse]\nслова\n[Chorus]\n"), "[Verse]\n\n[Chorus]\n");
        assert_eq!(lyric_section_tags("просто слова"), "");
    }
}
