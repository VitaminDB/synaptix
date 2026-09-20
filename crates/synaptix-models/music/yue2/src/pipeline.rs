//! Пайплайн YuE2: партитура → семантические токены → акустические латенты →
//! 48 кГц стерео.
//!
//! Стадии разделены, как в релизе: план можно сохранить, поправить партитуру
//! руками и отрендерить заново. Обе AR-фазы при обычном прогоне идут по одной
//! сессии — второй префилл текста и партитуры не делается.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::ar::{run_phase, ArSession, Phase, PhaseTiming, Yue2Ar};
use crate::nar::{synthesize, Yue2Nar};
use crate::protocol::{
    frames_to_seconds, negative_prefix, token_prefixes, Cot, GenerationConfig, SongRequest,
    ABC_END, CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_START, SAMPLE_RATE,
};
use crate::tokenizer::Yue2Tokenizer;
use crate::vae::Yue2Vae;
use crate::YueError;

type R<T> = Result<T, YueError>;

/// Где лежат бандлы.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yue2Paths {
    /// `yue2-3b.syn` — костяк AR–NAR и текстовый BPE.
    pub model: PathBuf,
    /// `yue2-vae.syn` — декодер звука.
    pub vae: PathBuf,
}

/// Как считать.
#[derive(Debug, Clone)]
pub struct Yue2Options {
    pub device: Device,
    /// Тип вычислений костяка (эталон — BF16).
    pub compute: DType,
    /// Квант весов костяка: `None` — плотно.
    pub quant: Option<DType>,
    /// Тип вычислений VAE (эталон — F32).
    pub vae_dtype: DType,
    /// Кадров в ядре тайла декодера и сколько контекста добавлять по краям.
    pub vae_core_frames: usize,
    pub vae_halo_frames: usize,
    pub generation: GenerationConfig,
}

impl Default for Yue2Options {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            compute: DType::BF16,
            quant: None,
            vae_dtype: DType::F32,
            vae_core_frames: 1024,
            vae_halo_frames: 16,
            generation: GenerationConfig::default(),
        }
    }
}

/// Колбэки прогресса и отмены. Любой из них можно не задавать.
#[derive(Default, Clone, Copy)]
pub struct Callbacks<'a> {
    pub cancel: Option<&'a dyn Fn() -> bool>,
    /// Токен фазы: фаза, id, номер шага.
    pub on_token: Option<&'a dyn Fn(Phase, u32, usize)>,
    /// Шаги ODE: сделано из всего.
    pub on_ode: Option<&'a dyn Fn(usize, usize)>,
    /// Тайлы VAE: сделано из всего.
    pub on_vae: Option<&'a dyn Fn(usize, usize)>,
}

impl Callbacks<'_> {
    fn cancelled(&self) -> bool {
        self.cancel.map(|f| f()).unwrap_or(false)
    }
}

/// План: партитура и префикс, из которого пойдёт музыка.
#[derive(Debug, Clone)]
pub struct SymbolicPlan {
    pub request: SongRequest,
    /// Текст партитуры (`None` при `cot=off`).
    pub abc: Option<String>,
    pub abc_ids: Vec<u32>,
    /// Префикс музыкальной фазы.
    pub prefix: Vec<u32>,
    pub timing: PhaseTiming,
    pub truncated: bool,
}

/// Семантические токены песни (значения `0..32768`).
#[derive(Debug, Clone)]
pub struct SemanticResult {
    pub plan: SymbolicPlan,
    pub tokens: Vec<u32>,
    pub timing: PhaseTiming,
    pub truncated: bool,
}

impl SemanticResult {
    pub fn seconds(&self) -> f32 {
        frames_to_seconds(self.tokens.len())
    }
}

/// Готовая песня.
pub struct SongResult {
    /// Стерео, чередующиеся отсчёты (L, R, L, R…).
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub channels: usize,
    pub semantic: SemanticResult,
    /// Акустические латенты `[кадры, 64]` — их можно передекодировать другим
    /// декодером, не повторяя генерацию.
    pub latents: Tensor,
    pub timing: SongTiming,
}

impl SongResult {
    pub fn seconds(&self) -> f32 {
        self.audio.len() as f32 / (self.channels as f32 * self.sample_rate as f32)
    }
}

#[derive(Debug, Clone, Default)]
pub struct SongTiming {
    pub abc: PhaseTiming,
    pub semantic: PhaseTiming,
    pub nar_seconds: f64,
    pub vae_seconds: f64,
    pub load_seconds: f64,
    pub total_seconds: f64,
}

/// Фаза музыки выдаёт токены в адресах словаря; наружу и в NAR-ветку идут
/// номера семантических токенов `0..32768`.
fn to_codec(tokens: Vec<u32>) -> R<Vec<u32>> {
    tokens
        .into_iter()
        .map(|t| {
            t.checked_sub(CODEC_OFFSET)
                .filter(|c| *c < CODEC_SIZE)
                .ok_or_else(|| YueError::Other(format!("фаза музыки выдала токен {t} вне кодека")))
        })
        .collect()
}

pub struct Yue2Pipeline {
    pub ar: Arc<Yue2Ar>,
    pub nar: Arc<Yue2Nar>,
    pub tokenizer: Arc<Yue2Tokenizer>,
    pub options: Yue2Options,
    /// Откуда брать декодер, если его не передали готовым.
    vae_path: Option<PathBuf>,
    /// Декодер держим отдельно: он нужен в самом конце, а весит полгигабайта.
    vae: Option<Arc<Yue2Vae>>,
    pub load_seconds: f64,
}

impl Yue2Pipeline {
    /// Открыть костяк. Декодер подтягивается при первом декоде
    /// ([`Self::decode`]) — так его вес не занимает место во время генерации.
    pub fn open(paths: &Yue2Paths, options: Yue2Options) -> R<Self> {
        let started = Instant::now();
        let tokenizer = Yue2Tokenizer::from_tiktoken_bytes(&crate::loader::read_bundle_file(
            &paths.model,
            "qwen.tiktoken",
        )?)?;
        let ar = Yue2Ar::open(
            &paths.model,
            options.device,
            options.compute,
            options.quant,
            options.generation.context,
        )?;
        let nar = Yue2Nar::open(&paths.model, options.device, options.compute, options.quant)?;
        Ok(Self {
            ar: Arc::new(ar),
            nar: Arc::new(nar),
            tokenizer: Arc::new(tokenizer),
            vae_path: Some(paths.vae.clone()),
            vae: None,
            options,
            load_seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// Собрать пайплайн из уже загруженных частей — так его строит приложение,
    /// у которого веса живут в общем кэше и переживают прогоны.
    pub fn from_parts(
        ar: Arc<Yue2Ar>,
        nar: Arc<Yue2Nar>,
        tokenizer: Arc<Yue2Tokenizer>,
        vae: Option<Arc<Yue2Vae>>,
        options: Yue2Options,
    ) -> Self {
        Self { ar, nar, tokenizer, vae, vae_path: None, options, load_seconds: 0.0 }
    }

    /// Подставить готовый декодер (загруженный снаружи).
    pub fn set_vae(&mut self, vae: Arc<Yue2Vae>) {
        self.vae = Some(vae);
    }

    /// Загрузить декодер заранее (нода «держать в памяти»).
    pub fn load_vae(&mut self) -> R<()> {
        if self.vae.is_some() {
            return Ok(());
        }
        let Some(path) = &self.vae_path else {
            return Err(YueError::Config(
                "декодер не передан и путь к нему неизвестен".into(),
            ));
        };
        self.vae = Some(Arc::new(Yue2Vae::open(
            path,
            self.options.device,
            self.options.vae_dtype,
            true,
        )?));
        Ok(())
    }

    /// Выгрузить декодер.
    pub fn unload_vae(&mut self) {
        self.vae = None;
    }

    /// Сколько токенов займут фазы и хватает ли окна.
    fn budget(&self, prefix_len: usize, with_abc: bool) -> R<usize> {
        let g = &self.options.generation;
        let abc = if with_abc { g.abc.max_tokens + 2 } else { 0 };
        let need = prefix_len + abc + g.semantic.max_tokens;
        if need > CONTEXT {
            return Err(YueError::Config(format!(
                "префикс и запрошенный бюджет генерации не влезают в окно: {need} > {CONTEXT}"
            )));
        }
        Ok(need)
    }

    /// Партитура и семантические токены за один проход: обе фазы идут по одной
    /// сессии, поэтому текст и партитура префиллятся ровно один раз.
    pub fn generate(&self, request: &SongRequest, cb: Callbacks<'_>) -> R<SemanticResult> {
        request.validate()?;
        let g = &self.options.generation;
        let noop_token = |_: Phase, _: u32, _: usize| {};
        let on_token: &dyn Fn(Phase, u32, usize) = cb.on_token.unwrap_or(&noop_token);
        let noop_cancel = || false;
        let cancel: &dyn Fn() -> bool = cb.cancel.unwrap_or(&noop_cancel);

        // Партитура пришла извне или не нужна вовсе — одна фаза.
        if request.cot == Cot::Off || request.abc.is_some() {
            let plan = self.plan(request, cb)?;
            return self.generate_semantic(&plan, cb);
        }

        let head = token_prefixes(request, &self.tokenizer, None)?;
        let max_seq = self.budget(head.len(), true)?;
        let mut session = ArSession::new(&self.ar, max_seq)?;
        session.feed(&head)?;
        let abc_out = run_phase(
            &mut session,
            None,
            Phase::Abc,
            &g.abc,
            1.0,
            request.seed,
            cancel,
            on_token,
        )?;
        let abc_ids = abc_out.tokens.clone();
        let abc_text = self.tokenizer.decode(&abc_ids);
        let prefix = token_prefixes(request, &self.tokenizer, Some(&abc_ids))?;
        let plan = SymbolicPlan {
            request: request.clone(),
            abc: Some(abc_text),
            abc_ids,
            prefix,
            timing: abc_out.timing,
            truncated: abc_out.truncated,
        };

        // Кэш уже содержит текст, `<abc>` и саму партитуру — осталось дописать
        // её конец и начало музыки.
        session.feed(&[ABC_END, MUSIC_START])?;
        let guidance = request.guidance();
        let mut negative_session = if guidance != 1.0 {
            let negative = negative_prefix(request, &self.tokenizer, Some(&plan.abc_ids))?;
            self.budget(negative.len(), false)?;
            let mut s = ArSession::new(&self.ar, negative.len() + g.semantic.max_tokens)?;
            s.feed(&negative)?;
            Some(s)
        } else {
            None
        };
        let semantic = run_phase(
            &mut session,
            negative_session.as_mut(),
            Phase::Semantic,
            &g.semantic,
            guidance,
            request.seed,
            cancel,
            on_token,
        )?;
        Ok(SemanticResult {
            plan,
            tokens: to_codec(semantic.tokens)?,
            timing: semantic.timing,
            truncated: semantic.truncated,
        })
    }

    /// Только партитура (для правки перед рендером).
    pub fn plan(&self, request: &SongRequest, cb: Callbacks<'_>) -> R<SymbolicPlan> {
        request.validate()?;
        let noop_token = |_: Phase, _: u32, _: usize| {};
        let on_token: &dyn Fn(Phase, u32, usize) = cb.on_token.unwrap_or(&noop_token);
        let noop_cancel = || false;
        let cancel: &dyn Fn() -> bool = cb.cancel.unwrap_or(&noop_cancel);

        if request.cot == Cot::Off {
            return Ok(SymbolicPlan {
                prefix: token_prefixes(request, &self.tokenizer, None)?,
                request: request.clone(),
                abc: None,
                abc_ids: Vec::new(),
                timing: PhaseTiming::default(),
                truncated: false,
            });
        }
        if let Some(abc) = &request.abc {
            let ids = self.tokenizer.encode(abc);
            return Ok(SymbolicPlan {
                prefix: token_prefixes(request, &self.tokenizer, Some(&ids))?,
                request: request.clone(),
                abc: Some(abc.clone()),
                abc_ids: ids,
                timing: PhaseTiming::default(),
                truncated: false,
            });
        }
        let head = token_prefixes(request, &self.tokenizer, None)?;
        let max_seq = self.budget(head.len(), true)?;
        let mut session = ArSession::new(&self.ar, max_seq)?;
        session.feed(&head)?;
        let out = run_phase(
            &mut session,
            None,
            Phase::Abc,
            &self.options.generation.abc,
            1.0,
            request.seed,
            cancel,
            on_token,
        )?;
        let abc_ids = out.tokens;
        Ok(SymbolicPlan {
            prefix: token_prefixes(request, &self.tokenizer, Some(&abc_ids))?,
            abc: Some(self.tokenizer.decode(&abc_ids)),
            abc_ids,
            request: request.clone(),
            timing: out.timing,
            truncated: out.truncated,
        })
    }

    /// Семантические токены по готовому плану (полный префилл префикса).
    pub fn generate_semantic(&self, plan: &SymbolicPlan, cb: Callbacks<'_>) -> R<SemanticResult> {
        let g = &self.options.generation;
        let noop_token = |_: Phase, _: u32, _: usize| {};
        let on_token: &dyn Fn(Phase, u32, usize) = cb.on_token.unwrap_or(&noop_token);
        let noop_cancel = || false;
        let cancel: &dyn Fn() -> bool = cb.cancel.unwrap_or(&noop_cancel);

        let expected = token_prefixes(&plan.request, &self.tokenizer, Some(&plan.abc_ids))?;
        if plan.request.cot != Cot::Off && expected != plan.prefix {
            return Err(YueError::Config(
                "префикс плана расходится с запросом и партитурой".into(),
            ));
        }
        let max_seq = self.budget(plan.prefix.len(), false)?;
        let mut session = ArSession::new(&self.ar, max_seq)?;
        session.feed(&plan.prefix)?;
        let guidance = plan.request.guidance();
        let mut negative_session = if guidance != 1.0 {
            let abc = if plan.request.cot == Cot::Off { None } else { Some(&plan.abc_ids[..]) };
            let negative = negative_prefix(&plan.request, &self.tokenizer, abc)?;
            let mut s = ArSession::new(&self.ar, negative.len() + g.semantic.max_tokens)?;
            s.feed(&negative)?;
            Some(s)
        } else {
            None
        };
        let out = run_phase(
            &mut session,
            negative_session.as_mut(),
            Phase::Semantic,
            &g.semantic,
            guidance,
            plan.request.seed,
            cancel,
            on_token,
        )?;
        Ok(SemanticResult {
            plan: plan.clone(),
            tokens: to_codec(out.tokens)?,
            timing: out.timing,
            truncated: out.truncated,
        })
    }

    /// Акустические латенты `[кадры, 64]`.
    pub fn synthesize(&self, semantic: &SemanticResult, cb: Callbacks<'_>) -> R<Tensor> {
        if semantic.tokens.is_empty() {
            return Err(YueError::Other("семантических токенов нет — нечего синтезировать".into()));
        }
        let noop_cancel = || false;
        let cancel: &dyn Fn() -> bool = cb.cancel.unwrap_or(&noop_cancel);
        let noop_progress = |_: usize, _: usize| {};
        let on_ode: &dyn Fn(usize, usize) = cb.on_ode.unwrap_or(&noop_progress);
        synthesize(
            &self.ar,
            &self.nar,
            &semantic.plan.prefix,
            &semantic.tokens,
            semantic.plan.request.seed,
            self.options.generation.ode_steps,
            self.options.generation.context,
            cancel,
            on_ode,
        )
    }

    /// Латенты → звук. Возвращает чередующиеся стерео-отсчёты.
    pub fn decode(&mut self, latents: &Tensor, cb: Callbacks<'_>) -> R<Vec<f32>> {
        self.load_vae()?;
        let vae = self.vae.clone().expect("декодер загружен выше");
        self.decode_with(&vae, latents, cb)
    }

    /// То же, но конкретным декодером — например `legacy`, когда латенты уже
    /// посчитаны и нужен другой звук из них же.
    pub fn decode_with(&self, vae: &Yue2Vae, latents: &Tensor, cb: Callbacks<'_>) -> R<Vec<f32>> {
        let noop_cancel = || false;
        let cancel: &dyn Fn() -> bool = cb.cancel.unwrap_or(&noop_cancel);
        let noop_progress = |_: usize, _: usize| {};
        let on_vae: &dyn Fn(usize, usize) = cb.on_vae.unwrap_or(&noop_progress);
        // Латенты приходят как [кадры, 64], декодеру нужно [1, 64, кадры].
        let z = latents
            .transpose(0, 1)?
            .contiguous()?
            .unsqueeze(0)?
            .to_device(self.options.device)?;
        let audio = vae.decode_tiled(
            &z,
            self.options.vae_core_frames,
            self.options.vae_halo_frames,
            cancel,
            on_vae,
        )?;
        interleave(&audio)
    }

    /// Весь путь целиком.
    pub fn run(&mut self, request: &SongRequest, cb: Callbacks<'_>) -> R<SongResult> {
        let started = Instant::now();
        let semantic = self.generate(request, cb)?;
        if cb.cancelled() {
            return Err(YueError::Cancelled("перед синтезом"));
        }
        let nar_started = Instant::now();
        let latents = self.synthesize(&semantic, cb)?;
        let nar_seconds = nar_started.elapsed().as_secs_f64();
        let vae_started = Instant::now();
        let audio = self.decode(&latents, cb)?;
        let vae_seconds = vae_started.elapsed().as_secs_f64();
        Ok(SongResult {
            audio,
            sample_rate: SAMPLE_RATE,
            channels: 2,
            timing: SongTiming {
                abc: semantic.plan.timing.clone(),
                semantic: semantic.timing.clone(),
                nar_seconds,
                vae_seconds,
                load_seconds: self.load_seconds,
                total_seconds: started.elapsed().as_secs_f64(),
            },
            semantic,
            latents,
        })
    }
}

/// `[1, каналы, отсчёты]` → чередующиеся отсчёты с ограничением в ±1.
pub fn interleave(audio: &Tensor) -> R<Vec<f32>> {
    let dims = audio.dims().to_vec();
    let (channels, samples) = match dims.len() {
        3 => (dims[1], dims[2]),
        2 => (dims[0], dims[1]),
        _ => return Err(YueError::Other(format!("аудио неожидаемой формы {dims:?}"))),
    };
    let flat: Vec<f32> = audio
        .to_dtype(DType::F32)?
        .to_device(Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let mut out = vec![0f32; channels * samples];
    for c in 0..channels {
        for s in 0..samples {
            out[s * channels + c] = flat[c * samples + s].clamp(-1.0, 1.0);
        }
    }
    Ok(out)
}

/// Разложить имя бандла по типовым кандидатам: так ноды и CLI находят модель
/// в каталоге, если путь не задан руками.
pub fn find_bundle(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists())
}

/// Имена бандлов костяка по убыванию предпочтения.
pub const MODEL_NAMES: &[&str] = &["yue2-3b.syn", "yue2.syn"];
/// Имена бандлов декодера.
pub const VAE_NAMES: &[&str] = &["yue2-vae.syn", "yue2-vae-legacy.syn"];

/// Семантический токен → индекс словаря.
pub fn to_vocab(token: u32) -> u32 {
    token + CODEC_OFFSET
}
