use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use synaptix_cli::commands::{
    bench, chat, convert, diff, h3, imagine, inspect, music, quantize, run as run_cmd, speak,
    train, transcribe, video,
};

#[derive(Parser)]
#[command(name = "synaptix", version, about = "Synaptix CLI: model inspection, conversion, inference")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Inspect {
        file: PathBuf,
        #[arg(short, long)]
        verbose: bool,
        #[arg(short = 'f', long)]
        filter: Option<String>,
    },
    Convert {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "syn")]
        format: String,
        #[arg(long)]
        arch: Option<String>,
        #[arg(long)]
        component: Option<String>,
        #[arg(long)]
        mmproj: Option<PathBuf>,
        #[arg(long, default_value = "auto")]
        dtype: String,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        id: Option<String>,
        #[arg(long, default_value_t = false)]
        sha256: bool,
        #[arg(long, default_value_t = false)]
        blake3: bool,
    },
    Bench {
        model: PathBuf,
        #[arg(long, default_value_t = 128)]
        n_tokens: usize,
        /// Принудительная длина prompt (паддинг последним токеном). 0 = как есть.
        #[arg(long, default_value_t = 0)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// Attention-backend: auto|flash-decode|fa2|fa4 (default auto).
        #[arg(long)]
        attn: Option<String>,
        /// Compute dtype: f32|bf16|f16 (default bf16).
        #[arg(long)]
        dtype: Option<String>,
    },
    Run {
        model: PathBuf,
        #[arg(default_value = "")]
        prompt: String,
        /// Прочитать prompt из файла (для длинных контекстов, обходит лимит argv).
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// Размер KV-буфера + RoPE capacity (long-context). По умолчанию
        /// prompt+max_tokens для KV, max_position_embeddings для RoPE.
        #[arg(long)]
        max_seq: Option<usize>,
        /// Attention-backend: auto|flash-decode|fa2|fa4 (default auto).
        #[arg(long)]
        attn: Option<String>,
        /// KV-кеш dtype: bf16 (default) | fp8/mxfp8 (MXFP8 block-scale, 256K-контекст).
        #[arg(long)]
        kv_dtype: Option<String>,
        /// Пресет точности: none (default) | nvfp4 | fp8.
        #[arg(long)]
        quant: Option<String>,
        /// Override compute (активаций): f16|bf16|f32.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Override веса attn+mlp групп: bf16|f16|fp8|nvfp4.
        #[arg(long)]
        storage_dtype: Option<String>,
        /// Override проекции в словарь (lm_head): bf16|f16|fp8|nvfp4.
        #[arg(long)]
        lm_head_dtype: Option<String>,
        /// Override таблицы эмбеддингов: bf16|f16|fp8.
        #[arg(long)]
        embed_dtype: Option<String>,
        /// Отключить CUDA-graph decode. По умолчанию граф ВКЛЮЧЁН на CUDA
        /// (захватывает single-token forward и реплеит, убирая launch-overhead
        /// ~280 ядер/токен). На CPU / не-cuda сборке граф авто-игнорируется.
        #[arg(long, default_value_t = false)]
        no_graph: bool,
        /// Прогрев NVRTC JIT (prefill+1 токен) до замера — для честного бенча.
        #[arg(long, default_value_t = false)]
        warmup: bool,
        /// Требовать MTP-декод на встроенной nextn-голове (greedy). Без флага
        /// MTP включается сам, когда доступен.
        #[arg(long, default_value_t = false)]
        mtp: bool,
        /// Запретить MTP-декод.
        #[arg(long, default_value_t = false)]
        no_mtp: bool,
        /// Отключить CUDA-graph в MTP-декоде.
        #[arg(long, default_value_t = false)]
        no_graph_mtp: bool,
        /// Изображение для мультимодального промпта.
        #[arg(long)]
        image: Option<PathBuf>,
        /// Видео для мультимодального промпта (Muse Glimmer).
        #[arg(long)]
        video: Option<PathBuf>,
        /// Отключить DFlash-спекуляцию (Muse Glimmer).
        #[arg(long, default_value_t = false)]
        no_dflash: bool,
    },
    Chat {
        model: PathBuf,
        #[arg(long)]
        system: Option<String>,
        /// Потолок токенов ответа. 0 = без лимита (генерим до <|im_end|> или пока
        /// не заполнится контекст). Ставь >0 только если хочешь жёсткий потолок.
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
        /// Размер контекста: KV-буфер + RoPE capacity (multi-turn headroom).
        #[arg(long, default_value_t = 4096)]
        context: usize,
        /// Batch size префила: промпт прогоняется кусками по N токенов.
        /// 0 → весь промпт за один forward (single-shot).
        #[arg(long, default_value_t = 0)]
        prefill_batch: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        /// Top-k сэмплинг: оставить k наиболее вероятных токенов. 0 → выкл.
        /// Default 40 (стандарт): ограничивает кандидатов → top-p/сэмпл по 40, а
        /// не по всему словарю (248K) — убирает полную сортировку на токен.
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// Top-p (nucleus) сэмплинг. 1.0 → выкл.
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        /// Min-p сэмплинг: порог = min_p × p(max). 0.0 → выкл.
        #[arg(long, default_value_t = 0.0)]
        min_p: f32,
        /// Repetition penalty (>1 штрафует повтор токенов). 1.0 → выкл.
        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f32,
        /// Seed сэмплинга. 0 → засев от времени при запуске.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// Attention-backend: auto|flash-decode|fa2|fa4 (default auto).
        #[arg(long)]
        attn: Option<String>,
        /// Пресет точности: none (default) | nvfp4 | fp8.
        #[arg(long)]
        quant: Option<String>,
        /// KV-кеш dtype: bf16 (default) | fp8/mxfp8 (MXFP8 block-scale, 256K-контекст).
        #[arg(long)]
        kv_dtype: Option<String>,
        /// Override compute (активаций): f16|bf16|f32.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Override веса attn+mlp групп: bf16|f16|fp8|nvfp4.
        #[arg(long)]
        storage_dtype: Option<String>,
        /// Override проекции в словарь (lm_head): bf16|f16|fp8|nvfp4.
        #[arg(long)]
        lm_head_dtype: Option<String>,
        /// Override таблицы эмбеддингов: bf16|f16|fp8.
        #[arg(long)]
        embed_dtype: Option<String>,
        /// Отключить reasoning-режим (<think>): enable_thinking=false в
        /// chat-template. Для reasoning-моделей (qwen3.6) ответ без размышлений.
        #[arg(long, default_value_t = false)]
        no_think: bool,
    },
    Diff {
        file_a: PathBuf,
        file_b: PathBuf,
        #[arg(long, default_value_t = 1e-4)]
        atol: f32,
        #[arg(long, default_value_t = 1e-3)]
        rtol: f32,
    },
    Train {
        model: PathBuf,
        data: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 8)]
        lora_r: usize,
        #[arg(long, default_value_t = 16.0)]
        lora_alpha: f32,
        #[arg(long, default_value_t = 1e-4)]
        lr: f64,
        #[arg(long, default_value_t = 3)]
        epochs: usize,
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
    },
    Quantize,
    /// Транскрибация аудио (Whisper ASR): WAV → текст.
    Transcribe {
        model: PathBuf,
        audio: PathBuf,
        /// Язык ISO-639-1 (en|ru|...). Опущено → авто-детекция.
        #[arg(long)]
        language: Option<String>,
        /// Задача: transcribe (default) | translate (→ английский).
        #[arg(long, default_value = "transcribe")]
        task: String,
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Compute dtype: f32 (default) | f16 | bf16.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Выводить сегменты с временными метками вместо сплошного текста.
        #[arg(long, default_value_t = false)]
        timestamps: bool,
    },
    /// Синтез речи (VoxCPM2 TTS): TEXT → WAV (48 кГц).
    Speak {
        /// Бандл voxcpm2.syn.
        bundle: PathBuf,
        /// Текст для озвучивания.
        text: String,
        #[arg(short, long, default_value = "speak.wav")]
        output: PathBuf,
        /// Reference WAV для клонирования голоса (изолированный промпт).
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Prompt WAV для режима continuation (вместе с --prompt-text).
        #[arg(long)]
        prompt_wav: Option<PathBuf>,
        /// Транскрипт prompt-аудио (вместе с --prompt-wav).
        #[arg(long)]
        prompt_text: Option<String>,
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Compute dtype: cpu→f32, cuda→bf16 по умолчанию; f32|f16|bf16.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Classifier-free guidance.
        #[arg(long, default_value_t = 2.0)]
        cfg: f32,
        /// Число шагов диффузии (CFM).
        #[arg(long, default_value_t = 10)]
        steps: usize,
        #[arg(long, default_value_t = 1988)]
        seed: u64,
        /// Максимум патчей генерации.
        #[arg(long, default_value_t = 2000)]
        max_len: usize,
    },
    /// Генерация музыки по тексту (ACE-Step v1.5): CAPTION → WAV (48 кГц).
    Music {
        /// Текстовое описание трека (жанр/настроение/инструменты).
        caption: String,
        #[arg(short, long, default_value = "music.wav")]
        output: PathBuf,
        /// Лирика (пусто → инструментал).
        #[arg(long, default_value = "")]
        lyrics: String,
        /// Директория с .syn-бандлами ACE-Step (lm/text-encoder/dit/vae).
        #[arg(long, default_value = "storage/syn_models")]
        models: PathBuf,
        /// Override пути 5Hz AR LM (.syn).
        #[arg(long)]
        lm: Option<PathBuf>,
        /// Override пути text-энкодера Qwen3-Embedding (.syn).
        #[arg(long)]
        text_encoder: Option<PathBuf>,
        /// Override пути DiT xl-base (.syn).
        #[arg(long)]
        dit: Option<PathBuf>,
        /// Override пути VAE (.syn).
        #[arg(long)]
        vae: Option<PathBuf>,
        /// Длительность: "auto" (Phase-1 CoT предсказывает сам) или число секунд.
        #[arg(long, default_value = "auto")]
        duration: String,
        /// Число шагов диффузии (xl-base ~32).
        #[arg(long, default_value_t = 32)]
        steps: usize,
        /// CFG диффузии (xl-base ~7; 1.0 = выкл CFG/APG).
        #[arg(long, default_value_t = 7.0)]
        cfg: f32,
        /// Timestep shift (xl-base 3.0).
        #[arg(long, default_value_t = 3.0)]
        shift: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// AR-семплинг: temperature.
        #[arg(long, default_value_t = 0.85)]
        temperature: f32,
        /// AR-семплинг: top-p.
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        /// AR-семплинг: top-k (0 = выкл).
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// AR-семплинг: min-p (0 = выкл).
        #[arg(long, default_value_t = 0.0)]
        min_p: f32,
        /// AR-CFG (LM classifier-free guidance) scale.
        #[arg(long, default_value_t = 2.0)]
        lm_cfg: f32,
        /// Phase-1 CoT (LM сам генерит метаданные перед кодами).
        #[arg(long)]
        use_cot: bool,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// Compute dtype: bf16 (default) | f16 | f32.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Квант весов DiT: none (default) | nvfp4 | mxfp8 (режет VRAM/ускоряет denoise).
        #[arg(long)]
        quant: Option<String>,
        /// Квант весов LM + text-enc: none (default) | nvfp4 | mxfp8 (форсит F16-compute энкодеру).
        #[arg(long)]
        quant_encoder: Option<String>,
        /// retake: дисперсия вариации [0,1] (0 = обычный text2music; >0 включает retake-микс).
        #[arg(long, default_value_t = 0.0)]
        retake_variance: f32,
        /// retake: seed второго шума, миксуемого при retake_variance>0.
        #[arg(long, default_value_t = 1)]
        retake_seed: u64,
        /// Режим: text2music (default) | retake | repaint | extend | edit.
        #[arg(long, default_value = "text2music")]
        mode: String,
        /// Исходное аудио (48 kHz wav) для repaint/extend/edit → VAE-латент.
        #[arg(long)]
        src_audio: Option<PathBuf>,
        /// repaint/extend: начало региона, сек.
        #[arg(long, default_value_t = 0.0)]
        repaint_start: f32,
        /// repaint/extend: конец региона, сек (<0 = до конца).
        #[arg(long, default_value_t = -1.0)]
        repaint_end: f32,
        /// repaint: сила [0,1] (0=макс. сохранение src, 1=полная регенерация региона).
        #[arg(long, default_value_t = 0.5)]
        repaint_strength: f32,
        /// edit: нижняя граница окна расписания [0,1].
        #[arg(long, default_value_t = 0.0)]
        edit_n_min: f32,
        /// edit: верхняя граница окна расписания [0,1] (уровень ре-шума src).
        #[arg(long, default_value_t = 1.0)]
        edit_n_max: f32,
        /// edit: исходный (старый) caption для source-ветки.
        #[arg(long, default_value = "")]
        edit_source_caption: String,
        /// edit: исходная (старая) лирика для source-ветки.
        #[arg(long, default_value = "")]
        edit_source_lyric: String,
        /// Выключить 5Hz AR-LM (turbo: DiT из шума + silence-src). Требует явную --duration.
        #[arg(long, default_value_t = false)]
        no_ar: bool,
        /// Метаданные: BPM (по умолчанию N/A — модель решает сама).
        #[arg(long)]
        bpm: Option<u32>,
        /// Метаданные: keyscale (напр. "A minor"; пусто = N/A).
        #[arg(long, default_value = "")]
        keyscale: String,
        /// Метаданные: timesignature (напр. "4/4", "6/8"; пусто = N/A).
        #[arg(long, default_value = "")]
        timesig: String,
        /// Нормализация выхода: peak | rms | off.
        #[arg(long, default_value = "peak")]
        norm: String,
    },
    /// Генерация изображения по тексту (SDXL txt2img): PROMPT → PNG.
    Imagine {
        /// HF-директория SDXL (text_encoder/, unet/, vae/, tokenizer/...).
        model: PathBuf,
        prompt: String,
        #[arg(short, long, default_value = "out.png")]
        output: PathBuf,
        /// Негативный промпт (что НЕ должно быть на картинке).
        #[arg(short = 'n', long, default_value = "")]
        negative: String,
        #[arg(long, default_value_t = 30)]
        steps: usize,
        /// Сила CFG (classifier-free guidance). SDXL-base: ~5-8.
        #[arg(long, default_value_t = 5.0)]
        cfg: f32,
        #[arg(long, default_value_t = 1024)]
        height: usize,
        #[arg(long, default_value_t = 1024)]
        width: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// cpu | cuda. CUDA ~много быстрее (1024²×25 ≈ 20с bf16 vs часы CPU).
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Compute dtype: f16 | bf16 | f32. Дефолт: bf16 на CUDA, f32 на CPU.
        /// VAE всегда F32 (f16/bf16 overflow).
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Квант весов: nvfp4 | mxfp8 (как в LLM). Режет VRAM (FLUX 23GB→~6/12GB),
        /// модель влезает резидентно на высоких разрешениях. Дефолт: none (dense).
        #[arg(long)]
        quant: Option<String>,
        /// Алиас --quant (storage-dtype: nvfp4 | mxfp8 | none).
        #[arg(long)]
        storage_dtype: Option<String>,
    },
    /// Генерация видео (+аудио) LTX-2.3 по текстовому промпту (живой Gemma).
    Video {
        /// LTX-2.3 .safetensors (DiT+VAE+vocoder+проекции).
        model: PathBuf,
        /// Промпт: сцена + (для аудио) описание звука/речи.
        prompt: String,
        #[arg(short, long, default_value = "out.mp4")]
        output: PathBuf,
        /// Директория Gemma-3-12B (text-энкодер).
        #[arg(long, default_value = "models/gemma-3-12b-qat")]
        gemma: PathBuf,
        /// Явное число кадров (округляется к 8·(f−1)+1). Переопределяет --duration.
        #[arg(long)]
        frames: Option<usize>,
        /// Длительность: «10s», «2.5s», «1m» или число секунд. Кадры = duration·fps.
        #[arg(long, default_value = "10s")]
        duration: String,
        #[arg(long, default_value_t = 1024)]
        width: usize,
        #[arg(long, default_value_t = 576)]
        height: usize,
        /// Кадров/сек: 24 | 25 | 48 | 50.
        #[arg(long, default_value_t = 24.0)]
        fps: f64,
        /// Без аудио-потока (только видео, VideoDit вместо AvDit).
        #[arg(long)]
        no_audio: bool,
        /// Пайплайн: one-stage|two-stage|av|ti2v-two-stage|a2v|keyframe|ic-lora|... .
        /// Переопределяет --two-stage/--no-audio. См. --list-pipelines.
        #[arg(long)]
        pipeline: Option<String>,
        /// Напечатать список пайплайнов и выйти.
        #[arg(long)]
        list_pipelines: bool,
        /// Two-stage HQ distilled: stage1 A/V (полразрешения) → spatial-upscaler ×2
        /// (видео) → stage2-refine A/V (аудио ре-нойзится и рефайнится, как в офиц.
        /// distilled-пайплайне). Требует --upscaler; --no-audio отключает аудио.
        #[arg(long)]
        two_stage: bool,
        /// Путь к spatial-upscaler ×2 .safetensors (обязателен при --two-stage).
        #[arg(long)]
        upscaler: Option<PathBuf>,
        /// Пропустить stage2-refine (только upscale+decode, ≈вдвое быстрее).
        #[arg(long)]
        no_refine: bool,
        /// LoRA-адаптер для мерджа в веса DiT при загрузке (distilled-lora-384).
        #[arg(long)]
        lora: Option<PathBuf>,
        /// Сила LoRA (официальный дефолт 1.0). Для two-stage см. per-stage флаги.
        #[arg(long, default_value_t = 1.0)]
        lora_strength: f32,
        /// Two-stage: сила LoRA на stage1 (офиц. HQ деф. 0.25; distilled-чекпойнт — 0).
        #[arg(long, default_value_t = 0.0)]
        lora_strength_stage1: f32,
        /// Two-stage: сила LoRA на stage2-refine (офиц. HQ 0.5, ti2v ~0.8;
        /// distilled-чекпойнт — 0). Дефолт: --lora-strength.
        #[arg(long)]
        lora_strength_stage2: Option<f32>,
        /// Negative prompt для CFG (multimodal guidance, не-distilled чекпойнт).
        #[arg(long, default_value = "")]
        negative_prompt: String,
        /// CFG scale (1.0 = выкл). Включает guided stage1 на two-stage (Фаза 3).
        #[arg(long, default_value_t = 1.0)]
        cfg_scale: f32,
        /// STG scale (0.0 = выкл) — spatio-temporal guidance.
        #[arg(long, default_value_t = 0.0)]
        stg_scale: f32,
        /// Число шагов guided stage1 (LTX2Scheduler). Дефолт 30.
        #[arg(long, default_value_t = 30)]
        steps: usize,
        /// Conditioning-изображение (image→video): кадр 0 фиксируется на это фото.
        #[arg(long)]
        image: Option<PathBuf>,
        /// Сила image-conditioning (1.0 = полная замена кадра 0).
        #[arg(long, default_value_t = 1.0)]
        image_strength: f32,
        /// Пиксель-кадр для image-conditioning: 0 = replace (image→video),
        /// >0 = keyframe (append). Одна стадия.
        #[arg(long, default_value_t = 0)]
        image_frame: usize,
        /// Исходное видео для retake (перегенерация региона). С --retake-start/-end.
        #[arg(long)]
        video: Option<PathBuf>,
        /// Retake: начало региона перегенерации (секунды).
        #[arg(long, default_value_t = 0.0)]
        retake_start: f64,
        /// Retake: конец региона перегенерации (секунды).
        #[arg(long, default_value_t = 1e9)]
        retake_end: f64,
        /// IC-LoRA reference-видео (control-сигнал: depth/pose/edges). С --lora <ic-lora>.
        #[arg(long)]
        ref_video: Option<PathBuf>,
        /// IC-LoRA: downscale reference относительно target (из метаданных LoRA, обычно 1/2).
        #[arg(long, default_value_t = 1)]
        ref_downscale: usize,
        /// IC-LoRA: сила reference-conditioning (1.0 = reference clean).
        #[arg(long, default_value_t = 1.0)]
        ref_strength: f32,
        /// Аудио-файл речи для lipdub (wav; с --ref-video = лицо). 16kHz ресемпл авто.
        #[arg(long)]
        audio: Option<PathBuf>,
        /// Препроцессор reference-видео: none | canny | depth (control-сигнал
        /// для union-control IC-LoRA, как ComfyUI Canny/Depth-ноды).
        #[arg(long, default_value = "none")]
        ref_preprocess: String,
        /// Директория Depth Anything V2 (для --ref-preprocess depth).
        #[arg(long, default_value = "models/depth-anything-v2-small")]
        depth_model: PathBuf,
        /// Canny: нижний порог гистерезиса (доля max-градиента).
        #[arg(long, default_value_t = 0.1)]
        canny_low: f32,
        /// Canny: верхний порог гистерезиса.
        #[arg(long, default_value_t = 0.2)]
        canny_high: f32,
        /// Квант блоков DiT: none|mxfp8|nvfp4. none → dense bf16 + streaming-offload
        /// (host-RAM≈0); квант → резидентно на GPU (меньше VRAM).
        #[arg(long)]
        quant_transformer: Option<String>,
        /// Квант весов Gemma: none|mxfp8|nvfp4. Дефолт mxfp8 (12B→~12GB).
        #[arg(long)]
        quant_encoder: Option<String>,
        /// Compute dtype: f16 | bf16. Дефолт bf16.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// cpu | cuda. Дефолт cuda.
        #[arg(long, default_value = "cuda")]
        device: String,
        /// NAG negative-prompt (Normalized Attention Guidance, stage1
        /// cross-attention). Дефолт подавляет субтитры/текст/вотермарки;
        /// --nag-prompt "" — выключить NAG.
        #[arg(long, default_value = synaptix_video_ltx23::pipeline::DEFAULT_NAG_PROMPT)]
        nag_prompt: Option<String>,
        /// NAG scale (экстраполяция pos·s − neg·(s−1)).
        #[arg(long, default_value_t = synaptix_video_ltx23::pipeline::NAG_DEFAULT_SCALE)]
        nag_scale: f32,
        /// NAG alpha (бленд guidance с pos).
        #[arg(long, default_value_t = synaptix_video_ltx23::pipeline::NAG_DEFAULT_ALPHA)]
        nag_alpha: f32,
        /// NAG tau (L1-кламп ||guidance||/||pos||).
        #[arg(long, default_value_t = synaptix_video_ltx23::pipeline::NAG_DEFAULT_TAU)]
        nag_tau: f32,
        /// Принудительный host-stream offload квантованного DiT (иначе авто по VRAM).
        #[arg(long)]
        force_offload: bool,
        /// Печать таймингов text-encoding ([LTX_PROF]).
        #[arg(long)]
        prof: bool,
        /// Режим стриминга DiT-блоков при dense-offload:
        /// 0=легаси-карусель, 1=слоты, 2=слоты+CUDA-graph (дефолт — см. runtime).
        #[arg(long)]
        block_mode: Option<usize>,
    },
    /// MiniMax-H3 33B: текст/кадры → видео + синхронное стерео 32 кГц.
    H3 {
        /// Каталог модели (корень MiniMax-H3 или сразу FL2VA/Ref2VA).
        #[arg(long)]
        model_dir: PathBuf,
        /// Текстовый промпт.
        #[arg(default_value = "")]
        prompt: String,
        /// Негативный промпт (нужен при cfg > 1).
        #[arg(long)]
        negative_prompt: Option<String>,
        #[arg(short, long, default_value = "h3.mp4")]
        output: PathBuf,
        /// Каталог энкодера Qwen3-VL (по умолчанию <model_dir>/text_encoder).
        #[arg(long)]
        encoder: Option<PathBuf>,
        /// Первый кадр (fl2va).
        #[arg(long)]
        first_frame: Option<PathBuf>,
        /// Последний кадр (fl2va).
        #[arg(long)]
        last_frame: Option<PathBuf>,
        #[arg(long, default_value_t = 1344)]
        width: usize,
        #[arg(long, default_value_t = 768)]
        height: usize,
        /// Длительность в секундах (снапится на сетку 17k+5 кадров при 24 fps).
        #[arg(long, default_value_t = 5.0)]
        duration: f64,
        /// Явное число кадров (перебивает --duration).
        #[arg(long)]
        frames: Option<usize>,
        /// Число шагов денойзинга (0 = из пресета пайплайна).
        #[arg(long, default_value_t = 0)]
        steps: usize,
        /// CFG scale (0 = из пресета; 1.0 = без негатива, режим Turbo).
        #[arg(long, default_value_t = 0.0)]
        cfg_scale: f32,
        #[arg(long)]
        seed: Option<u64>,
        /// LoRA-адаптер (Turbo LoRA для 4-8 шагов).
        #[arg(long)]
        lora: Option<PathBuf>,
        #[arg(long, default_value_t = 1.0)]
        lora_strength: f32,
        /// Квантование DiT: none|mxfp8|nvfp4 (дефолт nvfp4).
        #[arg(long)]
        quant_transformer: Option<String>,
        /// Квантование энкодера: none|mxfp8|nvfp4 (дефолт mxfp8).
        #[arg(long)]
        quant_encoder: Option<String>,
        /// Compute-dtype: bf16|f16 (дефолт bf16).
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Стратегия памяти: auto|precomputed-adaln|block-offload.
        #[arg(long, default_value = "auto")]
        memory_mode: String,
        /// Пресет пайплайна (см. --list-pipelines).
        #[arg(long)]
        pipeline: Option<String>,
        /// Показать доступные пресеты и выйти.
        #[arg(long, default_value_t = false)]
        list_pipelines: bool,
        /// Вариант весов: fl2va|ref2va.
        #[arg(long)]
        variant: Option<String>,
        #[arg(long, default_value_t = 0)]
        device: usize,
        #[arg(long, default_value_t = false)]
        prof: bool,
        /// Сохранить рядом с mp4 отдельный wav.
        #[arg(long, default_value_t = false)]
        keep_wav: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res: Result<(), Box<dyn std::error::Error>> = match cli.command {
        Commands::Inspect { file, verbose, filter } => {
            inspect::run(inspect::InspectArgs { file, verbose, filter })
        }
        Commands::Convert {
            input, output, format, arch, component, mmproj, dtype, tokenizer, id, sha256, blake3,
        } => convert::run(convert::ConvertArgs {
            input, output, format, arch, component, mmproj, dtype, tokenizer, id, sha256, blake3,
        }),
        Commands::Bench { model, n_tokens, prompt_tokens, batch_size, warmup, device, attn, dtype } => {
            bench::run(bench::BenchArgs { model, n_tokens, prompt_tokens, batch_size, warmup, device, attn, dtype })
        }
        Commands::Run {
            model, prompt, prompt_file, max_tokens, temperature, seed, device, max_seq, attn, kv_dtype,
            quant, compute_dtype, storage_dtype, lm_head_dtype, embed_dtype, no_graph,
            warmup, mtp, no_mtp, no_graph_mtp, image, video, no_dflash,
        } => {
            let prompt = match prompt_file {
                Some(pf) => std::fs::read_to_string(&pf)
                    .unwrap_or_else(|e| { eprintln!("prompt-file {}: {e}", pf.display()); std::process::exit(2) }),
                None => prompt,
            };
            run_cmd::run(run_cmd::RunArgs {
                model,
                prompt,
                max_tokens,
                temperature,
                seed,
                device,
                max_seq,
                attn,
                kv_dtype,
                quant,
                compute_dtype,
                storage_dtype,
                lm_head_dtype,
                embed_dtype,
                graph: !no_graph,
                mtp,
                no_mtp,
                no_graph_mtp,
                image,
                video,
                no_dflash,
                warmup,
            })
        }
        Commands::Chat {
            model,
            system,
            max_tokens,
            context,
            prefill_batch,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            seed,
            device,
            attn,
            quant,
            kv_dtype,
            compute_dtype,
            storage_dtype,
            lm_head_dtype,
            embed_dtype,
            no_think,
        } => chat::run(chat::ChatArgs {
            model,
            system,
            max_tokens,
            context,
            prefill_batch,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            seed,
            device,
            attn,
            quant,
            kv_dtype,
            compute_dtype,
            storage_dtype,
            lm_head_dtype,
            embed_dtype,
            no_think,
        }),
        Commands::Diff { file_a, file_b, atol, rtol } => {
            diff::run(diff::DiffArgs { file_a, file_b, atol, rtol })
        }
        Commands::Train { model, data, output, lora_r, lora_alpha, lr, epochs, batch_size } => {
            train::run(train::TrainArgs {
                model, data, output, lora_r, lora_alpha, lr, epochs, batch_size,
            })
        }
        Commands::Quantize => quantize::run(),
        Commands::Transcribe { model, audio, language, task, device, compute_dtype, timestamps } => {
            transcribe::run(transcribe::TranscribeArgs {
                model,
                audio,
                language,
                task,
                device,
                compute_dtype,
                timestamps,
            })
        }
        Commands::Speak {
            bundle, text, output, reference, prompt_wav, prompt_text, device, compute_dtype,
            cfg, steps, seed, max_len,
        } => speak::run(speak::SpeakArgs {
            bundle, text, output, reference, prompt_wav, prompt_text, device, compute_dtype,
            cfg, steps, seed, max_len,
        }),
        Commands::Music {
            caption, output, lyrics, models, lm, text_encoder, dit, vae, duration, steps, cfg,
            shift, seed, temperature, top_p, top_k, min_p, lm_cfg, use_cot, device, compute_dtype,
            quant, quant_encoder, retake_variance, retake_seed, mode, src_audio, repaint_start,
            repaint_end, repaint_strength, edit_n_min, edit_n_max, edit_source_caption,
            edit_source_lyric, no_ar, bpm, keyscale, timesig, norm,
        } => music::run(music::MusicArgs {
            caption, output, lyrics, models, lm, text_encoder, dit, vae, duration, steps, cfg,
            shift, seed, temperature, top_p, top_k, min_p, lm_cfg, use_cot, device, compute_dtype,
            quant, quant_encoder, retake_variance, retake_seed, mode, src_audio, repaint_start,
            repaint_end, repaint_strength, edit_n_min, edit_n_max, edit_source_caption,
            edit_source_lyric, use_ar: !no_ar, bpm, keyscale, timesig, norm,
        }),
        Commands::Imagine {
            model, prompt, output, negative, steps, cfg, height, width, seed, device, compute_dtype,
            quant, storage_dtype,
        } => imagine::run(imagine::ImagineArgs {
            model,
            prompt,
            output,
            negative,
            steps,
            guidance_scale: cfg,
            height,
            width,
            seed,
            device,
            compute_dtype,
            quant,
            storage_dtype,
        }),
        Commands::Video {
            model, prompt, output, gemma, frames, duration, width, height, fps, no_audio,
            pipeline, list_pipelines, two_stage, upscaler, no_refine, lora, lora_strength,
            lora_strength_stage1, lora_strength_stage2,
            negative_prompt, cfg_scale, stg_scale, steps, image, image_strength, image_frame,
            video, retake_start, retake_end, ref_video, ref_downscale, ref_strength, audio,
            ref_preprocess, canny_low, canny_high, depth_model,
            quant_transformer, quant_encoder, compute_dtype, device,
            nag_prompt, nag_scale, nag_alpha, nag_tau, force_offload, prof, block_mode,
        } => video::run(video::VideoArgs {
            model, prompt, output, gemma, frames, duration, width, height, fps, no_audio,
            pipeline, list_pipelines, two_stage, upscaler, no_refine, lora, lora_strength,
            lora_strength_stage1, lora_strength_stage2,
            negative_prompt, cfg_scale, stg_scale, steps, image, image_strength, image_frame,
            video, retake_start, retake_end, ref_video, ref_downscale, ref_strength, audio,
            ref_preprocess, canny_low, canny_high, depth_model,
            quant_transformer, quant_encoder, compute_dtype, device,
            nag_prompt, nag_scale, nag_alpha, nag_tau, force_offload, prof, block_mode,
        }),
        Commands::H3 {
            model_dir, prompt, negative_prompt, output, encoder, first_frame, last_frame,
            width, height, duration, frames, steps, cfg_scale, seed, lora, lora_strength,
            quant_transformer, quant_encoder, compute_dtype, memory_mode, pipeline,
            list_pipelines, variant, device, prof, keep_wav,
        } => h3::run(h3::H3Args {
            model_dir, prompt, negative_prompt, output, encoder, first_frame, last_frame,
            width, height, duration, frames, steps, cfg_scale, seed, lora, lora_strength,
            quant_transformer, quant_encoder, compute_dtype, memory_mode, pipeline,
            list_pipelines, variant, device, prof, keep_wav,
        }),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}
