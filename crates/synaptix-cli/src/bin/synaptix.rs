use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use synaptix_cli::commands::{
    bench, chat, convert, device, diff, h3, imagine, inspect, music, podcast, quantize, run as run_cmd, sheet, song,
    speak,
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
    /// CUDA-карты: compute capability, цель NVRTC, block-scale MMA/TMA.
    Devices,
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
        /// Compute dtype: f32|bf16|f16 (алиас --compute-dtype).
        #[arg(long, alias = "compute-dtype")]
        dtype: Option<String>,
        #[arg(long, default_value_t = 1, help = "Замеров подряд (среднее)")]
        repeat: usize,
        #[arg(long, help = "Точность: без флагов — оптимальный профиль; none | nvfp4 | fp8 | mxfp8 | sq1…sq8")]
        quant: Option<String>,
        #[arg(long)]
        storage_dtype: Option<String>,
        #[arg(long)]
        lm_head_dtype: Option<String>,
        #[arg(long)]
        embed_dtype: Option<String>,
        #[arg(long)]
        kv_dtype: Option<String>,
        #[arg(long, default_value_t = false)]
        no_graph: bool,
        #[arg(long, default_value_t = false)]
        no_spec: bool,
        #[arg(long, default_value_t = 0)]
        prefill_batch: usize,
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
        #[arg(long, help = "Temperature (без флага: 1.0; с --chat — из пресета модели)")]
        temperature: Option<f32>,
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
        /// Точность: без флагов — оптимальный профиль движка под модель; none (dense) | nvfp4 | fp8 | mxfp8 | sq1…sq8.
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
        /// Требовать MTP-декод на встроенной nextn-голове (greedy и сэмплинг). Без флага
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
        #[arg(long, default_value_t = false, help = "Muse Glimmer: без lookup-спекуляции на greedy")]
        no_lookup: bool,
        #[arg(long, help = "Пресет сэмплинга модели (с --chat; по умолчанию под режим размышлений)")]
        preset: Option<String>,
        #[arg(long, help = "Top-k (0 — выкл)")]
        top_k: Option<usize>,
        #[arg(long, help = "Top-p (1.0 — выкл)")]
        top_p: Option<f32>,
        #[arg(long, help = "Min-p (0 — выкл)")]
        min_p: Option<f32>,
        #[arg(long, help = "Repetition penalty (1.0 — выкл)")]
        repetition_penalty: Option<f32>,
        #[arg(long, help = "Окно repetition penalty (0 — весь контекст)")]
        repeat_last_n: Option<usize>,
        #[arg(long, help = "Presence penalty")]
        presence_penalty: Option<f32>,
        #[arg(long, help = "Frequency penalty")]
        frequency_penalty: Option<f32>,
        #[arg(long, default_value_t = 0, help = "Чанк префилла в токенах (0 — по умолчанию движка)")]
        prefill_batch: usize,
        #[arg(long, default_value_t = false, help = "Обернуть промпт chat-шаблоном модели (через фасад; сэмплинг — из пресета модели)")]
        chat: bool,
        #[arg(long, help = "Системный промпт (с --chat)")]
        system: Option<String>,
        #[arg(long, default_value_t = false, help = "Без размышлений (с --chat)")]
        no_think: bool,
        #[arg(long, help = "Глубина размышлений (с --chat): Qwen3.8 low|medium|xhigh, Muse low…xhigh")]
        reasoning_effort: Option<String>,
        #[arg(long, help = "Стоп-последовательность (повторяется; через фасад)")]
        stop: Vec<String>,
        #[arg(long, help = "Потолок vision-токенов на картинку")]
        max_image_tokens: Option<usize>,
        #[arg(long, help = "Синхронизация слоёв: auto | on | off")]
        layer_sync: Option<String>,
        #[arg(long, help = "Qwen4Exp: кэш экспертов на карте, ГБ (0 — все эксперты на устройстве)")]
        expert_cache_gb: Option<f64>,
        #[arg(long, help = "Qwen4Exp: зеркало весов в host-RAM, ГБ")]
        host_mirror_gb: Option<f64>,
        #[arg(long, default_value_t = false, help = "Qwen4Exp: спекулятивный декод")]
        qwen4_spec: bool,
        #[arg(long, default_value_t = false, help = "Qwen4Exp: таблица эмбеддингов в host-RAM")]
        embed_host: bool,
    },
    /// Интерактивный чат (TUI) через фасад: все LLM-архитектуры, картинки/видео, префикс-KV между ходами.
    Chat {
        model: PathBuf,
        #[arg(long)]
        system: Option<String>,
        /// Потолок токенов ответа. 0 = без лимита (до конца хода или контекста).
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
        /// Размер контекста: KV-буфер + RoPE capacity (multi-turn headroom).
        #[arg(long, default_value_t = 4096)]
        context: usize,
        /// Чанк префилла в токенах. 0 → по умолчанию движка.
        #[arg(long, default_value_t = 0)]
        prefill_batch: usize,
        #[arg(long, help = "Пресет сэмплинга модели (thinking, instruct, recommended, …); по умолчанию — под режим размышлений")]
        preset: Option<String>,
        #[arg(long, help = "Temperature (по умолчанию из пресета модели)")]
        temperature: Option<f32>,
        #[arg(long, help = "Top-k (0 — выкл; по умолчанию из пресета)")]
        top_k: Option<usize>,
        #[arg(long, help = "Top-p (1.0 — выкл; по умолчанию из пресета)")]
        top_p: Option<f32>,
        #[arg(long, help = "Min-p (0 — выкл)")]
        min_p: Option<f32>,
        #[arg(long, help = "Repetition penalty (1.0 — выкл)")]
        repetition_penalty: Option<f32>,
        #[arg(long, help = "Окно repetition penalty в токенах (64; 0 — весь контекст)")]
        repeat_last_n: Option<usize>,
        #[arg(long, help = "Presence penalty")]
        presence_penalty: Option<f32>,
        #[arg(long, help = "Frequency penalty")]
        frequency_penalty: Option<f32>,
        /// Seed сэмплинга. 0 → засев от времени на каждый ход.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "cuda")]
        device: String,
        /// Attention-backend: auto|flash-decode|fa2|fa4 (default auto).
        #[arg(long)]
        attn: Option<String>,
        /// Точность: без флагов — оптимальный профиль движка под модель; none | nvfp4 | fp8 | mxfp8 | sq1…sq8.
        #[arg(long)]
        quant: Option<String>,
        /// KV-кеш dtype: bf16 | f16 | f32 | mxfp8 (fp8).
        #[arg(long)]
        kv_dtype: Option<String>,
        /// Override compute (активаций): f16|bf16|f32.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Override веса attn+mlp групп: bf16|f16|fp8|nvfp4|sqN.
        #[arg(long)]
        storage_dtype: Option<String>,
        /// Override проекции в словарь (lm_head): bf16|f16|fp8|nvfp4.
        #[arg(long)]
        lm_head_dtype: Option<String>,
        /// Override таблицы эмбеддингов: bf16|f16|fp8.
        #[arg(long)]
        embed_dtype: Option<String>,
        /// Без размышлений (enable_thinking=false; у Muse — reasoning_strength=low). В чате: /think on|off.
        #[arg(long, default_value_t = false)]
        no_think: bool,
        #[arg(long, help = "Глубина размышлений (Qwen3.8: low|medium|xhigh, Muse: low…xhigh). В чате: /effort")]
        reasoning_effort: Option<String>,
        #[arg(long, help = "Стоп-последовательность (повторяется)")]
        stop: Vec<String>,
        #[arg(long, help = "Картинка к первому сообщению (повторяется). В чате: /image <путь>")]
        image: Vec<PathBuf>,
        #[arg(long, help = "Видео к первому сообщению (повторяется). В чате: /video <путь>")]
        video: Vec<PathBuf>,
        #[arg(long, help = "Потолок vision-токенов на картинку (по умолчанию — из конфига модели)")]
        max_image_tokens: Option<usize>,
        #[arg(long, default_value_t = false, help = "Выключить CUDA-graph декода")]
        no_graph: bool,
        #[arg(long, default_value_t = false, help = "Выключить спекулятивный декод (MTP/DFlash)")]
        no_spec: bool,
        #[arg(long, help = "Синхронизация слоёв: auto | on | off")]
        layer_sync: Option<String>,
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
    /// Квантование/перекодировка модели в `.syn`: плотные матрицы и уже
    /// квантованные веса (`.syn` NVFP4/MXFP8/SQ, `.gguf` ggml) → формат
    /// `--format` (nvfp4 | mxfp8 | sq1…sq8). Веса считаются на карте по
    /// одному, остальные тензоры и файлы копируются как есть.
    Quantize {
        input: PathBuf,
        output: PathBuf,
        /// Формат attn/mlp/экспертов/головы: nvfp4 | mxfp8 | sq1…sq8.
        #[arg(long, default_value = "sq4")]
        format: String,
        /// Формат проекций внимания, если отличается (например mxfp8).
        #[arg(long)]
        attn: Option<String>,
        /// Формат lm_head (`none` — оставить как есть).
        #[arg(long)]
        lm_head: Option<String>,
        /// Формат эмбеддинга (`none` — как есть; допустимы mxfp8 и sqN).
        #[arg(long)]
        embed: Option<String>,
        /// Не перекодировать уже квантованные веса (только плотные).
        #[arg(long, default_value_t = false)]
        keep_quant: bool,
        #[arg(long, default_value = "cuda:0")]
        device: String,
    },
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
    /// Многоголосый длинный синтез (VibeVoice): SCRIPT → WAV (24 кГц).
    Podcast {
        /// Бандл vibevoice-1.5b.syn / vibevoice-7b.syn.
        bundle: PathBuf,
        /// Сценарий: строки вида "Speaker 1: текст". Без префикса — Speaker 1.
        script: Option<String>,
        /// Файл со сценарием (.txt) вместо позиционного SCRIPT.
        #[arg(long)]
        script_file: Option<PathBuf>,
        #[arg(short, long, default_value = "podcast.wav")]
        output: PathBuf,
        /// Аудио-референс голоса (wav|mp3|ogg|flac). Повторяется по одному на спикера.
        #[arg(long = "voice")]
        voices: Vec<PathBuf>,
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Compute dtype: cpu→f32, cuda→bf16 по умолчанию; f32|f16|bf16.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Classifier-free guidance диффузионной головы.
        #[arg(long, default_value_t = 1.3)]
        cfg: f32,
        /// Число шагов DPM-Solver++ на один акустический кадр.
        #[arg(long, default_value_t = 20)]
        steps: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Потолок числа шагов = max_length_times × длина промпта.
        #[arg(long, default_value_t = 2.0)]
        max_length_times: f32,
        /// Детерминированный прогон без шума (для сверки с эталоном).
        #[arg(long, default_value_t = false)]
        zero_noise: bool,
    },
    /// Песня целиком (YuE2): стиль и лирика → партитура → музыка → WAV 48 кГц стерео.
    Song {
        /// Теги стиля: язык, жанр, инструменты, характер вокала.
        style: String,
        #[arg(short, long, default_value = "song.wav")]
        output: PathBuf,
        /// Лирика (с разметкой вида [verse] / [chorus]).
        #[arg(long, default_value = "")]
        lyrics: String,
        /// Лирика из файла (сильнее --lyrics).
        #[arg(long)]
        lyrics_file: Option<PathBuf>,
        /// Каталог с бандлами YuE2 (yue2-3b.syn, yue2-vae.syn).
        #[arg(long, default_value = "storage/syn_models")]
        models: PathBuf,
        /// Override пути костяка (.syn).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Override пути декодера (.syn).
        #[arg(long)]
        vae: Option<PathBuf>,
        /// Партитура: full (мелодия с аккордами) | melody (только мелодия) | off.
        #[arg(long, default_value = "full")]
        cot: String,
        /// Готовая партитура ABC из файла вместо сгенерированной.
        #[arg(long)]
        abc_file: Option<PathBuf>,
        /// Кавер: запись → мелодия (SheetSage2 из того же бандла) → песня с
        /// `cot = melody`. Сильнее --abc-file.
        #[arg(long)]
        cover: Option<PathBuf>,
        /// Какие мелодии записи взять в кавер: both | vocal | ins.
        #[arg(long, default_value = "both")]
        cover_voices: String,
        #[arg(long, default_value_t = false, help = "Кавер по полной партитуре записи (с аккордами, cot = full)")]
        cover_full: bool,
        #[arg(long, help = "Взять из записи для кавера только первые N секунд")]
        cover_max_seconds: Option<f64>,
        /// Куда сохранить партитуру.
        #[arg(long)]
        save_abc: Option<PathBuf>,
        #[arg(long, default_value_t = 831001)]
        seed: u64,
        /// CFG (по умолчанию 1.0, у режима off — 1.01).
        #[arg(long)]
        cfg: Option<f32>,
        /// Шагов решателя flow matching.
        #[arg(long, default_value_t = 32)]
        steps: usize,
        /// Потолок семантических токенов: 25 на секунду музыки.
        #[arg(long)]
        max_tokens: Option<usize>,
        #[arg(long, help = "Длительность музыки в секундах (25 токенов на секунду; --max-tokens сильнее)")]
        seconds: Option<f32>,
        #[arg(long, help = "До этого числа семантических токенов конец музыки запрещён (200)")]
        min_tokens: Option<usize>,
        #[arg(long, help = "Температура фазы музыки (1.0)")]
        temperature: Option<f32>,
        #[arg(long, help = "Top-p фазы музыки (0.95)")]
        top_p: Option<f32>,
        #[arg(long, help = "Top-k фазы музыки (100)")]
        top_k: Option<usize>,
        #[arg(long, help = "Штраф повторов фазы музыки (1.2)")]
        repetition_penalty: Option<f32>,
        #[arg(long, help = "Окно штрафа повторов фазы музыки, 1..100 (50)")]
        penalty_window: Option<usize>,
        #[arg(long, help = "Потолок токенов партитуры (4096)")]
        abc_max_tokens: Option<usize>,
        #[arg(long, help = "Минимум токенов партитуры (32)")]
        abc_min_tokens: Option<usize>,
        #[arg(long, help = "Температура фазы партитуры (0.7)")]
        abc_temperature: Option<f32>,
        #[arg(long, help = "Top-p фазы партитуры (0.9)")]
        abc_top_p: Option<f32>,
        #[arg(long, help = "Top-k фазы партитуры (30)")]
        abc_top_k: Option<usize>,
        #[arg(long, help = "Штраф повторов фазы партитуры (1.005)")]
        abc_repetition_penalty: Option<f32>,
        #[arg(long, help = "Окно штрафа повторов фазы партитуры, 1..100 (100)")]
        abc_penalty_window: Option<usize>,
        #[arg(long, help = "Окно модели для нарезки акустики на чанки (24576): меньше — меньше памяти, чаще швы")]
        context: Option<usize>,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        /// Тип вычислений костяка: bf16 (эталон) | f32.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Квант весов костяка: nvfp4 | mxfp8.
        #[arg(long)]
        quant: Option<String>,
        /// Тип вычислений декодера: f32 (эталон) | bf16.
        #[arg(long)]
        vae_dtype: Option<String>,
        /// Кадров в ядре тайла декодера (меньше — меньше памяти).
        #[arg(long, default_value_t = 1024)]
        vae_core_frames: usize,
        #[arg(long, default_value_t = 16, help = "Кадров перекрытия тайлов декодера")]
        vae_halo_frames: usize,
        #[arg(long, help = "Сохранить акустические латенты (safetensors) для song-decode")]
        save_latent: Option<PathBuf>,
    },
    /// Латенты YuE2 (song --save-latent) → WAV 48 кГц стерео другим декодером.
    SongDecode {
        latent: PathBuf,
        #[arg(short, long, default_value = "song.wav")]
        output: PathBuf,
        #[arg(long, default_value = "storage/syn_models")]
        models: PathBuf,
        #[arg(long, help = "Бандл декодера (yue2-vae.syn | yue2-vae-legacy.syn)")]
        vae: Option<PathBuf>,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        #[arg(long, help = "f32 (эталон) | bf16")]
        vae_dtype: Option<String>,
        #[arg(long, default_value_t = 1024)]
        vae_core_frames: usize,
        #[arg(long, default_value_t = 16)]
        vae_halo_frames: usize,
    },
    /// Запись → партитура ABC (SheetSage2): мелодия вокала и инструмента,
    /// аккорды, размер, тональность и секции. По умолчанию — без аккордов, как
    /// план кавера для YuE2.
    Sheet {
        /// Запись (любой формат, который читает ffmpeg; без ffmpeg — WAV).
        audio: PathBuf,
        #[arg(short, long, default_value = "score.abc")]
        output: PathBuf,
        /// Каталог с бандлами (SheetSage2 лежит компонентом в yue2-3b.syn).
        #[arg(long, default_value = "storage/syn_models")]
        models: PathBuf,
        /// Бандл с компонентом sheetsage2.
        #[arg(long)]
        model: Option<PathBuf>,
        /// С аккордами (полная партитура).
        #[arg(long, default_value_t = false)]
        full: bool,
        /// Какие мелодии оставить: both | vocal | ins.
        #[arg(long, default_value = "both")]
        voices: String,
        /// Взять только первые N секунд.
        #[arg(long)]
        max_seconds: Option<f64>,
        #[arg(long, help = "Перекрытие окон транскрипции, с (200)")]
        overlap_seconds: Option<f64>,
        #[arg(long, help = "Заглядывание вперёд окна, с (100)")]
        lookahead_seconds: Option<f64>,
        #[arg(long, default_value = "cuda:0")]
        device: String,
        /// bf16 (эталонный режим релиза) | f32.
        #[arg(long)]
        compute_dtype: Option<String>,
        /// Сохранить токены окон в JSON (для сверки с релизом).
        #[arg(long)]
        tokens_json: Option<PathBuf>,
    },
    /// Упаковать релиз SheetSage2 (+ MERT-v2-FullSong) компонентом в бандл.
    SheetPack {
        /// Каталог релиза SheetSage2 (config.json, model.safetensors).
        #[arg(long)]
        sheetsage: PathBuf,
        /// Каталог MERT-v2-FullSong (нужен релизу-адаптеру).
        #[arg(long)]
        mert: Option<PathBuf>,
        /// Бандл, в который дописать компонент (обычно yue2-3b.syn).
        #[arg(long)]
        into: PathBuf,
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
        /// Число шагов диффузии (по варианту DiT: turbo 8, base 32, sft 50).
        #[arg(long)]
        steps: Option<usize>,
        /// CFG диффузии (turbo 1.0, base/sft 7.0; 1.0 = выкл CFG/APG).
        #[arg(long)]
        cfg: Option<f32>,
        /// Timestep shift (turbo/base 3.0, sft 1.0).
        #[arg(long)]
        shift: Option<f32>,
        #[arg(long, default_value_t = 42, help = "Seed (0 — случайный)")]
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
        /// retake: seed второго шума, миксуемого при retake_variance>0 (0 — случайный).
        #[arg(long, default_value_t = 1)]
        retake_seed: u64,
        /// Режим: text2music (default) | retake | repaint | extend | edit | extract | cover.
        #[arg(long, default_value = "text2music")]
        mode: String,
        /// extract: дорожка — vocals | backing_vocals | drums | bass | guitar | keyboard |
        /// percussion | strings | synth | fx | brass | woodwinds.
        #[arg(long, default_value = "vocals")]
        track: String,
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
        #[arg(long, default_value_t = 1, help = "edit: усреднить N прогонов source-ветки")]
        edit_n_avg: usize,
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
        /// Прогонов подряд с резидентным кэшем компонентов (как «Держать в
        /// памяти» у нод synthos); seed растёт на единицу, пишется последний.
        #[arg(long, default_value_t = 1)]
        repeat: u32,
        #[arg(long, default_value_t = false, help = "DCW-коррекция денойза в вейвлет-домене")]
        dcw: bool,
        #[arg(long, default_value = "double", help = "DCW: low | high | double | pix")]
        dcw_mode: String,
        #[arg(long, default_value = "think", help = "DCW-пресет масштабов: think (0.02/0.06) | no-think (0.05/0.02)")]
        dcw_preset: String,
        #[arg(long, help = "DCW: масштаб низких частот (сильнее пресета)")]
        dcw_scaler: Option<f32>,
        #[arg(long, help = "DCW: масштаб высоких частот (сильнее пресета)")]
        dcw_high_scaler: Option<f32>,
        #[arg(long, default_value_t = 2, help = "Каналов в WAV: 2 (стерео VAE) | 1 (левый канал)")]
        channels: u16,
        #[arg(long, help = "Сохранить латент DiT [1, 64, T] (safetensors)")]
        save_latent: Option<PathBuf>,
    },
    /// Генерация изображения по тексту (SDXL txt2img): PROMPT → PNG.
    Imagine {
        /// Бандл .syn или каталог diffusers: SDXL, FLUX.1, FLUX.2, Qwen-Image (Edit/Edit-Plus), Qwen-Image 2.1.
        model: PathBuf,
        prompt: String,
        #[arg(short, long, default_value = "out.png")]
        output: PathBuf,
        #[arg(short = 'n', long, help = "Негатив: SDXL — CFG-негатив; Qwen-Image — true CFG (по умолчанию \" \"); Qwen-Image 2.1 — CFG при --cfg > 1")]
        negative: Option<String>,
        #[arg(long, help = "Шагов (0 или без флага — по модели: SDXL 30, FLUX.1 dev 28 / schnell 4, FLUX.2 по варианту, Qwen-Image 50/40, Qwen 2.1 40)")]
        steps: Option<usize>,
        #[arg(long, help = "CFG/guidance (по модели: SDXL 5.0, FLUX.1 3.5, FLUX.2 4.0 / klein 1.0, Qwen-Image 4.0, Qwen 2.1 1.0)")]
        cfg: Option<f32>,
        #[arg(long, help = "Высота (по умолчанию — с --init-image/референса или 1024)")]
        height: Option<usize>,
        #[arg(long, help = "Ширина (по умолчанию — с --init-image/референса или 1024)")]
        width: Option<usize>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[arg(long, help = "Активации FLUX.1: f16 | bf16 | f32 (по умолчанию bf16, f16 при кванте, f32 на CPU)")]
        compute_dtype: Option<String>,
        #[arg(long, help = "Квант весов DiT/UNet: none | nvfp4 | mxfp8")]
        quant: Option<String>,
        #[arg(long, help = "Алиас --quant")]
        storage_dtype: Option<String>,
        #[arg(long, help = "Референсы: FLUX.2 (до 10), Qwen-Image Edit (1) / Edit-Plus (до 4), Qwen-Image 2.1 (до 10)")]
        image: Vec<PathBuf>,
        #[arg(long, default_value_t = 1024, help = "Qwen-Image 2.1: output_resolution — сторона ~площади референсов и выхода")]
        resolution: usize,
        #[arg(long, help = "img2img: исходная картинка (SDXL, FLUX.1, FLUX.2)")]
        init_image: Option<PathBuf>,
        #[arg(long, help = "img2img: сила, доля шума 0..1 (SDXL 0.6, FLUX 0.75)")]
        strength: Option<f32>,
        #[arg(long, default_value = "stretch", help = "Подгонка --init-image под размер: stretch | crop")]
        fit: String,
        #[arg(long, default_value = "auto", help = "Где держать DiT (FLUX.1, FLUX.2, Qwen-Image, Qwen 2.1): auto | resident | stream")]
        memory: String,
        #[arg(long, help = "FLUX.1: длина T5 (dev 512, schnell 256)")]
        t5_len: Option<usize>,
        #[arg(long, default_value_t = false, help = "Qwen-Image 2.1: не кэшировать K/V текста и референсов")]
        no_kv_cache: bool,
    },
    /// Карта глубины Depth Anything V2: картинка → PNG (ближе — светлее).
    Depth {
        /// Каталог с model.safetensors Depth Anything V2.
        model: PathBuf,
        image: PathBuf,
        #[arg(short, long, default_value = "depth.png")]
        output: PathBuf,
        #[arg(long, default_value = "cuda")]
        device: String,
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
        /// Без аудио в выходе (text→video — VideoDit вместо AvDit; в режимах с условием звук считается, но не пишется).
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
        /// Two-stage: сила LoRA на stage1 (офиц. HQ 0.25; distilled — 0). Дефолт: 0, для IC-LoRA/lipdub — --lora-strength.
        #[arg(long)]
        lora_strength_stage1: Option<f32>,
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
        /// >0 = keyframe (append). На обеих стадиях при --two-stage.
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
        /// IC-LoRA: downscale reference относительно target (из метаданных LoRA; ref0.5-адаптеры — 2).
        #[arg(long, default_value_t = 2)]
        ref_downscale: usize,
        /// IC-LoRA: сила reference-conditioning (1.0 = reference clean).
        #[arg(long, default_value_t = 1.0)]
        ref_strength: f32,
        /// Аудио: с --ref-video — речь для lipdub; без него — audio→video (видео под готовый звук).
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
        #[arg(long, default_value_t = 0.3)]
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
        #[arg(long, help = "Seed шума (0 или без флага — случайный)")]
        seed: Option<u64>,
        #[arg(long, default_value_t = 0.7, help = "Guided: rescale CFG (0 — выкл)")]
        cfg_rescale: f32,
        #[arg(long, value_delimiter = ',', default_value = "29", help = "Guided: блоки DiT под STG-возмущение (через запятую)")]
        stg_blocks: Vec<usize>,
        #[arg(long, default_value_t = 0, help = "Guided: пропускать guidance каждые N шагов (0 — не пропускать)")]
        guider_skip_step: usize,
        #[arg(long, help = "Сохранить превью control-сигнала (canny/depth, кадр 0) в PNG")]
        control_preview: Option<PathBuf>,
        #[arg(long, help = "CRF libx264 (меньше — качественнее, дефолт ffmpeg 23)")]
        crf: Option<u32>,
    },
    /// MiniMax-H3 33B: текст/кадры → видео + синхронное стерео 32 кГц.
    H3 {
        /// Модель: .syn-бандл, корень MiniMax-H3 или сразу каталог FL2VA/Ref2VA.
        #[arg(long, alias = "model")]
        model_dir: PathBuf,
        /// Текстовый промпт.
        #[arg(default_value = "")]
        prompt: String,
        /// Негативный промпт; CFG (cfg > 1) работает только с ним.
        #[arg(long)]
        negative_prompt: Option<String>,
        #[arg(short, long, default_value = "h3.mp4")]
        output: PathBuf,
        /// Энкодер Qwen3-VL: .syn или каталог (по умолчанию — из модели).
        #[arg(long)]
        encoder: Option<PathBuf>,
        /// Первый кадр (fl2va).
        #[arg(long)]
        first_frame: Option<PathBuf>,
        /// Последний кадр (fl2va).
        #[arg(long)]
        last_frame: Option<PathBuf>,
        /// Референс (ref2va): картинка, видео или аудио — тип по расширению.
        /// Повторяется до 12 раз; порядок значим — он задаёт номера
        /// <Picture i> / <Video k> / <Audio j>, на которые ссылается промпт.
        #[arg(long = "ref")]
        refs: Vec<PathBuf>,
        /// Не брать звуковую дорожку видео-референсов.
        #[arg(long)]
        ref_mute_video: bool,
        /// Размер картинок-референсов: match — до площади кадра, max — 2048
        /// по короткой стороне, как у выпущенной модели (в разы медленнее).
        #[arg(long, default_value = "match")]
        ref_image_size: String,
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
        /// Seed (без флага — случайный, печатается).
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
        /// Квантование энкодера: none|mxfp8|nvfp4 (дефолт nvfp4).
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
        #[arg(long, default_value = "stretch", help = "Подгонка ключевых кадров под холст: stretch | crop")]
        keyframe_fit: String,
        #[arg(long, default_value = "res-multistep", help = "Сэмплер: res-multistep | euler")]
        sampler: String,
        #[arg(long, default_value_t = 0.0, help = "Rescale CFG (0 — выкл)")]
        cfg_rescale: f32,
        #[arg(long, default_value_t = 0, help = "Пропускать CFG каждые N шагов (0 — не пропускать)")]
        guider_skip_steps: usize,
        #[arg(long, help = "Сдвиг сигм видео (по умолчанию из конфига модели)")]
        sigma_shift_video: Option<f64>,
        #[arg(long, help = "Сдвиг сигм звука (по умолчанию из конфига модели)")]
        sigma_shift_audio: Option<f64>,
        #[arg(long, help = "Сторона тайла VAE-декода в пикселях (по умолчанию авто)")]
        vae_tile: Option<usize>,
        #[arg(long, value_delimiter = ',', help = "Номера --ref (с 1), у которых не брать звук видео")]
        ref_mute: Vec<usize>,
        #[arg(long, help = "av-restyle: исходное видео со звуком для частичного денойза")]
        restyle: Option<PathBuf>,
        #[arg(long, default_value_t = 0.6, help = "av-restyle: доля перегенерации 0..1")]
        restyle_strength: f32,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res: Result<(), Box<dyn std::error::Error>> = match cli.command {
        Commands::Devices => {
            device::list();
            Ok(())
        }
        Commands::Inspect { file, verbose, filter } => {
            inspect::run(inspect::InspectArgs { file, verbose, filter })
        }
        Commands::Convert {
            input, output, format, arch, component, mmproj, dtype, tokenizer, id, sha256, blake3,
        } => convert::run(convert::ConvertArgs {
            input, output, format, arch, component, mmproj, dtype, tokenizer, id, sha256, blake3,
        }),
        Commands::Bench {
            model, n_tokens, prompt_tokens, batch_size, warmup, device, attn, dtype, repeat, quant,
            storage_dtype, lm_head_dtype, embed_dtype, kv_dtype, no_graph, no_spec, prefill_batch,
        } => bench::run(bench::BenchArgs {
            model, n_tokens, prompt_tokens, batch_size, warmup, repeat, device, attn, quant,
            compute_dtype: dtype, storage_dtype, lm_head_dtype, embed_dtype, kv_dtype, no_graph, no_spec,
            prefill_batch,
        }),
        Commands::Run {
            model, prompt, prompt_file, max_tokens, temperature, seed, device, max_seq, attn, kv_dtype,
            quant, compute_dtype, storage_dtype, lm_head_dtype, embed_dtype, no_graph,
            warmup, mtp, no_mtp, no_graph_mtp, image, video, no_dflash, no_lookup, preset, top_k, top_p,
            min_p, repetition_penalty, repeat_last_n, presence_penalty, frequency_penalty, prefill_batch,
            chat, system, no_think, reasoning_effort, stop, max_image_tokens, layer_sync, expert_cache_gb,
            host_mirror_gb, qwen4_spec, embed_host,
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
                temperature: temperature.unwrap_or(1.0),
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
                no_lookup,
                sampling: synaptix_cli::commands::llm_facade::SamplingFlags {
                    preset,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty,
                    repeat_last_n,
                    presence_penalty,
                    frequency_penalty,
                },
                prefill_batch,
                chat,
                system,
                no_think,
                reasoning_effort,
                stop,
                max_image_tokens,
                layer_sync,
                expert_cache_gb,
                host_mirror_gb,
                qwen4_spec,
                embed_host,
            })
        }
        Commands::Chat {
            model, system, max_tokens, context, prefill_batch, preset, temperature, top_k, top_p, min_p,
            repetition_penalty, repeat_last_n, presence_penalty, frequency_penalty, seed, device, attn,
            quant, kv_dtype, compute_dtype, storage_dtype, lm_head_dtype, embed_dtype, no_think,
            reasoning_effort, stop, image, video, max_image_tokens, no_graph, no_spec, layer_sync,
        } => chat::run(chat::ChatArgs {
            model,
            system,
            max_tokens,
            context,
            prefill_batch,
            sampling: synaptix_cli::commands::llm_facade::SamplingFlags {
                preset,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                repeat_last_n,
                presence_penalty,
                frequency_penalty,
            },
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
            reasoning_effort,
            stop,
            image,
            video,
            max_image_tokens,
            no_graph,
            no_spec,
            layer_sync,
        }),
        Commands::Diff { file_a, file_b, atol, rtol } => {
            diff::run(diff::DiffArgs { file_a, file_b, atol, rtol })
        }
        Commands::Train { model, data, output, lora_r, lora_alpha, lr, epochs, batch_size } => {
            train::run(train::TrainArgs {
                model, data, output, lora_r, lora_alpha, lr, epochs, batch_size,
            })
        }
        Commands::Quantize { input, output, format, attn, lm_head, embed, keep_quant, device } => {
            quantize::run(quantize::QuantizeArgs { input, output, format, attn, lm_head, embed, keep_quant, device })
        }
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
        Commands::Podcast {
            bundle, script, script_file, output, voices, device, compute_dtype, cfg, steps, seed,
            max_length_times, zero_noise,
        } => podcast::run(podcast::PodcastArgs {
            bundle, script, script_file, output, voices, device, compute_dtype, cfg, steps, seed,
            max_length_times, zero_noise,
        }),
        Commands::Song {
            style, output, lyrics, lyrics_file, models, model, vae, cot, abc_file, cover,
            cover_voices, cover_full, cover_max_seconds, save_abc, seed, cfg, steps, max_tokens,
            seconds, min_tokens, temperature, top_p, top_k, repetition_penalty, penalty_window,
            abc_max_tokens, abc_min_tokens, abc_temperature, abc_top_p, abc_top_k,
            abc_repetition_penalty, abc_penalty_window, context, device, compute_dtype, quant,
            vae_dtype, vae_core_frames, vae_halo_frames, save_latent,
        } => song::run(song::SongArgs {
            style, output, lyrics, lyrics_file, models, model, vae, cot, abc_file, cover,
            cover_voices, cover_full, cover_max_seconds, save_abc, seed, cfg, steps, max_tokens,
            seconds, min_tokens, temperature, top_p, top_k, repetition_penalty, penalty_window,
            abc_max_tokens, abc_min_tokens, abc_temperature, abc_top_p, abc_top_k,
            abc_repetition_penalty, abc_penalty_window, context, device, compute_dtype, quant,
            vae_dtype, vae_core_frames, vae_halo_frames, save_latent,
        }),
        Commands::SongDecode {
            latent, output, models, vae, device, vae_dtype, vae_core_frames, vae_halo_frames,
        } => song::run_decode(song::SongDecodeArgs {
            latent, output, models, vae, device, vae_dtype, vae_core_frames, vae_halo_frames,
        }),
        Commands::Sheet {
            audio, output, models, model, full, voices, max_seconds, overlap_seconds,
            lookahead_seconds, device, compute_dtype, tokens_json,
        } => sheet::run(sheet::SheetArgs {
            audio, output, models, model, full, voices, max_seconds, overlap_seconds,
            lookahead_seconds, device, compute_dtype, tokens_json,
        }),
        Commands::SheetPack { sheetsage, mert, into } => {
            sheet::run_pack(sheet::SheetPackArgs { sheetsage, mert, into })
        }
        Commands::Music {
            caption, output, lyrics, models, lm, text_encoder, dit, vae, duration, steps, cfg,
            shift, seed, temperature, top_p, top_k, min_p, lm_cfg, use_cot, device, compute_dtype,
            quant, quant_encoder, retake_variance, retake_seed, mode, track, src_audio, repaint_start,
            repaint_end, repaint_strength, edit_n_min, edit_n_max, edit_n_avg, edit_source_caption,
            edit_source_lyric, no_ar, bpm, keyscale, timesig, norm, repeat, dcw, dcw_mode,
            dcw_preset, dcw_scaler, dcw_high_scaler, channels, save_latent,
        } => music::run(music::MusicArgs {
            caption, output, lyrics, models, lm, text_encoder, dit, vae, duration, steps, cfg,
            shift, seed, temperature, top_p, top_k, min_p, lm_cfg, use_cot, device, compute_dtype,
            quant, quant_encoder, retake_variance, retake_seed, mode, track, src_audio, repaint_start,
            repaint_end, repaint_strength, edit_n_min, edit_n_max, edit_n_avg, edit_source_caption,
            edit_source_lyric, use_ar: !no_ar, bpm, keyscale, timesig, norm, repeat, dcw, dcw_mode,
            dcw_preset, dcw_scaler, dcw_high_scaler, channels, save_latent,
        }),
        Commands::Imagine {
            model, prompt, output, negative, steps, cfg, height, width, seed, device, compute_dtype,
            quant, storage_dtype, image, resolution, init_image, strength, fit, memory, t5_len,
            no_kv_cache,
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
            image,
            resolution,
            init_image,
            strength,
            fit,
            memory,
            t5_len,
            no_kv_cache,
        }),
        Commands::Depth { model, image, output, device } => imagine::run_depth(&model, &image, &output, &device),
        Commands::Video {
            model, prompt, output, gemma, frames, duration, width, height, fps, no_audio,
            pipeline, list_pipelines, two_stage, upscaler, no_refine, lora, lora_strength,
            lora_strength_stage1, lora_strength_stage2,
            negative_prompt, cfg_scale, stg_scale, steps, image, image_strength, image_frame,
            video, retake_start, retake_end, ref_video, ref_downscale, ref_strength, audio,
            ref_preprocess, canny_low, canny_high, depth_model,
            quant_transformer, quant_encoder, compute_dtype, device,
            nag_prompt, nag_scale, nag_alpha, nag_tau, force_offload, prof, block_mode,
            seed, cfg_rescale, stg_blocks, guider_skip_step, control_preview, crf,
        } => video::run(video::VideoArgs {
            model, prompt, output, gemma, frames, duration, width, height, fps, no_audio,
            pipeline, list_pipelines, two_stage, upscaler, no_refine, lora, lora_strength,
            lora_strength_stage1, lora_strength_stage2,
            negative_prompt, cfg_scale, stg_scale, steps, image, image_strength, image_frame,
            video, retake_start, retake_end, ref_video, ref_downscale, ref_strength, audio,
            ref_preprocess, canny_low, canny_high, depth_model,
            quant_transformer, quant_encoder, compute_dtype, device,
            nag_prompt, nag_scale, nag_alpha, nag_tau, force_offload, prof, block_mode,
            seed, cfg_rescale, stg_blocks, guider_skip_step, control_preview, crf,
        }),
        Commands::H3 {
            model_dir, prompt, negative_prompt, output, encoder, first_frame, last_frame,
            refs, ref_mute_video, ref_image_size, width, height, duration, frames, steps, cfg_scale, seed, lora, lora_strength,
            quant_transformer, quant_encoder, compute_dtype, memory_mode, pipeline,
            list_pipelines, variant, device, prof, keep_wav, keyframe_fit, sampler, cfg_rescale,
            guider_skip_steps, sigma_shift_video, sigma_shift_audio, vae_tile, ref_mute, restyle,
            restyle_strength,
        } => h3::run(h3::H3Args {
            model_dir, prompt, negative_prompt, output, encoder, first_frame, last_frame,
            refs, ref_mute_video, ref_image_size, width, height, duration, frames, steps, cfg_scale, seed, lora, lora_strength,
            quant_transformer, quant_encoder, compute_dtype, memory_mode, pipeline,
            list_pipelines, variant, device, prof, keep_wav, keyframe_fit, sampler, cfg_rescale,
            guider_skip_steps, sigma_shift_video, sigma_shift_audio, vae_tile, ref_mute, restyle,
            restyle_strength,
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
