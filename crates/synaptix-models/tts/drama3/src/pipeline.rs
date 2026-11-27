//! Drama 3 целиком: образец → коды, промпт, основа + башня покадрово, кодек.

use std::path::Path;

use synaptix_core::precision::PrecisionConfig;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_llm_qwen3_next_hybrid::loader::HybridWeights;
use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;
use synaptix_tts_audio8::codec::{Codec, CodecShape, CodecSource};
use synaptix_tts_audio8::Audio8Error;

use crate::prompt::{build_rows, RefCodes};

use crate::sample::{sample, sample_main, Rng, CODEBOOK_TOP_K};
use crate::tower::{Src, Tower, TowerCache, DIM};
use crate::{is_semantic, Drama3Error, Row, CODEBOOKS, CODEBOOK_SIZE, EOS, FRAME_SAMPLES, R, RESIDUAL_SIZE, SEMANTIC_BEGIN, SEMANTIC_END};

/// Предел строк промпта и кадров ответа (как у эталона: контекст 16384,
/// промпт ≤ 8192, ответ по умолчанию ≤ 2048 кадров ≈ 95 с).
pub const MAX_PROMPT_ROWS: usize = 8192;
pub const MAX_FRAMES: usize = 2048;
/// Предел образца голоса, секунд.
pub const MAX_REFERENCE_SECONDS: usize = 60;

fn me(e: impl std::fmt::Display) -> Drama3Error {
    Drama3Error::Other(e.to_string())
}

fn host(t: &Tensor) -> R<Vec<f32>> {
    Ok(t.contiguous()?.to_dtype(DType::F32)?.to_device(Device::Cpu)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

struct CodecW {
    loader: SynBundleLoader,
    device: Device,
    dtype: DType,
}

impl CodecSource for CodecW {
    fn codec(&self, name: &str) -> synaptix_tts_audio8::Result<Tensor> {
        self.loader.load_to(name, self.device, self.dtype).map_err(|e| Audio8Error::Load(format!("codec {name}: {e}")))
    }
    fn codec_f32(&self, name: &str) -> synaptix_tts_audio8::Result<Tensor> {
        self.loader.load_to(name, self.device, DType::F32).map_err(|e| Audio8Error::Load(format!("codec {name}: {e}")))
    }
    fn codec_has(&self, name: &str) -> bool {
        WeightLoader::contains(&self.loader, name)
    }
    fn codec_dtype(&self) -> DType {
        self.dtype
    }
    fn device(&self) -> Device {
        self.device
    }
}

/// Образец голоса: расшифровка и коды кодека `[10][T]` ([`Drama3::encode_reference`]).
#[derive(Debug, Clone)]
pub struct Reference {
    pub transcript: String,
    pub codes: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct Drama3Request {
    /// Текст; говорящие — `<|speaker:N|>` (без тега — говорящий 0), события —
    /// в квадратных скобках (`[laughs]`).
    pub text: String,
    pub references: Vec<Reference>,
    pub temperature: f32,
    pub top_p: f32,
    /// top-k головного токена (0 — без ограничения, как в эталоне); у
    /// акустических голов — не больше 30.
    pub top_k: usize,
    pub max_frames: usize,
    pub seed: u64,
}

impl Default for Drama3Request {
    fn default() -> Self {
        Self { text: String::new(), references: Vec::new(), temperature: 1.0, top_p: 0.9, top_k: 0, max_frames: MAX_FRAMES, seed: 0 }
    }
}

pub struct Drama3 {
    pub lm: HybridPipeline,
    pub tower: Tower,
    codec: Codec,
    codec_src: CodecW,
    _zc: Option<std::sync::Arc<synaptix_llm_common::moe_zc::ZcPool>>,
    /// Кандидаты головного токена: eos и полоса семантики.
    head_ids: Vec<u32>,
    device: Device,
}

impl Drama3 {
    pub fn load(bundle: &Path, device: Device) -> R<Self> {
        let compute = if device.is_cuda() { DType::BF16 } else { DType::F32 };
        let cap = MAX_PROMPT_ROWS + MAX_FRAMES + 16;
        // Основа считает в BF16, как оригинал. F16 (`D3_LM_F16=1`, под него написан
        // быстрый device-шаг гибрида) портит речь: состояния префилла расходятся
        // с BF16 (cos 0,997, отдельные строки 0,94), речь обрывается через
        // несколько кадров — замер 27.11.2026.
        let lm_dtype = if device.is_cuda() && std::env::var_os("D3_LM_F16").is_some() { DType::F16 } else { compute };
        let mut lm = HybridPipeline::load_with_experts_in_ram(bundle, device, PrecisionConfig::dense(lm_dtype), Some(cap)).map_err(|e| Drama3Error::Load(format!("основа: {e}")))?;
        if lm.config.hidden_size != DIM {
            return Err(Drama3Error::Load(format!("hidden_size основы {} ≠ {DIM}", lm.config.hidden_size)));
        }
        let w = HybridWeights::load(bundle, device, compute).map_err(|e| Drama3Error::Load(e.to_string()))?;
        let tower = Tower::load(&Src { w: &w, device, dtype: compute }, cap)?;
        let codec_src = CodecW {
            loader: SynBundleLoader::open(bundle).map_err(|e| Drama3Error::Load(e.to_string()))?.with_component("codec").with_device(device),
            device,
            dtype: DType::F32,
        };
        let shape = CodecShape { num_codebooks: CODEBOOKS, post_layers: 8, post_heads: 16, post_local_heads: 16 };
        let codec = Codec::load(&codec_src, shape, false).map_err(|e| Drama3Error::Load(format!("кодек: {e}")))?;
        // Эксперты основы — в RAM; кэш экспертов на карте по умолчанию мал,
        // и каждый шаг тянул бы их по шине. Башня и кодек уже загружены —
        // остаток видеопамяти (минус запас под активации префилла) отдаём кэшу.
        // Декод MoE без синхронизации с хостом (zero-copy): выбор экспертов и
        // счёт — на карте, эксперт читается из слота VRAM или прямо из
        // закреплённой RAM. Результат побайтно тот же, шаг основы 41,6 → 18,8 мс
        // (96 слотов). Слотов на слой — по свободной видеопамяти (башня и кодек
        // уже загружены): 16…96, запас 3 ГБ под префилл, KV и кодек;
        // `D3_ZC=N` — задать, `D3_ZC=0` — выключить.
        let zc_slots = std::env::var("D3_ZC").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or_else(|| {
            let free = match device {
                Device::Cuda(ord) => synaptix_core::device::cuda::mem_info(ord).map(|(f, _)| f as usize).unwrap_or(0),
                _ => 0,
            };
            let c = &lm.config;
            // gate_up [2I, H] + down [H, I] в MXFP8: байт на вес + масштабы блоков по 32.
            let per_expert = 3 * c.hidden_size * 512 * 33 / 32;
            (free.saturating_sub(3usize << 30) / c.num_hidden_layers.max(1) / per_expert.max(1)).clamp(16, 96)
        });
        let zc = if zc_slots > 0 { lm.model.attach_zc(zc_slots) } else { None };
        if std::env::var_os("D3_PROF").is_some() {
            eprintln!("[d3] zero-copy MoE: {}", if zc.is_some() { format!("{zc_slots} слотов на слой") } else { "нет".into() });
        }
        let mut head_ids = vec![EOS];
        head_ids.extend(SEMANTIC_BEGIN..=SEMANTIC_END);
        Ok(Self { lm, tower, codec, codec_src, _zc: zc, head_ids, device })
    }

    /// Образец голоса (моно, 44,1 кГц) → коды `[10][T]`.
    pub fn encode_reference(&self, pcm: &[f32]) -> R<Vec<Vec<u32>>> {
        if pcm.len() > MAX_REFERENCE_SECONDS * crate::SAMPLE_RATE as usize {
            return Err(Drama3Error::Input(format!("образец длиннее {MAX_REFERENCE_SECONDS} с")));
        }
        self.codec.ensure_encoder(&self.codec_src).map_err(me)?;
        self.codec.encode(pcm).map_err(me)
    }

    /// Коды `[10][T]` → волна 44,1 кГц (остаточные коды клампятся до размера кодбука кодека).
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> R<Vec<f32>> {
        let clamped: Vec<Vec<u32>> = codes.iter().enumerate().map(|(k, c)| if k == 0 { c.clone() } else { c.iter().map(|&v| v.min(RESIDUAL_SIZE - 1)).collect() }).collect();
        self.codec.decode(&clamped).map_err(me)
    }

    pub fn prompt_rows(&self, text: &str, refs: &[Reference]) -> R<Vec<Row>> {
        let rc: Vec<RefCodes> = refs.iter().map(|r| RefCodes { transcript: &r.transcript, codes: &r.codes }).collect();
        let rows = build_rows(text, &rc, &|s| self.lm.encode(s).map_err(me))?;
        if rows.len() > MAX_PROMPT_ROWS {
            return Err(Drama3Error::Input(format!("промпт {} строк — больше предела {MAX_PROMPT_ROWS}", rows.len())));
        }
        Ok(rows)
    }

    /// Вход основы для строк: `[1, n, H]`.
    pub fn embed(&self, rows: &[Row]) -> R<Tensor> {
        let ids = Tensor::from_vec(rows.iter().map(|r| r[0]).collect::<Vec<u32>>(), vec![rows.len()], self.device)?;
        let text = self.lm.model.embed_rows(&ids).map_err(me)?;
        Ok(self.tower.backbone_input(rows, &text)?.to_dtype(self.lm.model.dtype)?.reshape(vec![1, rows.len(), DIM])?.contiguous()?)
    }

    /// Логиты кандидатов головного токена по состоянию после нормы `[1, H]`.
    pub fn head_logits(&self, normed: &Tensor) -> R<Vec<f32>> {
        let logits = self.lm.model.lm_head_forward(normed).map_err(me)?;
        let mut out = host(&logits.narrow(1, EOS as usize, 1)?)?;
        out.extend(host(&logits.narrow(1, SEMANTIC_BEGIN as usize, CODEBOOK_SIZE)?)?);
        Ok(out)
    }

    /// Префилл основы и башни по строкам промпта: состояние последней строки
    /// после нормы `[1, H]` и кэши.
    pub fn prefill(&self, rows: &[Row], max_frames: usize) -> R<(Tensor, synaptix_llm_common::KvCache, TowerCache)> {
        let l = rows.len();
        let mut kv = self.lm.model.make_kv_cache(1, l + max_frames + 8).map_err(me)?;
        let hid = self.lm.model.forward_from_hidden(&self.embed(rows)?, &mut kv).map_err(me)?;
        let normed = self.lm.model.normed_all(&hid).map_err(me)?.reshape(vec![l, DIM])?;
        let mut tc = self.tower.make_cache(l + max_frames + 8)?;
        if l > 1 {
            self.tower.prefill(&normed.narrow(0, 0, l - 1)?.contiguous()?, &rows[..l - 1], &rows[1..], 0, &mut tc, None)?;
        }
        Ok((normed.narrow(0, l - 1, 1)?.contiguous()?, kv, tc))
    }

    /// Кадры одного куска текста: `(кадры, дошло ли до конца речи)`.
    /// `history` — уже сказанные куски; `progress(кадр, максимум)`, `false` — остановить.
    pub fn generate_chunk(&self, text: &str, req: &Drama3Request, history: &[(&str, &[Row])], rng: &mut Rng, progress: &mut dyn FnMut(usize, usize) -> bool) -> R<(Vec<Row>, bool)> {
        let rc: Vec<RefCodes> = req.references.iter().map(|r| RefCodes { transcript: &r.transcript, codes: &r.codes }).collect();
        let rows = crate::prompt::build_rows_with_history(text, &rc, history, &|s| self.lm.encode(s).map_err(me))?;
        if rows.len() > MAX_PROMPT_ROWS {
            return Err(Drama3Error::Input(format!("промпт {} строк — больше предела {MAX_PROMPT_ROWS}", rows.len())));
        }
        let max_frames = req.max_frames.clamp(1, MAX_FRAMES);
        let prof = std::env::var_os("D3_PROF").is_some();
        let tp = std::time::Instant::now();
        // Префилл всегда идёт прежним путём MoE: со собранными слотами zero-copy
        // он пошёл бы через их групповой GEMM, а это другой счёт (второй кусок
        // реплики переставал совпадать с прогоном без zero-copy). Слоты
        // собираются заново на первом шаге декода.
        if self._zc.is_some() {
            self.lm.model.zc_release();
        }
        let (mut last, mut kv, mut tc) = self.prefill(&rows, max_frames)?;
        let (mut t_head, mut t_tower, mut t_lm) = (0f64, 0f64, 0f64);
        if prof {
            let _ = host(&last)?;
            eprintln!("[d3] префилл {} строк: {:.2} с", rows.len(), tp.elapsed().as_secs_f64());
        }
        let mut window: Vec<u32> = rows.iter().map(|r| r[0]).collect();
        let mut input = *rows.last().expect("промпт не пуст");
        let mut frames: Vec<Row> = Vec::new();
        let top_k_cb = if req.top_k == 0 { CODEBOOK_TOP_K } else { req.top_k.min(CODEBOOK_TOP_K) };
        let mut finished = false;
        // Быстрый device-шаг основы (слитые ядра декода); `D3_EAGER=1` — прежний
        // общий проход (для сравнения).
        let fast = self.lm.model.dtype == DType::F16 && self.lm.model.device_step_ready_hidden() && std::env::var_os("D3_EAGER").is_none();
        let mut state = if fast {
            self.lm.model.sync_decode_dev_state(&mut kv).map_err(me)?;
            Some(self.lm.model.make_decode_state().map_err(me)?)
        } else {
            None
        };
        if prof {
            eprintln!("[d3] шаг основы: {}", if fast { "device (слитые ядра)" } else { "общий проход" });
        }
        if std::env::var_os("D3_PROF_OPS").is_some() {
            synaptix_llm_common::model::set_decode_prof(true);
            let _ = synaptix_llm_common::profile::report();
        }
        for step in 0..max_frames {
            let pos = rows.len() - 1 + step;
            let t = std::time::Instant::now();
            let logits = self.head_logits(&last)?;
            t_head += t.elapsed().as_secs_f64();
            let main = sample_main(&logits, &self.head_ids, &window, req.temperature, req.top_k, req.top_p, rng);
            if main == EOS {
                finished = true;
                break;
            }
            let t = std::time::Instant::now();
            let codes = self.tower.step(&last, &input, main, pos, &mut tc, &mut |_, lg| sample(lg, req.temperature, top_k_cb, req.top_p, false, rng) as u32)?;
            t_tower += t.elapsed().as_secs_f64();
            let mut frame = [0u32; CODEBOOKS + 1];
            frame[0] = main;
            frame[1..].copy_from_slice(&codes);
            frames.push(frame);
            window.push(main);
            if !progress(frames.len(), max_frames) {
                return Err(Drama3Error::Cancelled);
            }
            let t = std::time::Instant::now();
            let emb = self.embed(std::slice::from_ref(&frame))?;
            last = match state.as_mut() {
                Some(st) => {
                    st.update(frame[0], (rows.len() + step) as u32).map_err(me)?;
                    self.lm.model.forward_decode_dev_hidden(st, &mut kv, &emb.reshape(vec![1, DIM])?).map_err(me)?.reshape(vec![1, DIM])?.contiguous()?
                }
                None => {
                    let hid = self.lm.model.forward_from_hidden(&emb, &mut kv).map_err(me)?;
                    self.lm.model.normed_at(&hid, 0).map_err(me)?.contiguous()?
                }
            };
            if prof {
                let _ = host(&last)?;
            }
            t_lm += t.elapsed().as_secs_f64();
            input = frame;
        }
        if prof {
            let (zh, zp, zs) = self.lm.model.zc_stats();
            eprintln!("[d3] zero-copy: в слотах {zh}, из RAM {zp}, слотов всего {zs}");
            if let Some(st) = self.lm.model.expert_cache_stats() {
                eprintln!("[d3] кэш экспертов: попаданий {}, промахов {}, на карте {} ({} МБ), батчем {} / поэкспертно {}", st.hits, st.misses, st.resident, st.bytes >> 20, st.batched, st.unbatched);
            }
        }
        if std::env::var_os("D3_PROF_OPS").is_some() {
            eprintln!("{}", synaptix_llm_common::model::decode_prof_report_and_clear());
            eprintln!("MoE:\n{}", synaptix_llm_common::profile::report());
            synaptix_llm_common::model::set_decode_prof(false);
        }
        if prof && !frames.is_empty() {
            let n = frames.len() as f64;
            eprintln!("[d3] на кадр: основа {:.1} мс, голова {:.1} мс, башня {:.1} мс ({} кадров)", t_lm / n * 1e3, t_head / n * 1e3, t_tower / n * 1e3, frames.len());
        }
        // Хвостовой `<|semantic:0|>` перед концом речи даёт щелчок — отбрасывается
        // (как удержание кадра в MLX-эталоне).
        if finished && frames.last().is_some_and(|f| f[0] == SEMANTIC_BEGIN) {
            frames.pop();
        }
        frames.retain(|f| is_semantic(f[0]));
        Ok((frames, finished))
    }

    /// Речь 44,1 кГц моно. Длинный текст идёт кусками ([`crate::text::split`]);
    /// каждый кусок видит первый и предыдущий как историю.
    /// `progress(кусок, кусков, кадр, максимум кадров)`; `false` — остановить.
    pub fn generate(&self, req: &Drama3Request, progress: &mut dyn FnMut(usize, usize, usize, usize) -> bool) -> R<Drama3Output> {
        crate::prompt::validate(&req.text, "текст")?;
        let chunks = crate::text::split(&req.text);
        let mut rng = Rng::new(req.seed);
        let t0 = std::time::Instant::now();
        let mut done: Vec<(String, Vec<Row>)> = Vec::new();
        let mut finished = true;
        for (ci, chunk) in chunks.iter().enumerate() {
            let hist: Vec<(&str, &[Row])> = match done.len() {
                0 => vec![],
                1 => vec![(done[0].0.as_str(), done[0].1.as_slice())],
                n => vec![(done[0].0.as_str(), done[0].1.as_slice()), (done[n - 1].0.as_str(), done[n - 1].1.as_slice())],
            };
            let (frames, fin) = self.generate_chunk(chunk, req, &hist, &mut rng, &mut |i, n| progress(ci, chunks.len(), i, n))?;
            finished &= fin;
            if !frames.is_empty() {
                done.push((chunk.clone(), frames));
            }
        }
        let total: usize = done.iter().map(|(_, f)| f.len()).sum();
        if total == 0 {
            return Err(Drama3Error::Input("модель не выдала ни одного кадра речи".into()));
        }
        let ar_seconds = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        // Слоты zero-copy держат до ~12 ГБ VRAM, а кодеку на длинную реплику
        // нужны гигабайты активаций — слоты отпускаются (соберутся на следующем
        // декоде). Кодек — по кускам текста, как в эталоне: память не растёт
        // с длиной текста.
        if self._zc.is_some() {
            self.lm.model.zc_release();
        }
        let mut pcm: Vec<f32> = Vec::with_capacity(total * FRAME_SAMPLES);
        for (_, frames) in &done {
            let codes: Vec<Vec<u32>> = (0..CODEBOOKS).map(|k| frames.iter().map(|f| f[1 + k]).collect()).collect();
            pcm.extend(self.decode_codes(&codes)?);
        }
        debug_assert_eq!(pcm.len(), total * FRAME_SAMPLES);
        let frames = vec![(); total];
        Ok(Drama3Output { pcm, frames: frames.len(), chunks: chunks.len(), finished, ar_seconds, codec_seconds: t1.elapsed().as_secs_f64() })
    }
}

/// Результат синтеза.
pub struct Drama3Output {
    /// 44,1 кГц моно.
    pub pcm: Vec<f32>,
    pub frames: usize,
    pub chunks: usize,
    /// Все куски дошли до конца речи (а не упёрлись в предел кадров).
    pub finished: bool,
    pub ar_seconds: f64,
    pub codec_seconds: f64,
}
