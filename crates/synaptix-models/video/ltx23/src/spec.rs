//! `PipelineSpec` — декларативное описание одного из официальных LTX-2.3
//! пайплайнов (ground-truth `Temp/LTX-2-official/packages/ltx-pipelines`). Единый
//! движок: CLI выбирает пресет по имени (`--pipeline <name>`), spec задаёт число
//! стадий, модальность, сигмы, требуемые conditioning/LoRA/upscaler. Маршрутизация
//! на `pipeline::generate_*`. Неполные пресеты помечены `Stage` (Фаза, в которой
//! путь реализуется) — CLI выдаёт понятную ошибку до её завершения.

use crate::pipeline::{DISTILLED_SIGMAS, STAGE2_SIGMAS};

/// Число диффузионных стадий пайплайна.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stages {
    /// Одна денойз-петля на целевом разрешении.
    One,
    /// stage1 (полразрешения) → spatial-upscaler ×2 → stage2-refine.
    Two,
}

/// Какие потоки денойзятся совместно.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Modality {
    /// Только видео-латент.
    Video,
    /// Совместный видео+аудио (joint cross-attn в `AvDit`).
    AudioVideo,
}

/// Вид входного conditioning (Фаза 4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Conditioning {
    /// txt→video (без conditioning-латентов).
    None,
    /// image→video (replace-latent первого кадра).
    Image,
    /// video→video (IC-LoRA reference).
    Video,
    /// интерполяция между keyframe'ами.
    Keyframe,
    /// audio→video.
    AudioInput,
    /// retake: временна́я маска региона.
    RetakeMask,
}

/// Декларативная спецификация пайплайна.
#[derive(Clone, Debug)]
pub struct PipelineSpec {
    /// Имя для `--pipeline`.
    pub name: &'static str,
    /// Краткое описание (для `--list-pipelines`).
    pub desc: &'static str,
    pub stages: Stages,
    pub modality: Modality,
    /// Two-stage: запускать stage2-refine (false → только upscale+decode).
    pub refine: bool,
    pub conditioning: Conditioning,
    /// Требует distilled/ic-LoRA для корректного результата.
    pub needs_lora: bool,
    /// Требует чекпойнт spatial-upscaler ×2.
    pub needs_upscaler: bool,
    /// Сигмы stage1 (flow-match Euler).
    pub stage1_sigmas: Vec<f64>,
    /// Сигмы stage2 (re-noise+refine; пусто для one-stage).
    pub stage2_sigmas: Vec<f64>,
    /// `None` → путь реализован; `Some(phase)` → ещё нет (CLI выдаёт ошибку).
    pub todo_phase: Option<u8>,
}

impl PipelineSpec {
    pub fn implemented(&self) -> bool {
        self.todo_phase.is_none()
    }
    pub fn needs_upscaler(&self) -> bool {
        self.needs_upscaler
    }
}

/// Все известные пайплайны (имена соответствуют официальным `ltx_pipelines/*`).
pub fn registry() -> Vec<PipelineSpec> {
    let s1 = DISTILLED_SIGMAS.to_vec();
    let s2 = STAGE2_SIGMAS.to_vec();
    vec![
        PipelineSpec {
            name: "one-stage",
            desc: "txt→video, одна стадия на целевом разрешении (быстро, без upscaler)",
            stages: Stages::One,
            modality: Modality::Video,
            refine: false,
            conditioning: Conditioning::None,
            needs_lora: false,
            needs_upscaler: false,
            stage1_sigmas: s1.clone(),
            stage2_sigmas: Vec::new(),
            todo_phase: None,
        },
        PipelineSpec {
            name: "two-stage",
            desc: "txt→video+audio, distilled 2-stage: stage1 A/V → upscaler ×2 → stage2-refine A/V",
            stages: Stages::Two,
            modality: Modality::AudioVideo,
            refine: true,
            conditioning: Conditioning::None,
            needs_lora: false,
            needs_upscaler: true,
            stage1_sigmas: s1.clone(),
            stage2_sigmas: s2.clone(),
            todo_phase: None,
        },
        PipelineSpec {
            name: "av",
            desc: "txt→video+audio, одна стадия (совместный denoise + вокодер)",
            stages: Stages::One,
            modality: Modality::AudioVideo,
            refine: false,
            conditioning: Conditioning::None,
            needs_lora: false,
            needs_upscaler: false,
            stage1_sigmas: s1.clone(),
            stage2_sigmas: Vec::new(),
            todo_phase: None,
        },
        PipelineSpec {
            name: "ti2v-two-stage",
            desc: "txt/image→video+audio, 2-stage с multimodal guidance (Фаза 3)",
            stages: Stages::Two,
            modality: Modality::AudioVideo,
            refine: true,
            conditioning: Conditioning::Image,
            needs_lora: false,
            needs_upscaler: true,
            stage1_sigmas: s1.clone(),
            stage2_sigmas: s2.clone(),
            todo_phase: Some(3),
        },
        PipelineSpec {
            name: "a2v",
            desc: "audio→video, 2-stage (Фаза 4: audio conditioning)",
            stages: Stages::Two,
            modality: Modality::AudioVideo,
            refine: true,
            conditioning: Conditioning::AudioInput,
            needs_lora: false,
            needs_upscaler: true,
            stage1_sigmas: s1.clone(),
            stage2_sigmas: s2.clone(),
            todo_phase: Some(4),
        },
        PipelineSpec {
            name: "keyframe",
            desc: "интерполяция между keyframe'ами (Фаза 4: keyframe conditioning)",
            stages: Stages::Two,
            modality: Modality::AudioVideo,
            refine: true,
            conditioning: Conditioning::Keyframe,
            needs_lora: false,
            needs_upscaler: true,
            stage1_sigmas: s1.clone(),
            stage2_sigmas: s2.clone(),
            todo_phase: Some(4),
        },
        PipelineSpec {
            name: "ic-lora",
            desc: "video→video через IC-LoRA reference (Фаза 5)",
            stages: Stages::Two,
            modality: Modality::Video,
            refine: true,
            conditioning: Conditioning::Video,
            needs_lora: true,
            needs_upscaler: true,
            stage1_sigmas: s1.clone(),
            stage2_sigmas: s2.clone(),
            todo_phase: Some(5),
        },
        PipelineSpec {
            name: "hdr-ic-lora",
            desc: "HDR через IC-LoRA (Фаза 5)",
            stages: Stages::Two,
            modality: Modality::Video,
            refine: true,
            conditioning: Conditioning::Video,
            needs_lora: true,
            needs_upscaler: true,
            stage1_sigmas: s1.clone(),
            stage2_sigmas: s2.clone(),
            todo_phase: Some(5),
        },
        PipelineSpec {
            name: "retake",
            desc: "перегенерация временно́го региона по маске (Фаза 5)",
            stages: Stages::Two,
            modality: Modality::AudioVideo,
            refine: true,
            conditioning: Conditioning::RetakeMask,
            needs_lora: false,
            needs_upscaler: true,
            stage1_sigmas: s1.clone(),
            stage2_sigmas: s2.clone(),
            todo_phase: Some(5),
        },
        PipelineSpec {
            name: "lipdub",
            desc: "переозвучка губ через IC-LoRA reference (Фаза 5)",
            stages: Stages::Two,
            modality: Modality::AudioVideo,
            refine: true,
            conditioning: Conditioning::Video,
            needs_lora: true,
            needs_upscaler: true,
            stage1_sigmas: s1,
            stage2_sigmas: s2,
            todo_phase: Some(5),
        },
    ]
}

/// Найти пайплайн по имени.
pub fn by_name(name: &str) -> Option<PipelineSpec> {
    registry().into_iter().find(|p| p.name == name)
}

/// Имена всех пайплайнов (для подсказки в ошибке/`--help`).
pub fn names() -> Vec<&'static str> {
    registry().iter().map(|p| p.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_invariants() {
        let r = registry();
        // имена уникальны
        let mut seen = std::collections::HashSet::new();
        for p in &r {
            assert!(seen.insert(p.name), "дубликат имени {}", p.name);
        }
        // one-stage не нуждается в upscaler и не имеет stage2-сигм; two-stage наоборот
        for p in &r {
            if p.stages == Stages::One {
                assert!(!p.needs_upscaler, "{}: one-stage не требует upscaler", p.name);
                assert!(p.stage2_sigmas.is_empty(), "{}: one-stage без stage2-сигм", p.name);
            } else {
                assert!(p.needs_upscaler, "{}: two-stage требует upscaler", p.name);
                assert!(!p.stage2_sigmas.is_empty(), "{}: two-stage со stage2-сигмами", p.name);
            }
            // todo_phase: реализованные → None
            assert_eq!(p.implemented(), p.todo_phase.is_none());
        }
        // три базовых готовы
        for n in ["one-stage", "two-stage", "av"] {
            assert!(by_name(n).unwrap().implemented(), "{n} должен быть готов");
        }
        assert!(by_name("nope").is_none());
    }
}
