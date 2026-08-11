use crate::config::H3Variant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conditioning {
    None,
    FirstFrame,
    LastFrame,
    FirstLastFrame,
    References,
    VideoAudio,
}

#[derive(Debug, Clone)]
pub struct PipelineSpec {
    pub name: &'static str,
    pub desc: &'static str,
    pub variant: H3Variant,
    pub conditioning: Conditioning,
    pub default_steps: usize,
    pub default_cfg: f32,
    pub needs_lora: bool,
    pub implemented: bool,
}

pub fn registry() -> Vec<PipelineSpec> {
    vec![
        PipelineSpec {
            name: "t2va",
            desc: "текст → видео + синхронное стерео",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::None,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "t2va-turbo",
            desc: "текст → видео + звук, Turbo LoRA, 6 шагов без CFG",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::None,
            default_steps: 6,
            default_cfg: 1.0,
            needs_lora: true,
            implemented: true,
        },
        PipelineSpec {
            name: "fl2va-first",
            desc: "первый кадр → видео + звук",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::FirstFrame,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "fl2va-last",
            desc: "последний кадр → видео + звук",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::LastFrame,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "fl2va-both",
            desc: "переход между первым и последним кадром",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::FirstLastFrame,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "ref2va",
            desc: "референсы (до 9 изображений, 3 видео, 3 аудио) → видео + звук",
            variant: H3Variant::Ref2va,
            conditioning: Conditioning::References,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "av-restyle",
            desc: "частичный денойзинг существующего видео со звуком",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::VideoAudio,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
    ]
}

pub fn by_name(name: &str) -> Option<PipelineSpec> {
    registry().into_iter().find(|s| s.name == name)
}

pub fn names() -> Vec<&'static str> {
    registry().iter().map(|s| s.name).collect()
}

pub fn for_conditioning(variant: H3Variant, cond: Conditioning) -> Option<PipelineSpec> {
    registry()
        .into_iter()
        .find(|s| s.variant == variant && s.conditioning == cond)
}
