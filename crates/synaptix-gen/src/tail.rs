//! Латент хвоста ролика для бесшовного продолжения сцен (MiniMax-H3, LTX-2.3).
//!
//! Хранит последние `context_frames` латентных кадров видео и соответствующий
//! отрезок аудиолатента, взятые прямо из денойзера до декодирования VAE, плюс
//! метаданные совместимости (семейство, модель, размер кадра, fps, перекрытие).
//! Сериализуется в один файл `.safetensors` (метаданные — в `__metadata__`).

use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::{Dtype as StDtype, SafeTensors, TensorView};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Метаданные хвоста (читаются из заголовка `.safetensors` без загрузки тензоров).
#[derive(Debug, Clone, PartialEq)]
pub struct TailMeta {
    /// Семейство модели: `"h3"` или `"ltx"`.
    pub family: String,
    /// Идентификатор модели в службе (например `"minimax-h3-fl2va"`).
    pub model: String,
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    /// Число латентных кадров видео в хвосте.
    pub context_frames: usize,
    /// Число пиксельных кадров, соответствующих хвосту (перекрытие).
    pub overlap_frames: usize,
    /// Число аудио-латентных кадров в хвосте (0 — без аудио).
    pub audio_frames: usize,
}

impl TailMeta {
    pub fn label(&self) -> String {
        let fam = match self.family.as_str() {
            "h3" => "H3",
            "ltx" => "LTX",
            other => other,
        };
        let aud = if self.audio_frames > 0 { " + звук" } else { "" };
        format!(
            "хвост {fam} {}×{} @ {:.0} fps, {} лат. к. ({} к.){aud}",
            self.width, self.height, self.fps, self.context_frames, self.overlap_frames
        )
    }

    /// Проверить совместимость хвоста с целевой моделью и параметрами.
    /// Несовпадение — явная ошибка, а не тихий откат на картинку.
    pub fn check_compat(
        &self,
        expected_family: &str,
        requested_model: Option<&str>,
        requested_width: Option<usize>,
        requested_height: Option<usize>,
        requested_fps: Option<f64>,
    ) -> R<()> {
        if self.family != expected_family {
            return Err(format!(
                "хвост от модели семейства «{}» нельзя подать в «{}»",
                self.family, expected_family
            ));
        }
        if let Some(m) = requested_model.map(str::trim).filter(|s| !s.is_empty()) {
            if !self.model.is_empty() && self.model != m {
                return Err(format!(
                    "несовпадение модели: хвост от «{}», а запрошена «{m}»",
                    self.model
                ));
            }
        }
        if let Some(w) = requested_width.filter(|&v| v > 0) {
            if w != self.width {
                return Err(format!(
                    "несовпадение ширины: хвост {}×{}, а запрошена ширина {w}",
                    self.width, self.height
                ));
            }
        }
        if let Some(h) = requested_height.filter(|&v| v > 0) {
            if h != self.height {
                return Err(format!(
                    "несовпадение высоты: хвост {}×{}, а запрошена высота {h}",
                    self.width, self.height
                ));
            }
        }
        if let Some(f) = requested_fps.filter(|&v| v > 0.0) {
            if (f - self.fps).abs() > 0.01 {
                return Err(format!(
                    "несовпадение fps: хвост {:.2} fps, а запрошено {:.2} fps",
                    self.fps, f
                ));
            }
        }
        Ok(())
    }

    fn to_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("format".into(), "synaptix-video-tail-v1".into());
        m.insert("family".into(), self.family.clone());
        m.insert("model".into(), self.model.clone());
        m.insert("width".into(), self.width.to_string());
        m.insert("height".into(), self.height.to_string());
        m.insert("fps".into(), format!("{:.6}", self.fps));
        m.insert("context_frames".into(), self.context_frames.to_string());
        m.insert("overlap_frames".into(), self.overlap_frames.to_string());
        m.insert("audio_frames".into(), self.audio_frames.to_string());
        m
    }

    fn from_map(m: &HashMap<String, String>) -> R<Self> {
        let get = |k: &str| m.get(k).cloned().ok_or_else(|| format!("tail safetensors: нет поля «{k}»"));
        let num = |k: &str| get(k)?.parse::<usize>().map_err(|_| format!("tail safetensors: нечисловое «{k}»"));
        Ok(Self {
            family: get("family")?,
            model: m.get("model").cloned().unwrap_or_default(),
            width: num("width")?,
            height: num("height")?,
            fps: get("fps")?.parse::<f64>().map_err(|_| "tail safetensors: неверный fps".to_string())?,
            context_frames: num("context_frames")?,
            overlap_frames: num("overlap_frames")?,
            audio_frames: m.get("audio_frames").and_then(|v| v.parse().ok()).unwrap_or(0),
        })
    }
}

/// Латент хвоста ролика (CPU-тензоры F32, несколько мегабайт).
#[derive(Clone)]
pub struct TailLatent {
    pub meta: TailMeta,
    /// Видеолатент хвоста: H3 `[1, 24, T_ctx, H', W']`, LTX `[1, 128, T_ctx, H', W']`.
    pub video: Tensor,
    /// Аудиолатент хвоста: H3 `[1, 32, 2, A_ctx]`, LTX `[1, Fa_ctx, 128]`.
    pub audio: Option<Tensor>,
    /// Половинный видеолатент хвоста для LTX two-stage (Stage1).
    pub video_stage1: Option<Tensor>,
}

impl TailLatent {
    pub fn label(&self) -> String {
        self.meta.label()
    }

    /// Сохранить хвост в `.safetensors` (F32, метаданные в `__metadata__`).
    pub fn save(&self, path: &Path) -> R<()> {
        let v_bytes = tensor_to_f32_bytes(&self.video)?;
        let v_shape = self.video.dims().to_vec();
        let a_data = match &self.audio {
            Some(a) => Some((tensor_to_f32_bytes(a)?, a.dims().to_vec())),
            None => None,
        };
        let s1_data = match &self.video_stage1 {
            Some(s1) => Some((tensor_to_f32_bytes(s1)?, s1.dims().to_vec())),
            None => None,
        };
        let mut views: Vec<(&str, TensorView<'_>)> = Vec::with_capacity(3);
        views.push((
            "video",
            TensorView::new(StDtype::F32, v_shape, &v_bytes).map_err(err)?,
        ));
        if let Some((ref bytes, ref shape)) = a_data {
            views.push((
                "audio",
                TensorView::new(StDtype::F32, shape.clone(), bytes).map_err(err)?,
            ));
        }
        if let Some((ref bytes, ref shape)) = s1_data {
            views.push((
                "video_stage1",
                TensorView::new(StDtype::F32, shape.clone(), bytes).map_err(err)?,
            ));
        }
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        safetensors::serialize_to_file(views, Some(self.meta.to_map()), path)
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Прочитать только метаданные из заголовка `.safetensors`.
    pub fn read_meta(path: &Path) -> R<TailMeta> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (_, header) = SafeTensors::read_metadata(&bytes)
            .map_err(|e| format!("{}: неверный safetensors: {e}", path.display()))?;
        let map = header
            .metadata()
            .as_ref()
            .ok_or_else(|| format!("{}: в safetensors нет метаданных хвоста", path.display()))?;
        TailMeta::from_map(map)
    }

    /// Загрузить хвост из `.safetensors` на CPU (F32).
    pub fn load(path: &Path) -> R<Self> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (_, header) = SafeTensors::read_metadata(&bytes)
            .map_err(|e| format!("{}: неверный safetensors: {e}", path.display()))?;
        let map = header
            .metadata()
            .as_ref()
            .ok_or_else(|| format!("{}: в safetensors нет метаданных хвоста", path.display()))?;
        let meta = TailMeta::from_map(map)?;
        let st = SafeTensors::deserialize(&bytes)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let video = view_to_tensor(&st.tensor("video").map_err(err)?)?;
        let audio = st
            .tensor("audio")
            .ok()
            .map(|v| view_to_tensor(&v))
            .transpose()?;
        let video_stage1 = st
            .tensor("video_stage1")
            .ok()
            .map(|v| view_to_tensor(&v))
            .transpose()?;
        Ok(Self { meta, video, audio, video_stage1 })
    }
}

fn tensor_to_f32_bytes(t: &Tensor) -> R<Vec<u8>> {
    let vals: Vec<f32> = t
        .to_device(Device::Cpu)
        .map_err(err)?
        .to_dtype(DType::F32)
        .map_err(err)?
        .flatten_all()
        .map_err(err)?
        .to_vec1::<f32>()
        .map_err(err)?;
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

fn view_to_tensor(v: &TensorView<'_>) -> R<Tensor> {
    if v.dtype() != StDtype::F32 {
        return Err(format!("tail safetensors: ожидался F32, получен {:?}", v.dtype()));
    }
    let data = v.data();
    let vals: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Tensor::from_vec(vals, v.shape().to_vec(), Device::Cpu).map_err(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_safetensors() {
        let dir = std::env::temp_dir().join(format!("synaptix-tail-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("tail.safetensors");
        let v = Tensor::from_vec((0..48).map(|i| i as f32 * 0.1).collect(), vec![1, 24, 2, 1, 1], Device::Cpu).unwrap();
        let a = Tensor::from_vec((0..64).map(|i| i as f32 * -0.05).collect(), vec![1, 32, 2, 1], Device::Cpu).unwrap();
        let tail = TailLatent {
            meta: TailMeta {
                family: "h3".into(),
                model: "minimax-h3-fl2va".into(),
                width: 1344,
                height: 768,
                fps: 24.0,
                context_frames: 2,
                overlap_frames: 5,
                audio_frames: 1,
            },
            video: v.clone(),
            audio: Some(a.clone()),
            video_stage1: None,
        };
        tail.save(&p).unwrap();
        let meta = TailLatent::read_meta(&p).unwrap();
        assert_eq!(meta, tail.meta);
        let loaded = TailLatent::load(&p).unwrap();
        assert_eq!(loaded.meta, tail.meta);
        assert_eq!(loaded.video.dims(), v.dims());
        assert_eq!(loaded.audio.as_ref().unwrap().dims(), a.dims());
        // Проверка несовместимости
        assert!(loaded.meta.check_compat("ltx", None, None, None, None).is_err());
        assert!(loaded.meta.check_compat("h3", Some("minimax-h3-ref2va"), None, None, None).is_err());
        assert!(loaded.meta.check_compat("h3", None, Some(1024), None, None).is_err());
        assert!(loaded.meta.check_compat("h3", Some("minimax-h3-fl2va"), Some(1344), Some(768), Some(24.0)).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
