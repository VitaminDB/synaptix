use std::path::Path;
use std::sync::Arc;

use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_nn::quant_linear::QuantLinear;
use synaptix_nn::Module;

use crate::config::{Audio8Config, GenDefaults};
use crate::{Audio8Error, Result};

const MAIN: &str = "main";
const CODEC: &str = "codec";

/// Открытый `.syn`-бандл Audio8: модель (`main`) и кодек (`codec`).
pub struct Checkpoint {
    bundle: Arc<Bundle>,
    main: SynBundleLoader,
    codec: SynBundleLoader,
    pub config: Audio8Config,
    pub gen: GenDefaults,
    pub device: Device,
    /// Счётный тип LM (BF16 на CUDA, F32 на CPU).
    pub compute: DType,
    /// Квант линейных слоёв LM на лету (NVFP4/MXFP8/Sq); `None` — плотные.
    pub quant: Option<DType>,
    /// Счётный тип кодека.
    pub codec_dtype: DType,
}

impl Checkpoint {
    pub fn open(path: &Path, device: Device, compute: DType, quant: Option<DType>) -> Result<Self> {
        if !path.exists() {
            return Err(Audio8Error::Load(format!("нет файла {}", path.display())));
        }
        let bundle = Bundle::open(path).map_err(|e| Audio8Error::Load(e.to_string()))?;
        let cfg = bundle.read_file("config.json").map_err(|e| Audio8Error::Load(format!("config.json: {e}")))?;
        let config = Audio8Config::from_json_bytes(&cfg)?;
        let gen = bundle
            .read_file("generation_config.json")
            .ok()
            .and_then(|b| serde_json::from_slice::<GenDefaults>(&b).ok())
            .unwrap_or_default();
        let open = |c: &str| -> Result<SynBundleLoader> {
            Ok(SynBundleLoader::open(path).map_err(|e| Audio8Error::Load(e.to_string()))?.with_component(c).with_device(device))
        };
        let main = open(MAIN)?;
        let codec = open(CODEC)?;
        if !codec.names().iter().any(|n| n.starts_with("decoder.")) {
            return Err(Audio8Error::Load("в бандле нет компонента codec (кодек 44.1 кГц)".into()));
        }
        // Кодек — свёртки по всей длине фразы: F32 на CPU, на карте — тоже F32
        // (как upstream на CPU; BF16-кодек заметно шумит на тихих местах).
        let codec_dtype = std::env::var("SYN_AUDIO8_CODEC_DTYPE")
            .ok()
            .and_then(|s| match s.as_str() {
                "bf16" => Some(DType::BF16),
                "f16" => Some(DType::F16),
                "f32" => Some(DType::F32),
                _ => None,
            })
            .unwrap_or(DType::F32);
        Ok(Self { bundle: Arc::new(bundle), main, codec, config, gen, device, compute, quant, codec_dtype })
    }

    pub fn read_file(&self, name: &str) -> Result<Vec<u8>> {
        self.bundle.read_file(name).map(|c| c.into_owned()).map_err(|e| Audio8Error::Load(format!("{name}: {e}")))
    }

    /// Вес модели в счётном типе на устройстве.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        self.main.load_to(name, self.device, self.compute).map_err(|e| Audio8Error::Load(format!("{name}: {e}")))
    }

    /// Вес модели в заданном типе и на заданном устройстве.
    pub fn get_as(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor> {
        self.main.load_to(name, device, dtype).map_err(|e| Audio8Error::Load(format!("{name}: {e}")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.main.names().iter().any(|n| *n == name)
    }

    /// Вес кодека (в типе кодека).
    pub fn codec(&self, name: &str) -> Result<Tensor> {
        self.codec.load_to(name, self.device, self.codec_dtype).map_err(|e| Audio8Error::Load(format!("codec {name}: {e}")))
    }

    /// Вес кодека в F32 (мелкие таблицы: кодбуки, alpha, g/v weight-norm).
    pub fn codec_f32(&self, name: &str) -> Result<Tensor> {
        self.codec.load_to(name, self.device, DType::F32).map_err(|e| Audio8Error::Load(format!("codec {name}: {e}")))
    }

    pub fn codec_has(&self, name: &str) -> bool {
        self.codec.names().iter().any(|n| *n == name)
    }

    /// Линейный слой LM: квант из бандла как есть, иначе плотный вес
    /// (`prep` — правка плотного веса и смещения до кванта, например перестановка
    /// строк q/k под split-RoPE) и квант на лету по `self.quant`.
    pub fn linear(
        &self,
        base: &str,
        bias: bool,
        quantize: bool,
        prep: Option<&dyn Fn(Tensor, Option<Tensor>) -> Result<(Tensor, Option<Tensor>)>>,
    ) -> Result<Lin> {
        let b = if bias { Some(self.get(&format!("{base}.bias"))?) } else { None };
        if let Some(q) = self.main.prequant(&format!("{base}.weight"), self.device) {
            if prep.is_some() {
                return Err(Audio8Error::Load(format!(
                    "{base}: вес квантован в бандле, а нужна правка плотного (держите attention.wqkv плотным)"
                )));
            }
            let q = q.map_err(|e| Audio8Error::Load(format!("{base}: {e}")))?;
            return Ok(Lin(QuantLinear::Quant { w: q, bias: b }));
        }
        let w = self.get(&format!("{base}.weight"))?;
        let (w, b) = match prep {
            Some(f) => f(w, b)?,
            None => (w, b),
        };
        let l = match self.quant.filter(|_| quantize && self.device.is_cuda()) {
            Some(q) => QuantLinear::build(w, b, q, self.compute)?,
            None => QuantLinear::dense(w, b)?,
        };
        Ok(Lin(l))
    }
}

pub struct Lin(pub QuantLinear);

impl Lin {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(x)?)
    }
}
