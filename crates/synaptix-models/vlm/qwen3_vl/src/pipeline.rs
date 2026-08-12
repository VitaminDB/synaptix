use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::SynaptixError;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_io::weights::WeightLoader;
use synaptix_tokenizer::{HfTokenizer, Tokenizer as _};

use crate::config::VisionConfig;
use crate::model::{VisionError, VisionTower, VisionWeights};
use crate::preprocess::{prepare_tensor, ImageGrid, PreprocessLimits};
use crate::presentation::{assemble, EncodedPresentation, H3Presentation};
use crate::text_model::{build_mrope, rope_positions, TextConfig, TextEncoder, VisionSpan};

pub const H3_ENCODER_LAYERS: usize = 50;

pub struct DirWeights {
    loader: SafetensorsLoader,
}

impl DirWeights {
    pub fn open(dir: impl AsRef<Path>, device: Device) -> Result<Self, VisionError> {
        let dir = dir.as_ref();
        let shards =
            scan_shards(dir).map_err(|e| VisionError::Load(format!("{}: {e}", dir.display())))?;
        if shards.is_empty() {
            return Err(VisionError::Load(format!("нет safetensors в {}", dir.display())));
        }
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| VisionError::Load(e.to_string()))?
            .with_device(device);
        Ok(Self { loader })
    }

    pub fn contains(&self, key: &str) -> bool {
        self.loader.names().iter().any(|n| *n == key)
    }
}

impl VisionWeights for DirWeights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, VisionError> {
        self.loader
            .load_to(key, device, dtype)
            .map_err(|e| VisionError::Load(format!("{key}: {e}")))
    }
}

pub struct H3Encoder {
    pub vision: VisionTower,
    pub text: TextEncoder,
    tokenizer: HfTokenizer,
    limits: PreprocessLimits,
    device: Device,
    dtype: DType,
}

pub struct H3Conditioning {
    pub hidden: Tensor,
    pub tags: Vec<u8>,
}

impl H3Encoder {
    pub fn load(
        encoder_dir: impl AsRef<Path>,
        tokenizer_json: Option<PathBuf>,
        device: Device,
        compute: DType,
        quant: DType,
        layers: usize,
    ) -> Result<Self, VisionError> {
        let dir = encoder_dir.as_ref();
        let cfg_bytes = std::fs::read(dir.join("config.json"))
            .map_err(|e| VisionError::Load(format!("config.json: {e}")))?;
        let vcfg = VisionConfig::from_hf_bytes(&cfg_bytes)
            .map_err(|e| VisionError::Load(e.to_string()))?;
        let tcfg = TextConfig::from_hf_bytes(&cfg_bytes)?;

        let weights = DirWeights::open(dir, device)?;
        let vision = VisionTower::build(vcfg, &weights, device, compute)?;
        let text = TextEncoder::build(tcfg, &weights, device, compute, quant, layers)?;

        let tok_path = tokenizer_json.unwrap_or_else(|| dir.join("tokenizer.json"));
        let tokenizer = HfTokenizer::from_file(&tok_path)
            .map_err(|e| VisionError::Load(format!("{}: {e}", tok_path.display())))?;

        Ok(Self {
            vision,
            text,
            tokenizer,
            limits: PreprocessLimits::default(),
            device,
            dtype: compute,
        })
    }

    pub fn with_limits(mut self, limits: PreprocessLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn merge_size(&self) -> usize {
        self.vision.config.spatial_merge_size
    }

    pub fn prepare_image(
        &self,
        rgb: &Tensor,
    ) -> Result<(Tensor, ImageGrid), VisionError> {
        let prepared = prepare_tensor(rgb, &self.vision.config, self.limits, self.device)
            .map_err(|e| VisionError::Forward(e.to_string()))?;
        Ok((prepared.patches, prepared.grid))
    }

    pub fn encode(
        &self,
        presentation: &H3Presentation,
        images: &[(Tensor, ImageGrid)],
    ) -> Result<H3Conditioning, VisionError> {
        let tok = &self.tokenizer;
        let encoded: EncodedPresentation = assemble(presentation, |s| {
            tok.encode(s, false).map(|e| e.ids).unwrap_or_default()
        });
        if encoded.vision_grids.len() != images.len() {
            return Err(VisionError::Forward(format!(
                "презентация ожидает {} vision-блоков, передано {}",
                encoded.vision_grids.len(),
                images.len()
            )));
        }

        if let Ok(dir) = std::env::var("H3_DUMP_DIR") {
            let p = std::path::Path::new(&dir);
            let _ = std::fs::create_dir_all(p);
            let _ = std::fs::write(p.join("tokens.json"), format!("{:?}", encoded.ids));
        }

        let mut hidden = self.text.embed_tokens(&encoded.ids)?;
        let e = |r: Result<Tensor, SynaptixError>| {
            r.map_err(|x| VisionError::Forward(x.to_string()))
        };

        let deepstack_slots = self.vision.deepstack_len();
        let mut deepstack_feats: Vec<Vec<Tensor>> = vec![Vec::new(); deepstack_slots];
        let mut deepstack_rows: Vec<Vec<usize>> = vec![Vec::new(); deepstack_slots];
        let mut spans: Vec<VisionSpan> = Vec::with_capacity(images.len());

        for (bi, (patches, grid)) in images.iter().enumerate() {
            let (merged, taps) = self.vision.forward_deepstack(patches, *grid)?;
            let rows = &encoded.vision_rows[bi];
            hidden = replace_rows(&hidden, &merged, rows)?;
            for (slot, tap) in taps.into_iter().enumerate() {
                if slot < deepstack_slots {
                    deepstack_feats[slot].push(tap);
                    deepstack_rows[slot].extend(rows.iter().copied());
                }
            }
            let merge = self.merge_size();
            spans.push(VisionSpan {
                start: rows[0],
                len: rows.len(),
                grid_t: grid.t,
                grid_h: grid.h / merge,
                grid_w: grid.w / merge,
            });
        }

        let mut deepstack: Vec<(Tensor, Vec<usize>)> = Vec::with_capacity(deepstack_slots);
        for slot in 0..deepstack_slots {
            if deepstack_feats[slot].is_empty() {
                continue;
            }
            let refs: Vec<&Tensor> = deepstack_feats[slot].iter().collect();
            let cat = e(Tensor::cat(&refs, 0))?;
            deepstack.push((cat, std::mem::take(&mut deepstack_rows[slot])));
        }

        let positions = rope_positions(encoded.ids.len(), &spans);
        let rope = build_mrope(&positions, &self.text.config, self.device)?;
        let out = self.text.forward(&hidden, &rope, &deepstack)?;
        let l = out.dims()[0];
        let d = out.dims()[1];
        Ok(H3Conditioning {
            hidden: e(out.reshape(vec![1, l, d]))?,
            tags: encoded.tags,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

fn replace_rows(x: &Tensor, src: &Tensor, rows: &[usize]) -> Result<Tensor, VisionError> {
    if rows.is_empty() {
        return Ok(x.clone());
    }
    let e = |r: Result<Tensor, SynaptixError>| r.map_err(|v| VisionError::Forward(v.to_string()));
    let n = x.dims()[0];
    let start = rows[0];
    let len = rows.len();
    let contiguous_block = rows.iter().enumerate().all(|(i, r)| *r == start + i);
    if !contiguous_block {
        return Err(VisionError::Forward("vision-строки не непрерывны".into()));
    }
    let src = if src.dtype() == x.dtype() { src.clone() } else { e(src.to_dtype(x.dtype()))? };
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if start > 0 {
        parts.push(e(e(x.narrow(0, 0, start))?.contiguous())?);
    }
    parts.push(src);
    if start + len < n {
        parts.push(e(e(x.narrow(0, start + len, n - start - len))?.contiguous())?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    e(Tensor::cat(&refs, 0))
}
