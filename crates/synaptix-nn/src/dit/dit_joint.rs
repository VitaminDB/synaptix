use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::dit::dit_block::DitBlock;

/// Dual-stream DiT (SD3 / Flux схема, semantic-минимальный вариант).
///
/// На каждом слое image- и text-потоки проходят через **независимые**
/// `DitBlock`'и с общим conditioning'ом. Реальная SD3/Flux схема дополнительно
/// объединяет img+txt в одну MHA на каждом слое — здесь оставлено dual-stream
/// без cross-stream взаимодействия (для bit-exact теста зацементирован
/// публичный API). Возвращает обработанный image-стрим.
pub struct DitJoint {
    pub img_blocks: Vec<DitBlock>,
    pub txt_blocks: Vec<DitBlock>,
    pub hidden_size: usize,
}

impl DitJoint {
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        ffn_dim: usize,
        cond_dim: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut img_blocks = Vec::with_capacity(num_layers);
        let mut txt_blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            img_blocks.push(DitBlock::new(hidden_size, num_heads, ffn_dim, cond_dim, device, dtype)?);
            txt_blocks.push(DitBlock::new(hidden_size, num_heads, ffn_dim, cond_dim, device, dtype)?);
        }
        Ok(Self { img_blocks, txt_blocks, hidden_size })
    }

    pub fn from_blocks(img_blocks: Vec<DitBlock>, txt_blocks: Vec<DitBlock>) -> Result<Self> {
        if img_blocks.is_empty() {
            return Err(SynaptixError::Unsupported("DitJoint: empty img_blocks"));
        }
        if img_blocks.len() != txt_blocks.len() {
            return Err(SynaptixError::Unsupported("DitJoint: img/txt blocks must have equal length"));
        }
        let hidden_size = img_blocks[0].hidden_size;
        Ok(Self { img_blocks, txt_blocks, hidden_size })
    }

    /// `img: [B, T_img, hidden]`, `txt: [B, T_txt, hidden]`, `cond: [B, cond_dim]`.
    /// Возвращает обработанный image-стрим `[B, T_img, hidden]`. Внутренний
    /// txt-стрим прогоняется параллельно (нужен для согласованных обновлений
    /// adaLN-модулей; результат txt не возвращается caller'у в этом stub'е).
    pub fn forward(&self, img: &Tensor, txt: &Tensor, cond: &Tensor) -> Result<Tensor> {
        if img.rank() != 3 || img.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported("DitJoint: img must be [B, T_img, hidden]"));
        }
        if txt.rank() != 3 || txt.dims()[2] != self.hidden_size {
            return Err(SynaptixError::Unsupported("DitJoint: txt must be [B, T_txt, hidden]"));
        }
        let mut img_h = img.clone();
        let mut txt_h = txt.clone();
        for (img_b, txt_b) in self.img_blocks.iter().zip(self.txt_blocks.iter()) {
            img_h = img_b.forward(&img_h, cond)?;
            txt_h = txt_b.forward(&txt_h, cond)?;
        }
        // Возвращаем именно img-стрим; txt_h собран ради согласованного
        // прогона (чтобы forward читался один-в-один с Python-эталоном).
        let _ = txt_h;
        Ok(img_h)
    }
}
