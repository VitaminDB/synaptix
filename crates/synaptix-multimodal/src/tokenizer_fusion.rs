use synaptix_core::tensor::Tensor;

use crate::error::{MultimodalError, Result};

#[derive(Debug, Clone)]
pub enum FusionSpan {
    Text { token_ids: Vec<u32> },
    ImageFeature { token_count: usize, source_idx: usize },
    AudioFeature { token_count: usize, source_idx: usize },
}

#[derive(Debug, Clone)]
pub struct FusionPlan {
    pub spans: Vec<FusionSpan>,
}

impl FusionPlan {
    pub fn from_text_with_image_marker(
        token_ids: &[u32],
        image_marker_id: u32,
        feature_token_counts: &[usize],
    ) -> Result<Self> {
        let occurrences = token_ids.iter().filter(|&&t| t == image_marker_id).count();
        if occurrences != feature_token_counts.len() {
            return Err(MultimodalError::invalid_arg(format!(
                "marker count {occurrences} != feature batches {}",
                feature_token_counts.len()
            )));
        }
        let mut spans = Vec::new();
        let mut buffer: Vec<u32> = Vec::new();
        let mut feature_idx = 0usize;
        for &id in token_ids {
            if id == image_marker_id {
                if !buffer.is_empty() {
                    spans.push(FusionSpan::Text { token_ids: std::mem::take(&mut buffer) });
                }
                spans.push(FusionSpan::ImageFeature {
                    token_count: feature_token_counts[feature_idx],
                    source_idx: feature_idx,
                });
                feature_idx += 1;
            } else {
                buffer.push(id);
            }
        }
        if !buffer.is_empty() {
            spans.push(FusionSpan::Text { token_ids: buffer });
        }
        Ok(Self { spans })
    }

    pub fn total_tokens(&self) -> usize {
        self.spans
            .iter()
            .map(|s| match s {
                FusionSpan::Text { token_ids } => token_ids.len(),
                FusionSpan::ImageFeature { token_count, .. } => *token_count,
                FusionSpan::AudioFeature { token_count, .. } => *token_count,
            })
            .sum()
    }
}

pub fn fuse_image_features(
    plan: &FusionPlan,
    text_embed_table: &Tensor,
    image_features: &[Tensor],
) -> Result<Tensor> {
    if text_embed_table.rank() != 2 {
        return Err(MultimodalError::shape(format!(
            "text_embed_table must be [V, D], got {:?}",
            text_embed_table.dims()
        )));
    }
    let dim = text_embed_table.dims()[1];
    for (i, f) in image_features.iter().enumerate() {
        if f.rank() != 2 || f.dims()[1] != dim {
            return Err(MultimodalError::shape(format!(
                "image_features[{i}] must be [N, {dim}], got {:?}",
                f.dims()
            )));
        }
    }
    let mut pieces: Vec<Tensor> = Vec::new();
    for span in &plan.spans {
        match span {
            FusionSpan::Text { token_ids } => {
                let indices = Tensor::from_vec(
                    token_ids.to_vec(),
                    (token_ids.len(),),
                    text_embed_table.device(),
                )?;
                let emb = text_embed_table.index_select(0, &indices)?;
                pieces.push(emb);
            }
            FusionSpan::ImageFeature { source_idx, token_count } => {
                let feat = &image_features[*source_idx];
                if feat.dims()[0] != *token_count {
                    return Err(MultimodalError::shape(format!(
                        "image_features[{source_idx}].dims[0] {} != token_count {}",
                        feat.dims()[0],
                        token_count
                    )));
                }
                pieces.push(feat.clone());
            }
            FusionSpan::AudioFeature { .. } => {
                return Err(MultimodalError::invalid_arg(
                    "audio fusion: pass audio features через отдельную функцию",
                ));
            }
        }
    }
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Tensor::cat(&refs, 0).map_err(MultimodalError::from)
}
