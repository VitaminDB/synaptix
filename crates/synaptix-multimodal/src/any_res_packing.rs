use synaptix_core::tensor::Tensor;

use crate::error::{MultimodalError, Result};

#[derive(Debug, Clone, Copy)]
pub struct TilePosition {
    pub row: usize,
    pub col: usize,
    pub n_rows: usize,
    pub n_cols: usize,
}

#[derive(Debug, Clone)]
pub struct AnyResPackPlan {
    pub positions: Vec<TilePosition>,
    pub tokens_per_tile: usize,
}

pub fn pack_any_res_tokens(
    tile_features: &[Tensor],
    n_rows: usize,
    n_cols: usize,
) -> Result<(Tensor, AnyResPackPlan)> {
    if tile_features.is_empty() {
        return Err(MultimodalError::invalid_arg("pack_any_res_tokens: empty tiles"));
    }
    if n_rows * n_cols != tile_features.len() {
        return Err(MultimodalError::invalid_arg(format!(
            "n_rows*n_cols = {}*{} != tile_features.len() = {}",
            n_rows,
            n_cols,
            tile_features.len()
        )));
    }
    let first = &tile_features[0];
    if first.rank() != 2 {
        return Err(MultimodalError::shape(format!(
            "tile_features[0] must be [N, D], got {:?}",
            first.dims()
        )));
    }
    let tokens_per_tile = first.dims()[0];
    let dim = first.dims()[1];
    for (i, t) in tile_features.iter().enumerate() {
        if t.rank() != 2 || t.dims()[0] != tokens_per_tile || t.dims()[1] != dim {
            return Err(MultimodalError::shape(format!(
                "tile_features[{i}] shape {:?} mismatch [{tokens_per_tile}, {dim}]",
                t.dims()
            )));
        }
    }
    let mut positions = Vec::with_capacity(tile_features.len());
    for row in 0..n_rows {
        for col in 0..n_cols {
            positions.push(TilePosition { row, col, n_rows, n_cols });
        }
    }
    let refs: Vec<&Tensor> = tile_features.iter().collect();
    let packed = Tensor::cat(&refs, 0).map_err(MultimodalError::from)?;
    Ok((packed, AnyResPackPlan { positions, tokens_per_tile }))
}
