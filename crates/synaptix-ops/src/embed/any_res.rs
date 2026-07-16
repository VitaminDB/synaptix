use synaptix_core::error::{Result, SynaptixError};

#[derive(Debug, Clone, Copy)]
pub struct AnyResGrid {
    pub tile_h: usize,
    pub tile_w: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

pub fn select_anyres_grid(
    image_h: usize,
    image_w: usize,
    candidates: &[(usize, usize)],
    tile: usize,
) -> Result<AnyResGrid> {
    if candidates.is_empty() {
        return Err(SynaptixError::Unsupported("anyres: empty candidates"));
    }
    if tile == 0 {
        return Err(SynaptixError::Unsupported("anyres: tile zero"));
    }
    let aspect_image = image_h as f64 / image_w as f64;
    let mut best = candidates[0];
    let mut best_score = f64::MAX;
    for &(gh, gw) in candidates {
        let target_h = (gh * tile) as f64;
        let target_w = (gw * tile) as f64;
        let aspect = target_h / target_w;
        let score = (aspect - aspect_image).abs();
        if score < best_score {
            best = (gh, gw);
            best_score = score;
        }
    }
    Ok(AnyResGrid { tile_h: tile, tile_w: tile, grid_h: best.0, grid_w: best.1 })
}
