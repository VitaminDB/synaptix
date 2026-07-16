use crate::error::Result;
use crate::image_buf::RgbImage;
use crate::transforms::resize::resize_bilinear;

#[derive(Debug, Clone, Copy)]
pub struct AnyResConfig {
    pub tile: usize,
    pub max_tiles: usize,
}

pub fn any_res_tiles(img: &RgbImage, cfg: &AnyResConfig) -> Result<Vec<RgbImage>> {
    let src = img.to_hwc();
    let tile = cfg.tile.max(1);
    let target_w = ((src.width + tile - 1) / tile) * tile;
    let target_h = ((src.height + tile - 1) / tile) * tile;
    let mut wt = target_w / tile;
    let mut ht = target_h / tile;
    while wt * ht > cfg.max_tiles.max(1) && (wt > 1 || ht > 1) {
        if wt >= ht && wt > 1 {
            wt -= 1;
        } else if ht > 1 {
            ht -= 1;
        } else {
            break;
        }
    }
    let resized_w = wt * tile;
    let resized_h = ht * tile;
    let resized = resize_bilinear(&src, resized_w, resized_h)?;

    let mut out = Vec::with_capacity(wt * ht);
    for ty in 0..ht {
        for tx in 0..wt {
            let mut piece = RgbImage::zeros_hwc(tile, tile, resized.channels);
            for y in 0..tile {
                for x in 0..tile {
                    for c in 0..resized.channels {
                        piece.set_pixel(x, y, c, resized.pixel(tx * tile + x, ty * tile + y, c));
                    }
                }
            }
            out.push(piece);
        }
    }
    Ok(out)
}
