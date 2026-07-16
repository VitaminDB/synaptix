use synaptix_core::device::Device;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::pos::rope::{RopeLayout, apply_rope};
use crate::pos::rope_cache::RopeCache;

pub fn build_m_rope_positions(
    sections: &[usize],
    layouts: &[(usize, usize, usize)],
    device: Device,
) -> Result<Tensor> {
    if sections.len() != layouts.len() {
        return Err(SynaptixError::Other(format!(
            "m_rope: sections {} != layouts {}",
            sections.len(),
            layouts.len()
        )));
    }
    let total: usize = sections.iter().sum();
    let mut positions = Vec::with_capacity(total);
    let mut base = 0u32;
    for (&count, &(t, h, w)) in sections.iter().zip(layouts.iter()) {
        let expected = t * h * w;
        if expected != count && expected != 0 {
            return Err(SynaptixError::Other(format!(
                "m_rope: section {count} mismatch t*h*w={expected}"
            )));
        }
        if expected == 0 {
            for i in 0..count {
                positions.push(base + i as u32);
            }
            base += count as u32;
        } else {
            for ti in 0..t {
                for _hi in 0..h {
                    for _wi in 0..w {
                        positions.push(base + ti as u32);
                    }
                }
            }
            base += t as u32;
        }
    }
    Tensor::from_vec(positions, (total,), device)
}

pub fn apply_m_rope(
    x: &Tensor,
    cache: &RopeCache,
    positions: &Tensor,
) -> Result<Tensor> {
    apply_rope(x, cache, Some(positions), RopeLayout::Split)
}
