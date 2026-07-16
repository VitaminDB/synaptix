use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;
use crate::tensor::layout::Layout;
use crate::tensor::shape::{IntoShape, Shape};

impl Tensor {
    pub fn expand<S: IntoShape>(&self, shape: S) -> Result<Self> {
        let target = shape.into_shape();
        let layout = self.layout.expand(&target)?;
        Ok(self.with_layout(layout))
    }

    pub fn broadcast_as<S: IntoShape>(&self, shape: S) -> Result<Self> {
        self.expand(shape)
    }
}

pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Shape> {
    let nd = a.len().max(b.len());
    let mut out = vec![0usize; nd];
    for i in 0..nd {
        let ai = if i + a.len() < nd { 1 } else { a[i + a.len() - nd] };
        let bi = if i + b.len() < nd { 1 } else { b[i + b.len() - nd] };
        if ai == bi {
            out[i] = ai;
        } else if ai == 1 {
            out[i] = bi;
        } else if bi == 1 {
            out[i] = ai;
        } else {
            return Err(SynaptixError::BroadcastMismatch {
                lhs: a.to_vec(),
                rhs: b.to_vec(),
            });
        }
    }
    Ok(Shape::new(out))
}

pub(crate) fn broadcast_layouts(a: &Layout, b: &Layout) -> Result<(Layout, Layout, Shape)> {
    let target = broadcast_shape(a.dims(), b.dims())?;
    let a2 = a.expand(&target)?;
    let b2 = b.expand(&target)?;
    Ok((a2, b2, target))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_simple() {
        let s = broadcast_shape(&[3, 1], &[1, 4]).unwrap();
        assert_eq!(s.dims(), &[3, 4]);
    }

    #[test]
    fn broadcast_rank_diff() {
        let s = broadcast_shape(&[5], &[3, 1, 5]).unwrap();
        assert_eq!(s.dims(), &[3, 1, 5]);
    }

    #[test]
    fn broadcast_incompatible() {
        assert!(broadcast_shape(&[3, 2], &[1, 4]).is_err());
    }
}
