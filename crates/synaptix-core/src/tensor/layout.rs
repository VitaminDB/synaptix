use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::tensor::shape::Shape;
use crate::tensor::strides::Strides;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) offset: usize,
    pub(crate) dtype: DType,
}

impl Layout {
    pub fn contiguous(shape: Shape, dtype: DType) -> Self {
        let strides = Strides::contiguous(&shape);
        Self { shape, strides, offset: 0, dtype }
    }

    pub fn shape(&self) -> &Shape { &self.shape }
    pub fn strides(&self) -> &Strides { &self.strides }
    pub fn offset(&self) -> usize { self.offset }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn rank(&self) -> usize { self.shape.rank() }
    pub fn numel(&self) -> usize { self.shape.numel() }
    pub fn dims(&self) -> &[usize] { self.shape.dims() }
    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides.is_contiguous(&self.shape)
    }

    pub fn reshape(&self, new: Shape) -> Result<Self> {
        if !self.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        if self.shape.numel() != new.numel() {
            return Err(SynaptixError::ReshapeMismatch {
                from: self.shape.dims().to_vec(),
                to: new.dims().to_vec(),
            });
        }
        let strides = Strides::contiguous(&new);
        Ok(Self { shape: new, strides, offset: self.offset, dtype: self.dtype })
    }

    pub fn transpose(&self, d1: usize, d2: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if d1 >= rank {
            return Err(SynaptixError::DimOutOfRange { dim: d1, rank });
        }
        if d2 >= rank {
            return Err(SynaptixError::DimOutOfRange { dim: d2, rank });
        }
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.0.clone();
        dims.swap(d1, d2);
        strides.swap(d1, d2);
        Ok(Self {
            shape: Shape::new(dims),
            strides: Strides::new(strides),
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        let rank = self.shape.rank();
        if perm.len() != rank {
            return Err(SynaptixError::RankMismatch { expected: rank, got: perm.len() });
        }
        let mut seen = vec![false; rank];
        for &p in perm {
            if p >= rank {
                return Err(SynaptixError::DimOutOfRange { dim: p, rank });
            }
            if seen[p] {
                return Err(SynaptixError::Unsupported("permute: repeated axis"));
            }
            seen[p] = true;
        }
        let dims = self.shape.dims();
        let strides = self.strides.as_slice();
        let new_dims: Vec<usize> = perm.iter().map(|&p| dims[p]).collect();
        let new_strides: Vec<isize> = perm.iter().map(|&p| strides[p]).collect();
        Ok(Self {
            shape: Shape::new(new_dims),
            strides: Strides::new(new_strides),
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    pub fn unsqueeze(&self, d: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if d > rank {
            return Err(SynaptixError::DimOutOfRange { dim: d, rank });
        }
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.0.clone();
        let new_stride = if d < strides.len() {
            strides[d].saturating_mul(dims[d] as isize)
        } else if let Some(&last) = strides.last() {
            last
        } else {
            1
        };
        dims.insert(d, 1);
        strides.insert(d, new_stride);
        Ok(Self {
            shape: Shape::new(dims),
            strides: Strides::new(strides),
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    pub fn squeeze(&self, d: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if d >= rank {
            return Err(SynaptixError::DimOutOfRange { dim: d, rank });
        }
        let dims = self.shape.dims();
        if dims[d] != 1 {
            return Err(SynaptixError::Unsupported("squeeze: dim size != 1"));
        }
        let mut new_dims = dims.to_vec();
        let mut new_strides = self.strides.0.clone();
        new_dims.remove(d);
        new_strides.remove(d);
        Ok(Self {
            shape: Shape::new(new_dims),
            strides: Strides::new(new_strides),
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    pub fn narrow(&self, d: usize, off: usize, len: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if d >= rank {
            return Err(SynaptixError::DimOutOfRange { dim: d, rank });
        }
        let size = self.shape.dims()[d];
        if off + len > size {
            return Err(SynaptixError::NarrowOutOfBounds { dim: d, off, len, size });
        }
        let mut new_dims = self.shape.dims().to_vec();
        new_dims[d] = len;
        let stride = self.strides.as_slice()[d];
        let new_offset_delta = (off as isize) * stride;
        if new_offset_delta < 0 {
            return Err(SynaptixError::Unsupported("narrow on negative-stride axis"));
        }
        Ok(Self {
            shape: Shape::new(new_dims),
            strides: self.strides.clone(),
            offset: self.offset + new_offset_delta as usize,
            dtype: self.dtype,
        })
    }

    pub fn expand(&self, target: &Shape) -> Result<Self> {
        let target_dims = target.dims();
        let self_dims = self.shape.dims();
        if target_dims.len() < self_dims.len() {
            return Err(SynaptixError::ShapeMismatch {
                expected: target_dims.to_vec(),
                got: self_dims.to_vec(),
            });
        }
        let offset_axes = target_dims.len() - self_dims.len();
        let mut new_strides = vec![0isize; target_dims.len()];
        for (i, &dim) in target_dims.iter().enumerate() {
            if i < offset_axes {
                if dim == 0 {
                    return Err(SynaptixError::ShapeMismatch {
                        expected: target_dims.to_vec(),
                        got: self_dims.to_vec(),
                    });
                }
                new_strides[i] = 0;
            } else {
                let self_axis = i - offset_axes;
                let self_dim = self_dims[self_axis];
                let self_stride = self.strides.as_slice()[self_axis];
                if self_dim == dim {
                    new_strides[i] = self_stride;
                } else if self_dim == 1 {
                    new_strides[i] = 0;
                } else {
                    return Err(SynaptixError::ShapeMismatch {
                        expected: target_dims.to_vec(),
                        got: self_dims.to_vec(),
                    });
                }
            }
        }
        Ok(Self {
            shape: target.clone(),
            strides: Strides::new(new_strides),
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    pub fn flatten_all(&self) -> Result<Self> {
        if !self.is_contiguous() {
            return Err(SynaptixError::NonContiguous);
        }
        let n = self.numel();
        Ok(Self::contiguous(Shape::new(vec![n]), self.dtype))
    }

    pub fn squeeze_all(&self) -> Self {
        let mut dims = Vec::new();
        let mut strides = Vec::new();
        for (d, &s) in self.shape.dims().iter().zip(self.strides.as_slice()) {
            if *d != 1 {
                dims.push(*d);
                strides.push(s);
            }
        }
        Self {
            shape: Shape::new(dims),
            strides: Strides::new(strides),
            offset: self.offset,
            dtype: self.dtype,
        }
    }

    pub fn byte_offset(&self) -> usize {
        self.offset * (self.dtype.size_in_bits() / 8).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_layout() {
        let l = Layout::contiguous(Shape::new(vec![2, 3]), DType::F32);
        assert!(l.is_contiguous());
        assert_eq!(l.strides().as_slice(), &[3, 1]);
        assert_eq!(l.numel(), 6);
    }

    #[test]
    fn transpose_layout() {
        let l = Layout::contiguous(Shape::new(vec![2, 3]), DType::F32);
        let t = l.transpose(0, 1).unwrap();
        assert_eq!(t.shape().dims(), &[3, 2]);
        assert_eq!(t.strides().as_slice(), &[1, 3]);
        assert!(!t.is_contiguous());

        let tt = t.transpose(0, 1).unwrap();
        assert_eq!(tt.shape().dims(), &[2, 3]);
        assert!(tt.is_contiguous());
    }

    #[test]
    fn permute_layout() {
        let l = Layout::contiguous(Shape::new(vec![2, 3, 4]), DType::F32);
        let p = l.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape().dims(), &[4, 2, 3]);
        assert_eq!(p.strides().as_slice(), &[1, 12, 4]);
    }

    #[test]
    fn reshape_only_contig() {
        let l = Layout::contiguous(Shape::new(vec![2, 3]), DType::F32);
        let r = l.reshape(Shape::new(vec![6])).unwrap();
        assert_eq!(r.shape().dims(), &[6]);
        assert!(r.is_contiguous());

        let bad = l.reshape(Shape::new(vec![5]));
        assert!(bad.is_err());

        let t = l.transpose(0, 1).unwrap();
        assert!(t.reshape(Shape::new(vec![6])).is_err());
    }

    #[test]
    fn narrow_offsets_correctly() {
        let l = Layout::contiguous(Shape::new(vec![4, 5]), DType::F32);
        let n = l.narrow(0, 1, 2).unwrap();
        assert_eq!(n.shape().dims(), &[2, 5]);
        assert_eq!(n.offset(), 5);
        assert_eq!(n.strides().as_slice(), &[5, 1]);
    }

    #[test]
    fn expand_broadcasts_with_zero_stride() {
        let l = Layout::contiguous(Shape::new(vec![1, 3]), DType::F32);
        let e = l.expand(&Shape::new(vec![4, 3])).unwrap();
        assert_eq!(e.shape().dims(), &[4, 3]);
        assert_eq!(e.strides().as_slice(), &[0, 1]);
    }

    #[test]
    fn unsqueeze_squeeze_roundtrip() {
        let l = Layout::contiguous(Shape::new(vec![2, 3]), DType::F32);
        let u = l.unsqueeze(1).unwrap();
        assert_eq!(u.shape().dims(), &[2, 1, 3]);
        let s = u.squeeze(1).unwrap();
        assert_eq!(s.shape().dims(), &[2, 3]);
    }

    #[test]
    fn flatten_all_contig() {
        let l = Layout::contiguous(Shape::new(vec![2, 3, 4]), DType::F32);
        let f = l.flatten_all().unwrap();
        assert_eq!(f.shape().dims(), &[24]);
    }
}
