use crate::error::{Result, SynaptixError};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub(crate) Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self { Self(dims.into()) }

    pub fn scalar() -> Self { Self(Vec::new()) }

    pub fn rank(&self) -> usize { self.0.len() }

    pub fn dims(&self) -> &[usize] { &self.0 }

    pub fn numel(&self) -> usize { self.0.iter().product() }

    pub fn dim(&self, i: usize) -> Result<usize> {
        self.0
            .get(i)
            .copied()
            .ok_or(SynaptixError::DimOutOfRange { dim: i, rank: self.rank() })
    }

    pub fn into_vec(self) -> Vec<usize> { self.0 }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] { &self.0 }
}

pub trait IntoShape {
    fn into_shape(self) -> Shape;
}

impl IntoShape for Shape { fn into_shape(self) -> Shape { self } }
impl IntoShape for &Shape { fn into_shape(self) -> Shape { self.clone() } }
impl IntoShape for () { fn into_shape(self) -> Shape { Shape::scalar() } }
impl IntoShape for usize { fn into_shape(self) -> Shape { Shape::new(vec![self]) } }
impl IntoShape for Vec<usize> { fn into_shape(self) -> Shape { Shape::new(self) } }
impl IntoShape for &[usize] { fn into_shape(self) -> Shape { Shape::new(self.to_vec()) } }
impl<const N: usize> IntoShape for [usize; N] { fn into_shape(self) -> Shape { Shape::new(self.to_vec()) } }
impl<const N: usize> IntoShape for &[usize; N] { fn into_shape(self) -> Shape { Shape::new(self.to_vec()) } }

impl IntoShape for (usize,) { fn into_shape(self) -> Shape { Shape::new(vec![self.0]) } }
impl IntoShape for (usize, usize) { fn into_shape(self) -> Shape { Shape::new(vec![self.0, self.1]) } }
impl IntoShape for (usize, usize, usize) {
    fn into_shape(self) -> Shape { Shape::new(vec![self.0, self.1, self.2]) }
}
impl IntoShape for (usize, usize, usize, usize) {
    fn into_shape(self) -> Shape { Shape::new(vec![self.0, self.1, self.2, self.3]) }
}
impl IntoShape for (usize, usize, usize, usize, usize) {
    fn into_shape(self) -> Shape { Shape::new(vec![self.0, self.1, self.2, self.3, self.4]) }
}
impl IntoShape for (usize, usize, usize, usize, usize, usize) {
    fn into_shape(self) -> Shape {
        Shape::new(vec![self.0, self.1, self.2, self.3, self.4, self.5])
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Dim {
    Idx(usize),
    Last,
    MinusN(usize),
}

impl Dim {
    pub fn resolve(self, rank: usize) -> Result<usize> {
        match self {
            Dim::Idx(i) => {
                if i < rank {
                    Ok(i)
                } else {
                    Err(SynaptixError::DimOutOfRange { dim: i, rank })
                }
            }
            Dim::Last => {
                if rank == 0 {
                    Err(SynaptixError::DimOutOfRange { dim: 0, rank })
                } else {
                    Ok(rank - 1)
                }
            }
            Dim::MinusN(n) => {
                if n == 0 || n > rank {
                    Err(SynaptixError::DimOutOfRange { dim: n, rank })
                } else {
                    Ok(rank - n)
                }
            }
        }
    }
}

impl From<usize> for Dim {
    fn from(i: usize) -> Self { Dim::Idx(i) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_basics() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dims(), &[2, 3, 4]);
        assert_eq!(s.dim(0).unwrap(), 2);
        assert!(s.dim(3).is_err());
    }

    #[test]
    fn into_shape_variants() {
        assert_eq!((2usize, 3).into_shape().dims(), &[2, 3]);
        assert_eq!([2, 3, 4].into_shape().dims(), &[2, 3, 4]);
        assert_eq!(vec![5usize, 6].into_shape().dims(), &[5, 6]);
        assert_eq!(().into_shape().rank(), 0);
        assert_eq!(7usize.into_shape().dims(), &[7]);
    }

    #[test]
    fn dim_resolution() {
        assert_eq!(Dim::Idx(2).resolve(5).unwrap(), 2);
        assert_eq!(Dim::Last.resolve(5).unwrap(), 4);
        assert_eq!(Dim::MinusN(1).resolve(5).unwrap(), 4);
        assert_eq!(Dim::MinusN(3).resolve(5).unwrap(), 2);
        assert!(Dim::Idx(5).resolve(5).is_err());
        assert!(Dim::Last.resolve(0).is_err());
        assert!(Dim::MinusN(6).resolve(5).is_err());
    }
}
