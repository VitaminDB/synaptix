use crate::tensor::shape::Shape;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(pub(crate) Vec<isize>);

impl Strides {
    pub fn new(s: impl Into<Vec<isize>>) -> Self { Self(s.into()) }

    pub fn contiguous(shape: &Shape) -> Self {
        let dims = shape.dims();
        let mut strides = vec![1isize; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1] as isize;
        }
        Self(strides)
    }

    pub fn as_slice(&self) -> &[isize] { &self.0 }

    pub fn len(&self) -> usize { self.0.len() }

    pub fn is_empty(&self) -> bool { self.0.is_empty() }

    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        let dims = shape.dims();
        if dims.len() != self.0.len() { return false; }
        let mut expected = 1isize;
        for i in (0..dims.len()).rev() {
            if dims[i] == 1 { continue; }
            if self.0[i] != expected { return false; }
            expected *= dims[i] as isize;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_3d() {
        let s = Strides::contiguous(&Shape::new(vec![2, 3, 4]));
        assert_eq!(s.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn contiguous_strides_1d() {
        let s = Strides::contiguous(&Shape::new(vec![5]));
        assert_eq!(s.as_slice(), &[1]);
    }

    #[test]
    fn contiguous_strides_scalar() {
        let s = Strides::contiguous(&Shape::scalar());
        assert!(s.is_empty());
    }

    #[test]
    fn is_contiguous_check() {
        let shape = Shape::new(vec![2, 3, 4]);
        let contig = Strides::contiguous(&shape);
        assert!(contig.is_contiguous(&shape));

        let transposed = Strides::new(vec![1isize, 8, 2]);
        assert!(!transposed.is_contiguous(&shape));
    }

    #[test]
    fn is_contiguous_with_unit_dim() {
        let shape = Shape::new(vec![2, 1, 4]);
        let strides = Strides::new(vec![4isize, 0, 1]);
        assert!(strides.is_contiguous(&shape));
    }
}
