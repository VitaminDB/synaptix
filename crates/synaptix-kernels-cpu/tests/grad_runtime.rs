use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::grad::{GradFn, GradMeta, GradOp, NoGradGuard, try_attach_grad_fn};
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;

fn setup() {
    ensure_registered();
}

fn read_f32(t: &Tensor) -> Vec<f32> {
    t.to_vec1::<f32>().unwrap()
}

struct IdentityGradFn {
    parents: Vec<Tensor>,
}

impl GradFn for IdentityGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(self.parents.iter().map(|_| Some(output_grad.clone())).collect())
    }

    fn parents(&self) -> &[Tensor] {
        &self.parents
    }

    fn name(&self) -> &'static str {
        "IdentityGradFn"
    }
}

fn leaf_ones(numel: usize) -> Tensor {
    Tensor::ones((numel,), DType::F32, Device::Cpu)
        .unwrap()
        .requires_grad_(true)
}

#[test]
fn backward_accumulates_repeated_calls() {
    setup();
    let a = leaf_ones(3);
    let mut c = a.ones_like().unwrap();
    let gf: Arc<dyn GradFn> = Arc::new(IdentityGradFn { parents: vec![a.clone()] });
    c.set_grad_meta(Some(GradMeta::intermediate(gf)));

    c.backward().unwrap();
    let g1 = a.grad().expect("g1");
    assert_eq!(read_f32(&g1), vec![1.0, 1.0, 1.0]);

    c.backward().unwrap();
    let g2 = a.grad().expect("g2");
    assert_eq!(read_f32(&g2), vec![2.0, 2.0, 2.0]);

    a.zero_grad();
    assert!(a.grad().is_none());

    c.backward().unwrap();
    let g3 = a.grad().expect("g3");
    assert_eq!(read_f32(&g3), vec![1.0, 1.0, 1.0]);
}

#[test]
fn no_grad_blocks_attach_factory_call() {
    setup();
    let a = leaf_ones(2);
    let mut c = a.ones_like().unwrap();
    let _g = NoGradGuard::new();
    try_attach_grad_fn(GradOp::Neg { input: &a }, &mut c).unwrap();
    assert!(c.grad_meta().is_none(), "no_grad must skip attach");
}

#[test]
fn detach_strips_grad_meta() {
    setup();
    let a = leaf_ones(2);
    let d = a.detach();
    assert!(a.requires_grad());
    assert!(!d.requires_grad());
    assert!(d.grad_meta().is_none());
}

#[test]
fn requires_grad_set_clear() {
    setup();
    let a = Tensor::ones((3,), DType::F32, Device::Cpu).unwrap();
    assert!(!a.requires_grad());
    let a = a.requires_grad_(true);
    assert!(a.requires_grad());
    let a = a.requires_grad_(false);
    assert!(!a.requires_grad());
}
