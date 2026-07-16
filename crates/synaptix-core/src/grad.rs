use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::dtype::DType;
use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;

pub trait GradFn: Send + Sync {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>>;

    fn parents(&self) -> &[Tensor];

    fn name(&self) -> &'static str;
}

pub type BackwardHook = Arc<dyn Fn(&Tensor) -> Option<Tensor> + Send + Sync>;

pub struct GradMeta {
    requires_grad: AtomicBool,
    is_leaf: bool,
    grad: Mutex<Option<Tensor>>,
    grad_fn: Option<Arc<dyn GradFn>>,
    hooks: Mutex<Vec<BackwardHook>>,
}

impl GradMeta {
    pub fn leaf(requires_grad: bool) -> Arc<Self> {
        Arc::new(Self {
            requires_grad: AtomicBool::new(requires_grad),
            is_leaf: true,
            grad: Mutex::new(None),
            grad_fn: None,
            hooks: Mutex::new(Vec::new()),
        })
    }

    pub fn intermediate(grad_fn: Arc<dyn GradFn>) -> Arc<Self> {
        Arc::new(Self {
            requires_grad: AtomicBool::new(true),
            is_leaf: false,
            grad: Mutex::new(None),
            grad_fn: Some(grad_fn),
            hooks: Mutex::new(Vec::new()),
        })
    }

    pub fn register_hook(&self, hook: BackwardHook) {
        if let Ok(mut h) = self.hooks.lock() {
            h.push(hook);
        }
    }

    pub fn run_hooks(&self, grad: &Tensor) -> Tensor {
        let mut g = grad.clone();
        if let Ok(hooks) = self.hooks.lock() {
            for h in hooks.iter() {
                if let Some(new_g) = h(&g) {
                    g = new_g;
                }
            }
        }
        g
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad.load(Ordering::Relaxed)
    }

    pub fn set_requires_grad(&self, value: bool) {
        self.requires_grad.store(value, Ordering::Relaxed);
    }

    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    pub fn grad_fn(&self) -> Option<&Arc<dyn GradFn>> {
        self.grad_fn.as_ref()
    }

    pub fn grad(&self) -> Option<Tensor> {
        self.grad.lock().ok().and_then(|g| g.clone())
    }

    pub fn zero_grad(&self) {
        if let Ok(mut g) = self.grad.lock() {
            *g = None;
        }
    }

    pub(crate) fn accumulate(&self, incoming: Tensor) -> Result<()> {
        let mut guard = self
            .grad
            .lock()
            .map_err(|_| SynaptixError::Other("GradMeta.grad mutex poisoned".into()))?;
        match guard.as_ref() {
            None => *guard = Some(incoming),
            Some(existing) => {
                let summed = existing.add(&incoming)?;
                *guard = Some(summed);
            }
        }
        Ok(())
    }
}

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|c| c.get())
}

pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = GRAD_ENABLED.with(|c| {
            let p = c.get();
            c.set(false);
            p
        });
        Self { prev }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        GRAD_ENABLED.with(|c| c.set(prev));
    }
}

pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _g = NoGradGuard::new();
    f()
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryGradKind {
    Erf,
    Sigmoid,
    GeLUExact,
    GeLUTanh,
    QuickGelu,
    SiLU,
    SwishBeta,
    Relu,
    Relu2,
    LeakyRelu,
    PRelu,
    Tanh,
    Softplus,
    SoftSign,
    Mish,
    Exp,
    Log,
    Recip,
    Sqrt,
    Rsqrt,
    Abs,
    Square,
    Neg,
    Sign,
    StepGtZero,
}

#[non_exhaustive]
pub enum GradOp<'a> {
    Identity { input: &'a Tensor },
    Add { lhs: &'a Tensor, rhs: &'a Tensor },
    Sub { lhs: &'a Tensor, rhs: &'a Tensor },
    Mul { lhs: &'a Tensor, rhs: &'a Tensor },
    Div { lhs: &'a Tensor, rhs: &'a Tensor },
    Neg { input: &'a Tensor },
    Affine { input: &'a Tensor, mul: f32, add: f32 },
    AddScalar { input: &'a Tensor, scalar: f32 },
    MulScalar { input: &'a Tensor, scalar: f32 },
    Unary { input: &'a Tensor, kind: UnaryGradKind, alpha: Option<f32> },
    Cast { input: &'a Tensor, target_dtype: DType },
    MatMul { lhs: &'a Tensor, rhs: &'a Tensor },
    Sum { input: &'a Tensor, dims: Vec<usize>, keepdim: bool },
    Mean { input: &'a Tensor, dims: Vec<usize>, keepdim: bool },
    Max { input: &'a Tensor, dims: Vec<usize>, keepdim: bool },
    Softmax { input: &'a Tensor, dim: i32 },
    LogSoftmax { input: &'a Tensor, dim: i32 },
    Reshape { input: &'a Tensor },
    Transpose { input: &'a Tensor, dim0: usize, dim1: usize },
    Permute { input: &'a Tensor, perm: Vec<usize> },
    Expand { input: &'a Tensor },
    Narrow { input: &'a Tensor, dim: usize, start: usize, len: usize },
    Squeeze { input: &'a Tensor, dim: usize },
    Unsqueeze { input: &'a Tensor, dim: usize },
    Cat { inputs: Vec<&'a Tensor>, dim: usize },
    Stack { inputs: Vec<&'a Tensor>, dim: usize },
    Gather { input: &'a Tensor, indices: &'a Tensor, dim: usize },
    IndexSelect { input: &'a Tensor, indices: &'a Tensor, dim: usize },
    MaskedFill { input: &'a Tensor, mask: &'a Tensor, value: f32 },
    WhereCond { cond: &'a Tensor, a: &'a Tensor, b: &'a Tensor },
}

pub trait GradFnBuilder: Send + Sync {
    fn build(&self, op: GradOp<'_>, output: &Tensor) -> Option<Arc<dyn GradFn>>;
}

static GRAD_BUILDER: OnceLock<Arc<dyn GradFnBuilder>> = OnceLock::new();

pub fn set_grad_fn_builder(builder: Arc<dyn GradFnBuilder>) -> Result<()> {
    GRAD_BUILDER
        .set(builder)
        .map_err(|_| SynaptixError::Other("grad fn builder already registered".into()))
}

pub fn grad_fn_builder() -> Option<Arc<dyn GradFnBuilder>> {
    GRAD_BUILDER.get().cloned()
}

pub fn try_attach_grad_fn(op: GradOp<'_>, output: &mut Tensor) -> Result<()> {
    if !is_grad_enabled() {
        return Ok(());
    }
    if !op_has_grad_input(&op) {
        return Ok(());
    }
    let Some(builder) = grad_fn_builder() else {
        return Ok(());
    };
    let Some(grad_fn) = builder.build(op, output) else {
        return Ok(());
    };
    output.set_grad_meta(Some(GradMeta::intermediate(grad_fn)));
    Ok(())
}

fn op_has_grad_input(op: &GradOp<'_>) -> bool {
    match op {
        GradOp::Identity { input } => input.requires_grad(),
        GradOp::Add { lhs, rhs }
        | GradOp::Sub { lhs, rhs }
        | GradOp::Mul { lhs, rhs }
        | GradOp::Div { lhs, rhs }
        | GradOp::MatMul { lhs, rhs } => lhs.requires_grad() || rhs.requires_grad(),
        GradOp::Neg { input }
        | GradOp::Affine { input, .. }
        | GradOp::AddScalar { input, .. }
        | GradOp::MulScalar { input, .. }
        | GradOp::Unary { input, .. }
        | GradOp::Cast { input, .. }
        | GradOp::Sum { input, .. }
        | GradOp::Mean { input, .. }
        | GradOp::Max { input, .. }
        | GradOp::Softmax { input, .. }
        | GradOp::LogSoftmax { input, .. }
        | GradOp::Reshape { input }
        | GradOp::Transpose { input, .. }
        | GradOp::Permute { input, .. }
        | GradOp::Expand { input }
        | GradOp::Narrow { input, .. }
        | GradOp::Squeeze { input, .. }
        | GradOp::Unsqueeze { input, .. } => input.requires_grad(),
        GradOp::Cat { inputs, .. } | GradOp::Stack { inputs, .. } => {
            inputs.iter().any(|t| t.requires_grad())
        }
        GradOp::Gather { input, .. } | GradOp::IndexSelect { input, .. } => input.requires_grad(),
        GradOp::MaskedFill { input, .. } => input.requires_grad(),
        GradOp::WhereCond { a, b, .. } => a.requires_grad() || b.requires_grad(),
    }
}

pub fn backward(output: &Tensor) -> Result<()> {
    let initial = output.ones_like()?;
    run_backward(output, initial)
}

pub fn backward_with(output: &Tensor, gradient: Tensor) -> Result<()> {
    run_backward(output, gradient)
}

type MetaId = usize;

fn meta_id(meta: &Arc<GradMeta>) -> MetaId {
    Arc::as_ptr(meta) as usize
}

fn run_backward(output: &Tensor, initial: Tensor) -> Result<()> {
    let Some(out_meta) = output.grad_meta() else {
        return Err(SynaptixError::Other(
            "backward called on tensor without grad metadata".into(),
        ));
    };
    if !out_meta.requires_grad() {
        return Err(SynaptixError::Other(
            "backward called on tensor that does not require grad".into(),
        ));
    }

    let order = topological_order(output);

    let mut grads: HashMap<MetaId, Tensor> = HashMap::new();
    grads.insert(meta_id(&out_meta), initial);

    for node in order {
        let Some(meta) = node.grad_meta() else { continue };
        let mid = meta_id(&meta);
        let Some(incoming) = grads.remove(&mid) else { continue };

        let incoming = meta.run_hooks(&incoming);
        if meta.is_leaf() && meta.requires_grad() {
            meta.accumulate(incoming.clone())?;
        }

        let Some(grad_fn) = meta.grad_fn() else { continue };
        let parent_grads = grad_fn.backward(&incoming)?;
        let parents = grad_fn.parents();
        if parent_grads.len() != parents.len() {
            return Err(SynaptixError::Other(format!(
                "grad_fn `{}` returned {} grads for {} parents",
                grad_fn.name(),
                parent_grads.len(),
                parents.len()
            )));
        }
        for (parent, p_grad) in parents.iter().zip(parent_grads.into_iter()) {
            let Some(p_grad) = p_grad else { continue };
            let Some(parent_meta) = parent.grad_meta() else { continue };
            if !parent_meta.requires_grad() && parent_meta.is_leaf() {
                continue;
            }
            let pid = meta_id(&parent_meta);
            match grads.remove(&pid) {
                Some(existing) => {
                    let summed = existing.add(&p_grad)?;
                    grads.insert(pid, summed);
                }
                None => {
                    grads.insert(pid, p_grad);
                }
            }
        }
    }

    Ok(())
}

fn topological_order(output: &Tensor) -> Vec<Tensor> {
    let mut visited: std::collections::HashSet<MetaId> = std::collections::HashSet::new();
    let mut stack: Vec<(Tensor, bool)> = Vec::new();
    let mut order: Vec<Tensor> = Vec::new();
    stack.push((output.clone(), false));
    while let Some((node, processed)) = stack.pop() {
        let Some(meta) = node.grad_meta() else { continue };
        let mid = meta_id(&meta);
        if processed {
            order.push(node);
            continue;
        }
        if !visited.insert(mid) {
            continue;
        }
        stack.push((node.clone(), true));
        if let Some(grad_fn) = meta.grad_fn() {
            for parent in grad_fn.parents() {
                if let Some(pm) = parent.grad_meta() {
                    if !visited.contains(&meta_id(&pm)) {
                        stack.push((parent.clone(), false));
                    }
                }
            }
        }
    }
    order.reverse();
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_grad_disables_attach() {
        assert!(is_grad_enabled());
        let inside = no_grad(|| is_grad_enabled());
        assert!(!inside);
        assert!(is_grad_enabled());
    }

    #[test]
    fn nested_no_grad() {
        assert!(is_grad_enabled());
        no_grad(|| {
            assert!(!is_grad_enabled());
            no_grad(|| assert!(!is_grad_enabled()));
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }

    #[test]
    fn meta_leaf_construction() {
        let meta = GradMeta::leaf(true);
        assert!(meta.requires_grad());
        assert!(meta.is_leaf());
        assert!(meta.grad().is_none());
        assert!(meta.grad_fn().is_none());
    }

    #[test]
    fn meta_set_clear_requires_grad() {
        let meta = GradMeta::leaf(false);
        assert!(!meta.requires_grad());
        meta.set_requires_grad(true);
        assert!(meta.requires_grad());
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

    fn leaf_f32(numel: usize) -> Tensor {
        Tensor::ones((numel,), DType::F32, crate::device::Device::Cpu)
            .unwrap()
            .requires_grad_(true)
    }

    #[test]
    fn backward_identity_distributes_to_two_leaves() {
        let a = leaf_f32(3);
        let b = leaf_f32(3);
        let mut c = a.ones_like().unwrap();
        let grad_fn: Arc<dyn GradFn> =
            Arc::new(IdentityGradFn { parents: vec![a.clone(), b.clone()] });
        c.set_grad_meta(Some(GradMeta::intermediate(grad_fn)));

        c.backward().unwrap();

        let ga = a.grad().expect("a.grad after backward");
        let gb = b.grad().expect("b.grad after backward");
        assert_eq!(ga.shape().dims(), &[3]);
        assert_eq!(gb.shape().dims(), &[3]);
    }

    #[test]
    fn backward_skips_under_no_grad_attach() {
        let _g = NoGradGuard::new();
        assert!(!is_grad_enabled());
        let a = leaf_f32(2);
        let mut c = a.ones_like().unwrap();
        let grad_fn: Arc<dyn GradFn> = Arc::new(IdentityGradFn { parents: vec![a.clone()] });
        try_attach_grad_fn(GradOp::Neg { input: &a }, &mut c).unwrap();
        assert!(c.grad_meta().is_none(), "no_grad must skip attach");
        let _ = grad_fn;
    }

}
