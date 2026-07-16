use synaptix_core::tensor::Tensor;

use crate::error::Result;
use crate::ir::{IrGraph, IrOp};

/// Запускает `f(inputs)` и фиксирует факт вызова в IR: каждый input/output
/// называется по индексу (`in_0`, …, `out_0`, …) и оборачивается одним
/// `IrOp::Fused(Vec::new())` как opaque-вызовом. Полная per-op трассировка
/// требует hook'ов в `synaptix-autograd::tape`; здесь — leaf-level контракт.
pub fn trace_forward<F>(f: F, inputs: &[Tensor]) -> Result<IrGraph>
where
    F: Fn(&[Tensor]) -> Result<Vec<Tensor>>,
{
    let outputs = f(inputs)?;
    let mut g = IrGraph::new();
    for i in 0..inputs.len() {
        g.inputs.push(format!("in_{i}"));
    }
    for i in 0..outputs.len() {
        g.outputs.push(format!("out_{i}"));
    }
    g.push(IrOp::Fused(Vec::new()));
    Ok(g)
}

/// Multi-input/multi-output tracer с recording-флагом. Пользователь сам зовёт
/// `push` после каждой операции в forward (или в кастомном hook'е). Метод
/// `stop()` возвращает накопленный граф и снимает recording.
pub struct Tracer {
    pub graph: IrGraph,
    pub record: bool,
}

impl Tracer {
    pub fn new() -> Self { Self { graph: IrGraph::new(), record: false } }
    pub fn start(&mut self) { self.record = true; }
    pub fn stop(&mut self) -> IrGraph { self.record = false; std::mem::take(&mut self.graph) }
    pub fn push(&mut self, op: IrOp) {
        if self.record {
            self.graph.push(op);
        }
    }
    pub fn add_input(&mut self, name: impl Into<String>) {
        if self.record {
            self.graph.inputs.push(name.into());
        }
    }
    pub fn add_output(&mut self, name: impl Into<String>) {
        if self.record {
            self.graph.outputs.push(name.into());
        }
    }
}

impl Default for Tracer { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::device::Device;
    use synaptix_core::dtype::DType;
    use synaptix_kernels_cpu::ensure_registered;

    #[test]
    fn trace_forward_records_io_arity() {
        ensure_registered();
        let x = Tensor::ones(vec![2usize, 3], DType::F32, Device::Cpu).unwrap();
        let y = Tensor::ones(vec![2usize, 3], DType::F32, Device::Cpu).unwrap();
        let g = trace_forward(
            |inp| Ok(vec![inp[0].add(&inp[1]).map_err(|e| crate::error::CompileError::Codegen(e.to_string()))?]),
            &[x, y],
        ).unwrap();
        assert_eq!(g.inputs, vec!["in_0".to_string(), "in_1".to_string()]);
        assert_eq!(g.outputs, vec!["out_0".to_string()]);
        assert_eq!(g.num_ops(), 1);
    }

    #[test]
    fn trace_propagates_forward_error() {
        ensure_registered();
        let x = Tensor::ones(vec![1usize], DType::F32, Device::Cpu).unwrap();
        let r = trace_forward(
            |_| Err::<Vec<Tensor>, _>(crate::error::CompileError::Codegen("boom".into())),
            &[x],
        );
        assert!(r.is_err());
    }

    #[test]
    fn tracer_records_only_when_active() {
        let mut tr = Tracer::new();
        tr.push(IrOp::Add { a: "x".into(), b: "y".into(), out: "z".into() });
        assert_eq!(tr.graph.num_ops(), 0);

        tr.start();
        tr.add_input("x");
        tr.add_input("y");
        tr.push(IrOp::Add { a: "x".into(), b: "y".into(), out: "z".into() });
        tr.add_output("z");
        assert_eq!(tr.graph.num_ops(), 1);

        let g = tr.stop();
        assert_eq!(g.inputs, vec!["x".to_string(), "y".to_string()]);
        assert_eq!(g.outputs, vec!["z".to_string()]);
        assert_eq!(g.num_ops(), 1);
    }

    #[test]
    fn tracer_stop_drains_graph() {
        let mut tr = Tracer::new();
        tr.start();
        tr.push(IrOp::Relu { inp: "a".into(), out: "b".into() });
        let g1 = tr.stop();
        assert_eq!(g1.num_ops(), 1);
        let g2 = tr.stop();
        assert_eq!(g2.num_ops(), 0);
    }
}
