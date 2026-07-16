#[derive(Debug, Clone)]
pub enum IrOp {
    MatMul { a: String, b: String, out: String },
    Add { a: String, b: String, out: String },
    Relu { inp: String, out: String },
    Softmax { inp: String, dim: usize, out: String },
    LayerNorm { inp: String, weight: String, bias: String, out: String },
    Fused(Vec<IrOp>),
}

#[derive(Debug, Clone)]
pub struct IrGraph {
    pub ops: Vec<IrOp>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl IrGraph {
    pub fn new() -> Self { Self { ops: Vec::new(), inputs: Vec::new(), outputs: Vec::new() } }
    pub fn push(&mut self, op: IrOp) { self.ops.push(op); }
    pub fn num_ops(&self) -> usize { self.ops.len() }
}

impl Default for IrGraph { fn default() -> Self { Self::new() } }
