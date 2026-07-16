pub mod cuda_codegen;
pub mod ptx_codegen;

pub trait Codegen {
    fn generate(&self, graph: &crate::ir::IrGraph) -> crate::error::Result<String>;
}
