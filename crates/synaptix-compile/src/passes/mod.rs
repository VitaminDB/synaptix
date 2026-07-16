pub mod const_fold;
pub mod dce;
pub mod fusion;
pub mod layout_opt;
pub mod memory_planner;
pub mod precision_lower;

pub trait Pass {
    fn name(&self) -> &'static str;
    fn apply(&self, graph: &mut crate::ir::IrGraph) -> crate::error::Result<bool>;
}

pub fn run_passes(graph: &mut crate::ir::IrGraph, passes: &[&dyn Pass]) -> crate::error::Result<()> {
    for p in passes { p.apply(graph)?; }
    Ok(())
}
