pub struct MemoryPlanner;

impl crate::passes::Pass for MemoryPlanner {
    fn name(&self) -> &'static str { "memory_planner" }
    fn apply(&self, _g: &mut crate::ir::IrGraph) -> crate::error::Result<bool> { Ok(false) }
}
