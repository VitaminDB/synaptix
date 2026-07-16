pub struct Fusion;

impl crate::passes::Pass for Fusion {
    fn name(&self) -> &'static str { "fusion" }
    fn apply(&self, _g: &mut crate::ir::IrGraph) -> crate::error::Result<bool> { Ok(false) }
}
