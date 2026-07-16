pub struct Dce;

impl crate::passes::Pass for Dce {
    fn name(&self) -> &'static str { "dce" }
    fn apply(&self, _g: &mut crate::ir::IrGraph) -> crate::error::Result<bool> { Ok(false) }
}
