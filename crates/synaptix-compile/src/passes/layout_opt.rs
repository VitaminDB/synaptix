pub struct LayoutOpt;

impl crate::passes::Pass for LayoutOpt {
    fn name(&self) -> &'static str { "layout_opt" }
    fn apply(&self, _g: &mut crate::ir::IrGraph) -> crate::error::Result<bool> { Ok(false) }
}
