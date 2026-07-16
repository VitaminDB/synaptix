pub struct ConstFold;

impl crate::passes::Pass for ConstFold {
    fn name(&self) -> &'static str { "const_fold" }
    fn apply(&self, _g: &mut crate::ir::IrGraph) -> crate::error::Result<bool> { Ok(false) }
}
