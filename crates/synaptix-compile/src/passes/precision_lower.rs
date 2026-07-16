pub struct PrecisionLower;

impl crate::passes::Pass for PrecisionLower {
    fn name(&self) -> &'static str { "precision_lower" }
    fn apply(&self, _g: &mut crate::ir::IrGraph) -> crate::error::Result<bool> { Ok(false) }
}
