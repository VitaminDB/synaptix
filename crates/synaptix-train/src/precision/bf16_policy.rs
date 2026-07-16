pub fn should_cast_to_bf16(op: &str) -> bool {
    !matches!(op, "layer_norm" | "softmax")
}
