use synaptix_core::{error::Result, tensor::Tensor};

pub struct TreeAttnMask {
    pub draft_len: usize,
    pub target_len: usize,
}

impl TreeAttnMask {
    pub fn new(draft_len: usize, target_len: usize) -> Self {
        Self { draft_len, target_len }
    }

    pub fn build_causal(&self) -> Vec<Vec<bool>> {
        let n = self.target_len + self.draft_len;
        (0..n).map(|i| (0..n).map(|j| j <= i).collect()).collect()
    }
}

/// Tree attention для спекулятивного декодинга. Структура дерева кандидатов
/// кодируется в `tree_mask` (additive: 0 для разрешённых пар предок→потомок,
/// `-inf` для остальных), поэтому операция — это обычный masked softmax-attention
/// с этой маской. Маску строит [`TreeAttnMask`] (или вызывающий код). Scale = `1/√D`.
pub fn tree_attention(q: &Tensor, k: &Tensor, v: &Tensor, tree_mask: Option<&Tensor>) -> Result<Tensor> {
    let scale = (*q.dims().last().unwrap_or(&64) as f32).sqrt().recip();
    super::softmax::scaled_dot::scaled_dot_attention(q, k, v, scale, tree_mask)
}
