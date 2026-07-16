use synaptix_core::{error::Result, tensor::Tensor};

/// Flash-decoding (split-K): attention запроса по KV-кэшу с разбиением ключей на
/// чанки и online-комбинацией частичных (m, l, acc) — то самое online-softmax ядро
/// FlashAttention. Здесь `block_k=32` имитирует split-K чанки; числовой результат
/// bit-exact со стандартным attention. `q:[B,H,Sq,D]`, `k_cache,v_cache:[B,H,Sk,D]`.
pub fn flash_decode(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let scale = (*q.dims().last().unwrap_or(&64) as f32).sqrt().recip();
    super::flash_v2::flash_attn_core(q, k_cache, v_cache, scale, mask, 32)
}
