use synaptix_core::{error::Result, tensor::Tensor};

/// Expert-parallelism all-to-all обмен. На одном устройстве (single-rank) это
/// тождественная операция — реальный all-to-all это распределённый примитив,
/// перетасовывающий токены между рангами; здесь, без коммуникатора, токены уже
/// локальны, поэтому возвращаем вход без изменений.
pub fn ep_all_to_all(x: &Tensor) -> Result<Tensor> {
    Ok(x.clone())
}
