use synaptix_core::{
    error::{Result, SynaptixError},
    tensor::Tensor,
};

use super::conv1d::conv1d;

pub fn depthwise_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    _groups: usize,
) -> Result<Tensor> {
    // Depthwise conv1d: groups == C_in, weight [C_in, 1, K].
    if input.rank() != 3 || weight.rank() != 3 {
        return Err(SynaptixError::Unsupported(
            "depthwise_conv: input [B,C,L], weight [C,1,K]",
        ));
    }
    if weight.dims()[1] != 1 {
        return Err(SynaptixError::Unsupported(
            "depthwise_conv: weight[1] must be 1 (depthwise)",
        ));
    }
    // Быстрый путь: настоящее depthwise-ядро (CUDA, thread = выходной элемент).
    // Канальный цикл ниже = C микро-launch'ей — вокодер LTX (C до 1536) жёг
    // секунды на каждом Act1d. Unsupported (CPU) → decompose.
    match input.dwconv1d(weight, bias, stride, padding, false) {
        Ok(out) => return Ok(out),
        Err(SynaptixError::Unsupported(_)) | Err(SynaptixError::NonContiguous) => {}
        Err(e) => return Err(e),
    }
    // Decompose-фоллбэк: канал-за-каналом через conv1d.
    let c = input.dims()[1];
    let mut out_channels: Vec<Tensor> = Vec::with_capacity(c);
    for ci in 0..c {
        // narrow на канал-измерении (1) даёт non-contiguous view; conv1d
        // дальше использует Tensor::cat, который требует contiguous input
        // → принудительно копируем после narrow.
        let x_c = input.narrow(1, ci, 1)?.contiguous()?; // [B, 1, L]
        let w_c = weight.narrow(0, ci, 1)?;              // [1, 1, K] (contig: narrow по leading dim)
        let b_c = bias.map(|b| b.narrow(0, ci, 1)).transpose()?; // [1]
        let out_c = conv1d(&x_c, &w_c, b_c.as_ref(), stride, padding)?; // [B, 1, L_out]
        out_channels.push(out_c);
    }
    let refs: Vec<&Tensor> = out_channels.iter().collect();
    Tensor::cat(&refs, 1)
}
