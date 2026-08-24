use synaptix_core::tensor::Tensor;
use synaptix_ops::conv::conv1d::conv1d_dilated;
use synaptix_ops::conv::conv_transpose1d::conv_transpose1d;
use synaptix_ops::conv::depthwise::depthwise_conv;

use crate::{err, Result};

#[derive(Default)]
pub struct StreamingCache {
    states: Vec<Option<Tensor>>,
    reallocated: bool,
}

impl StreamingCache {
    pub fn new(slots: usize) -> Self {
        Self {
            states: vec![None; slots],
            reallocated: false,
        }
    }

    pub fn begin_pass(&mut self) {
        self.reallocated = false;
    }

    pub fn was_stable(&self) -> bool {
        !self.reallocated
    }

    pub fn get(&self, id: usize) -> Option<&Tensor> {
        self.states.get(id).and_then(|s| s.as_ref())
    }

    pub fn store(&mut self, id: usize, state: &Tensor) -> Result<()> {
        if id >= self.states.len() {
            self.states.resize(id + 1, None);
        }
        let reusable = self.states[id]
            .as_ref()
            .is_some_and(|b| b.dims() == state.dims() && b.dtype() == state.dtype());
        if reusable {
            let buf = self.states[id].as_mut().expect("slot");
            return buf.copy_from(state).map_err(err);
        }
        self.reallocated = true;
        self.states[id] = Some(state.contiguous().map_err(err)?);
        Ok(())
    }

    pub fn zero_all(&mut self) -> Result<()> {
        for slot in self.states.iter_mut() {
            if let Some(t) = slot {
                let z = Tensor::zeros(t.dims().to_vec(), t.dtype(), t.device()).map_err(err)?;
                t.copy_from(&z).map_err(err)?;
            }
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        for slot in self.states.iter_mut() {
            *slot = None;
        }
    }
}

pub struct ConvIds {
    next: usize,
}

impl ConvIds {
    pub fn new() -> Self {
        Self { next: 0 }
    }

    pub fn take(&mut self) -> usize {
        let id = self.next;
        self.next += 1;
        id
    }

    pub fn count(&self) -> usize {
        self.next
    }
}

impl Default for ConvIds {
    fn default() -> Self {
        Self::new()
    }
}

fn pad_zeros(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let d = x.dims().to_vec();
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if left > 0 {
        parts.push(Tensor::zeros(vec![d[0], d[1], left], x.dtype(), x.device()).map_err(err)?);
    }
    parts.push(if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous().map_err(err)?
    });
    if right > 0 {
        parts.push(Tensor::zeros(vec![d[0], d[1], right], x.dtype(), x.device()).map_err(err)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2).map_err(err)
}

fn tail(x: &Tensor, n: usize) -> Result<Tensor> {
    let len = x.dims()[2];
    if n >= len {
        return x.contiguous().map_err(err);
    }
    x.narrow(2, len - n, n).and_then(|t| t.contiguous()).map_err(err)
}

pub struct SConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    causal: bool,
    padding_total: usize,
    context_size: usize,
    id: usize,
}

impl SConv1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        dilation: usize,
        groups: usize,
        causal: bool,
        id: usize,
    ) -> Self {
        let kernel_size = weight.dims()[2];
        let span = (kernel_size - 1) * dilation;
        let padding_total = span.saturating_sub(stride - 1);
        Self {
            weight,
            bias,
            kernel_size,
            stride,
            dilation,
            groups,
            causal,
            padding_total,
            context_size: padding_total,
            id,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        if self.groups > 1 {
            depthwise_conv(x, &self.weight, self.bias.as_ref(), self.stride, 0, self.groups)
                .map_err(err)
        } else {
            conv1d_dilated(x, &self.weight, self.bias.as_ref(), self.stride, 0, self.dilation)
                .map_err(err)
        }
    }

    fn extra_padding(&self, len: usize) -> usize {
        let k = self.kernel_size as f64;
        let s = self.stride as f64;
        let pt = self.padding_total as f64;
        let l = len as f64;
        let n_frames = (l - k + pt) / s + 1.0;
        let ideal = (n_frames.ceil() - 1.0) * s + (k - pt);
        let extra = ideal - l;
        if extra <= 0.0 {
            0
        } else {
            extra.round() as usize
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let extra = self.extra_padding(x.dims()[2]);
        let padded = if self.causal {
            pad_zeros(x, self.padding_total, extra)?
        } else {
            let right = self.padding_total / 2;
            let left = self.padding_total - right;
            pad_zeros(x, left, right + extra)?
        };
        self.apply(&padded)
    }

    pub fn forward_streaming(&self, x: &Tensor, cache: &mut StreamingCache) -> Result<Tensor> {
        let d = x.dims().to_vec();
        let cached = match cache.get(self.id) {
            Some(t) => t.clone(),
            None => Tensor::zeros(
                vec![d[0], d[1], self.context_size],
                x.dtype(),
                x.device(),
            )
            .map_err(err)?,
        };
        let combined = if cached.dims()[2] > 0 {
            let xc = if x.is_contiguous() {
                x.clone()
            } else {
                x.contiguous().map_err(err)?
            };
            Tensor::cat(&[&cached, &xc], 2).map_err(err)?
        } else if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous().map_err(err)?
        };
        drop(cached);
        let out = self.apply(&combined)?;
        if self.context_size > 0 {
            let next = tail(&combined, self.context_size)?;
            cache.store(self.id, &next)?;
        }
        Ok(out)
    }
}

pub struct SConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    causal: bool,
    padding_total: usize,
    context_size: usize,
    id: usize,
}

impl SConvTranspose1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        causal: bool,
        id: usize,
    ) -> Self {
        let kernel_size = weight.dims()[2];
        Self {
            weight,
            bias,
            stride,
            causal,
            padding_total: kernel_size - stride,
            context_size: kernel_size - 1,
            id,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        conv_transpose1d(x, &self.weight, self.bias.as_ref(), self.stride, 0, 0, 1, 1).map_err(err)
    }

    fn trim(&self, y: &Tensor) -> Result<Tensor> {
        let (left, right) = if self.causal {
            (0usize, self.padding_total)
        } else {
            let r = self.padding_total / 2;
            (self.padding_total - r, r)
        };
        if left + right == 0 {
            return Ok(y.clone());
        }
        let len = y.dims()[2];
        y.narrow(2, left, len - left - right)
            .and_then(|t| t.contiguous())
            .map_err(err)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.apply(x)?;
        self.trim(&y)
    }

    pub fn forward_streaming(&self, x: &Tensor, cache: &mut StreamingCache) -> Result<Tensor> {
        let d = x.dims().to_vec();
        let xc = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous().map_err(err)?
        };
        let cached = cache.get(self.id).cloned();
        let had_cache = cached.as_ref().map(|c| c.dims()[2] > 0).unwrap_or(false);
        let combined = match cached {
            Some(c) if c.dims()[2] > 0 => {
                let joined = Tensor::cat(&[&c, &xc], 2).map_err(err)?;
                drop(c);
                joined
            }
            _ => xc,
        };
        let full = self.apply(&combined)?;
        let full = self.trim(&full)?;
        let out = if !had_cache {
            full
        } else {
            let expected = d[2] * self.stride;
            let len = full.dims()[2];
            if len >= expected {
                full.narrow(2, len - expected, expected)
                    .and_then(|t| t.contiguous())
                    .map_err(err)?
            } else {
                full
            }
        };
        let next = tail(&combined, self.context_size)?;
        cache.store(self.id, &next)?;
        Ok(out)
    }
}
