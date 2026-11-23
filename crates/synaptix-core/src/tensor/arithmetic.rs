use crate::backend::{BinaryOp, UnaryOp};
use crate::dtype::DType;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::ops::{
    run_binary, run_linear_quant, run_linear_quant_bias, run_matmul, run_quantize_mxfp8,
    run_quantize_ggml, run_quantize_nvfp4, run_quantize_sq, run_unary,
};
use crate::tensor::quant::QuantWeight;

impl Tensor {
    pub fn add(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Add) }
    pub fn sub(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Sub) }
    pub fn mul(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Mul) }
    pub fn div(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Div) }
    pub fn maximum(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Max) }
    pub fn minimum(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Min) }

    pub fn broadcast_add(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Add) }
    pub fn broadcast_sub(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Sub) }
    pub fn broadcast_mul(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Mul) }
    pub fn broadcast_div(&self, rhs: &Tensor) -> Result<Self> { run_binary(self, rhs, BinaryOp::Div) }

    pub fn matmul(&self, rhs: &Tensor) -> Result<Self> { run_matmul(self, rhs) }
    pub fn broadcast_matmul(&self, rhs: &Tensor) -> Result<Self> { run_matmul(self, rhs) }
    pub fn linear_quant(&self, weight: &QuantWeight) -> Result<Self> { run_linear_quant(self, weight) }
    /// `linear_quant` + `bias` `[N]` (сложение в эпилоге ядра, где backend это умеет).
    pub fn linear_quant_bias(&self, weight: &QuantWeight, bias: Option<&Tensor>) -> Result<Self> { run_linear_quant_bias(self, weight, bias) }
    pub fn quantize_to_nvfp4(&self) -> Result<QuantWeight> { run_quantize_nvfp4(self) }
    pub fn quantize_to_mxfp8(&self) -> Result<QuantWeight> { run_quantize_mxfp8(self) }
    /// Квант в SQ`bits` (1..=8 бит, супер-блоки по 256): F16/BF16 `[n, k]`.
    pub fn quantize_to_sq(&self, bits: u8) -> Result<QuantWeight> { run_quantize_sq(self, bits) }
    /// Квант в одноблобный формат ggml на устройстве (`GgmlType::Iq3S128`):
    /// F16/BF16 `[n, k]`, `imp` — важность столбцов.
    pub fn quantize_to_ggml(&self, ty: crate::quant::ggml::GgmlType, imp: Option<&[f32]>) -> Result<QuantWeight> {
        run_quantize_ggml(self, ty, imp)
    }

    /// Квантовать плотный вес `[N, K]` при загрузке в `quant` (NVFP4, MXFP8,
    /// `Sq{bits}`): так любая модель получает квантование на лету независимо
    /// от того, как сохранены её веса. `None` — вес не 2D или его форма не
    /// подходит ядрам (NVFP4: N и K кратны 64; MXFP8 и Sq: K кратен 32), для
    /// других `quant` — тоже `None`: слой остаётся плотным.
    pub fn quantize_weight(&self, quant: DType) -> Result<Option<QuantWeight>> {
        let dims = self.dims();
        if dims.len() != 2 {
            return Ok(None);
        }
        let (n, k) = (dims[0], dims[1]);
        let half = || -> Result<Tensor> {
            if matches!(self.dtype(), DType::F16 | DType::BF16) { Ok(self.clone()) } else { self.to_dtype(DType::F16) }
        };
        Ok(match quant {
            DType::NVFP4 if n % 64 == 0 && k % 64 == 0 => Some(half()?.to_dtype(DType::F16)?.quantize_to_nvfp4()?),
            DType::MXFP8 if k % 32 == 0 => Some(half()?.to_dtype(DType::F16)?.quantize_to_mxfp8()?),
            DType::Sq { bits } if k % 32 == 0 => Some(half()?.quantize_to_sq(bits)?),
            DType::Q8Row => Some(self.quantize_to_q8row()?),
            _ => None,
        })
    }

    /// Квант в q8row (int8 по выходному каналу, шкала строки по absmax) на хосте;
    /// упакованные байты и шкалы выгружаются на устройство веса как параметры
    /// модели (на NPU — статикой графа).
    pub fn quantize_to_q8row(&self) -> Result<QuantWeight> {
        use crate::backend::registry;
        use crate::tensor::storage::{CpuBuf, Storage};
        let dims = self.dims();
        if dims.len() != 2 {
            return Err(crate::error::SynaptixError::Unsupported("quantize_to_q8row: вес не 2D"));
        }
        let (n, k) = (dims[0], dims[1]);
        // CPU, плотный F32/F16/BF16: строки читаются из памяти тензора и квантуются
        // параллельно — без полной F32-копии веса (загрузка Whisper large на телефоне).
        let direct = match (self.storage().as_cpu(), self.layout().is_contiguous()) {
            (Some(buf), true) if matches!(self.dtype(), DType::F32 | DType::F16 | DType::BF16) => {
                let es = self.dtype().bytes_for_numel(1);
                let off = self.layout().offset() * es;
                let bytes = &buf.as_bytes()[off..off + n * k * es];
                Some(crate::quant::q8row::quantize_rows_with(n, k, |r, dst| {
                    let src = &bytes[r * k * es..(r + 1) * k * es];
                    match self.dtype() {
                        DType::F32 => dst.copy_from_slice(bytemuck::cast_slice(src)),
                        DType::F16 => {
                            for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                                *d = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                            }
                        }
                        _ => {
                            for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                                *d = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                            }
                        }
                    }
                }))
            }
            _ => None,
        };
        let (q, scales) = match direct {
            Some(r) => r,
            None => {
                let w = self.to_dtype(DType::F32)?.to_device(crate::device::Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
                crate::quant::q8row::quantize(&w, n, k, None)
            }
        };
        let sb = crate::quant::q8row::scales_to_bytes(&scales);
        let dev = self.device();
        let (p, s) = if dev.is_cpu() {
            (Storage::Cpu(CpuBuf::from_vec(q)), Storage::Cpu(CpuBuf::from_vec(sb)))
        } else {
            let backend = registry::backend_for(dev)?;
            crate::backend::with_param_upload(|| -> Result<_> { Ok((backend.upload(&q, dev)?, backend.upload(&sb, dev)?)) })?
        };
        QuantWeight::new(std::sync::Arc::new(p), std::sync::Arc::new(s), DType::Q8Row, n, k)
    }
    /// Квант в формат по `dtype`: NVFP4, MXFP8 или SQ; иначе `Unsupported`.
    pub fn quantize_to(&self, dtype: DType) -> Result<QuantWeight> {
        match dtype {
            DType::NVFP4 => self.quantize_to_nvfp4(),
            DType::MXFP8 => self.quantize_to_mxfp8(),
            DType::Sq { bits } => self.quantize_to_sq(bits),
            _ => Err(crate::error::SynaptixError::Unsupported("quantize_to: формат не квантованный вес (NVFP4/MXFP8/SQ)")),
        }
    }

    pub fn neg(&self) -> Result<Self> { run_unary(self, UnaryOp::Neg) }
    pub fn abs(&self) -> Result<Self> { run_unary(self, UnaryOp::Abs) }
    pub fn sqrt(&self) -> Result<Self> { run_unary(self, UnaryOp::Sqrt) }
    pub fn sqr(&self) -> Result<Self> { run_unary(self, UnaryOp::Sqr) }
    pub fn recip(&self) -> Result<Self> { run_unary(self, UnaryOp::Recip) }
    pub fn exp(&self) -> Result<Self> { run_unary(self, UnaryOp::Exp) }
    pub fn log(&self) -> Result<Self> { run_unary(self, UnaryOp::Log) }
    pub fn sin(&self) -> Result<Self> { run_unary(self, UnaryOp::Sin) }
    pub fn cos(&self) -> Result<Self> { run_unary(self, UnaryOp::Cos) }
    pub fn tanh(&self) -> Result<Self> { run_unary(self, UnaryOp::Tanh) }
    pub fn silu(&self) -> Result<Self> { run_unary(self, UnaryOp::Silu) }
    pub fn gelu_tanh(&self) -> Result<Self> { run_unary(self, UnaryOp::GeluTanh) }
    pub fn gelu_exact(&self) -> Result<Self> { run_unary(self, UnaryOp::GeluExact) }
    pub fn clamp(&self, lo: f32, hi: f32) -> Result<Self> { run_unary(self, UnaryOp::Clamp(lo, hi)) }
    pub fn powf(&self, e: f32) -> Result<Self> { run_unary(self, UnaryOp::Powf(e)) }
    pub fn affine(&self, mul: f32, add: f32) -> Result<Self> { run_unary(self, UnaryOp::Affine(mul, add)) }
    pub fn add_scalar(&self, v: f32) -> Result<Self> { self.affine(1.0, v) }
    pub fn mul_scalar(&self, v: f32) -> Result<Self> { self.affine(v, 0.0) }
    pub fn erf(&self) -> Result<Self> { run_unary(self, UnaryOp::Erf) }
    pub fn sigmoid(&self) -> Result<Self> { run_unary(self, UnaryOp::Sigmoid) }
    pub fn relu(&self) -> Result<Self> { run_unary(self, UnaryOp::Relu) }
    pub fn relu2(&self) -> Result<Self> { run_unary(self, UnaryOp::Relu2) }
    pub fn leaky_relu(&self, alpha: f32) -> Result<Self> { run_unary(self, UnaryOp::LeakyRelu(alpha)) }
    pub fn sign(&self) -> Result<Self> { run_unary(self, UnaryOp::Sign) }
    pub fn step_gt_zero(&self) -> Result<Self> { run_unary(self, UnaryOp::StepGtZero) }
    pub fn round(&self) -> Result<Self> { run_unary(self, UnaryOp::Round) }
    pub fn floor(&self) -> Result<Self> { run_unary(self, UnaryOp::Floor) }
    pub fn ceil(&self) -> Result<Self> { run_unary(self, UnaryOp::Ceil) }
}
