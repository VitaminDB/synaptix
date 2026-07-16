use crate::backend::{BinaryOp, UnaryOp};
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::ops::{
    run_binary, run_linear_quant, run_matmul, run_quantize_mxfp8,
    run_quantize_nvfp4, run_unary,
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
    pub fn quantize_to_nvfp4(&self) -> Result<QuantWeight> { run_quantize_nvfp4(self) }
    pub fn quantize_to_mxfp8(&self) -> Result<QuantWeight> { run_quantize_mxfp8(self) }

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
