//! Адаптер LoRA: `y = W·x + s · B·(A·x)`, `s = alpha / r`. `A` — `[r, K]`,
//! `B` — `[N, r]`, оба в F32; `B` стартует нулём, так что до первого шага
//! модель совпадает с базой.

use synaptix_core::backend::train::TrainOp;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::Result;

/// Какие проекции учить и с каким рангом.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub r: usize,
    pub alpha: f32,
    /// Имена проекций как в HF: `q_proj`, `k_proj`, `v_proj`, `o_proj`,
    /// `gate_proj`, `up_proj`, `down_proj`.
    pub targets: Vec<String>,
    pub seed: u64,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 16,
            alpha: 32.0,
            targets: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            seed: 0,
        }
    }
}

impl LoraConfig {
    pub fn targets(&self, name: &str) -> bool {
        self.targets.iter().any(|t| t == name)
    }
}

pub struct LoraLayer {
    /// Путь модуля как в PEFT: `model.layers.3.self_attn.q_proj`.
    pub module: String,
    pub a: Tensor,
    pub b: Tensor,
    pub scale: f32,
    /// Накопленные градиенты (между шагами оптимизатора).
    pub ga: Option<Tensor>,
    pub gb: Option<Tensor>,
    /// Моменты AdamW.
    pub ma: Tensor,
    pub va: Tensor,
    pub mb: Tensor,
    pub vb: Tensor,
}

/// Детерминированный равномерный шум `[-bound, bound)` (splitmix64).
fn uniform(seed: u64, n: usize, bound: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let u = (z >> 40) as f32 / (1u64 << 24) as f32;
            (u * 2.0 - 1.0) * bound
        })
        .collect()
}

fn op1(op: TrainOp, ins: &[&Tensor], dims: Vec<usize>, dt: DType) -> Result<Tensor> {
    Ok(Tensor::train_op(op, ins, &[(dims, dt)])?.remove(0))
}

fn add_into(acc: &mut Option<Tensor>, g: Tensor) -> Result<()> {
    *acc = Some(match acc.take() {
        Some(a) => a.add(&g)?,
        None => g,
    });
    Ok(())
}

impl LoraLayer {
    /// Новый адаптер для проекции `[n, k]`: `A` — Kaiming-uniform как у PEFT
    /// (граница `1/√k`), `B` — нули.
    pub fn new(module: String, n: usize, k: usize, cfg: &LoraConfig, seed: u64, device: Device) -> Result<Self> {
        let r = cfg.r;
        let a = Tensor::from_vec(uniform(seed, r * k, 1.0 / (k as f32).sqrt()), vec![r, k], device)?;
        let b = Tensor::zeros(vec![n, r], DType::F32, device)?;
        Self::from_weights(module, a, b, cfg.alpha / r as f32)
    }

    pub fn from_weights(module: String, a: Tensor, b: Tensor, scale: f32) -> Result<Self> {
        let device = a.device();
        let (ad, bd) = (a.dims().to_vec(), b.dims().to_vec());
        Ok(Self {
            module,
            ma: Tensor::zeros(ad.clone(), DType::F32, device)?,
            va: Tensor::zeros(ad, DType::F32, device)?,
            mb: Tensor::zeros(bd.clone(), DType::F32, device)?,
            vb: Tensor::zeros(bd, DType::F32, device)?,
            a,
            b,
            scale,
            ga: None,
            gb: None,
        })
    }

    pub fn rank(&self) -> usize {
        self.a.dims()[0]
    }

    /// `x` — `[T, K]` (вход проекции), `y` — выход замороженной проекции
    /// `[T, N]`. Возвращает `(z = x·Aᵀ [T, r] F32, y + s·z·Bᵀ)`; `z` нужен
    /// обратному проходу. Узкие умножения — своими ядрами (упираются в
    /// память, а не в тайлы GEMM).
    pub fn forward_add(&self, x: &Tensor, y: &Tensor) -> Result<(Tensor, Tensor)> {
        let (t, r) = (x.numel() / x.dims()[x.rank() - 1], self.rank());
        let z = op1(TrainOp::LoraDown, &[x, &self.a], vec![t, r], DType::F32)?;
        let out = op1(TrainOp::LoraUpAdd { scale: self.scale }, &[y, &z, &self.b], y.dims().to_vec(), y.dtype())?;
        Ok((z, out))
    }

    /// Градиенты по `A` и `B` копятся (F32); к `dx` (`[T, K]` F32, градиент
    /// через замороженный вес) прибавляется вклад адаптера.
    pub fn backward_add(&mut self, x: &Tensor, z: &Tensor, dy: &Tensor, dx: &Tensor) -> Result<Tensor> {
        let (t, r) = (z.dims()[0], self.rank());
        let (n, k) = (self.b.dims()[0], self.a.dims()[1]);
        let bt = self.b.t()?.contiguous()?; // [r, N]
        let dz = op1(TrainOp::LoraDown, &[dy, &bt], vec![t, r], DType::F32)?.mul_scalar(self.scale)?;
        let at = self.a.t()?.contiguous()?; // [K, r]
        let dx = op1(TrainOp::LoraUpAdd { scale: 1.0 }, &[dx, &dz, &at], dx.dims().to_vec(), DType::F32)?;
        let parts = t.div_ceil(128);
        let ga = op1(TrainOp::LoraOuter, &[&dz, x], vec![parts, r, k], DType::F32)?.sum_keepdim(0)?.reshape(vec![r, k])?;
        let gbt = op1(TrainOp::LoraOuter, &[z, dy], vec![parts, r, n], DType::F32)?.sum_keepdim(0)?.reshape(vec![r, n])?;
        let gb = gbt.t()?.contiguous()?.mul_scalar(self.scale)?;
        add_into(&mut self.ga, ga)?;
        add_into(&mut self.gb, gb)?;
        Ok(dx)
    }

    pub fn zero_grad(&mut self) {
        self.ga = None;
        self.gb = None;
    }

    pub fn num_params(&self) -> usize {
        self.a.numel() + self.b.numel()
    }
}
