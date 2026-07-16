//! Multimodal guidance (CFG + STG + isolated-modality + rescale) — bit-faithful
//! к `ltx_core/components/guiders.py::MultiModalGuider`. Комбинирует до 4 проходов
//! DiT (cond / uncond_text / uncond_perturbed / uncond_modality) в один предикт:
//!
//! ```text
//! pred = cond
//!      + (cfg-1)·(cond - uncond_text)        # classifier-free guidance
//!      + stg·(cond - uncond_perturbed)       # spatio-temporal guidance (skip-attn блоки)
//!      + (mod-1)·(cond - uncond_modality)    # isolated modality (видео↔аудио)
//! если rescale≠0: pred *= rescale·(cond.std/pred.std) + (1-rescale)
//! ```
//!
//! Distilled-модель тренирована БЕЗ guidance (`SimpleDenoiser`); guidance нужен
//! полному (не-distilled) чекпойнту. Здесь — переиспользуемая математика + расчёт
//! числа проходов; интеграция в denoise-петлю отдельно (требует negative-context
//! и STG-пертурбации DiT).

use synaptix_core::tensor::Tensor;

use crate::LtxError;

type R<T> = std::result::Result<T, LtxError>;

/// Параметры multimodal-гайдера для одной модальности (видео/аудио). Дефолты —
/// официальные distilled (video: cfg 3, stg 1, rescale 0.7, mod 3, stg_blocks [29]).
#[derive(Clone, Debug, PartialEq)]
pub struct GuiderParams {
    /// Classifier-free guidance scale (1.0 → выкл).
    pub cfg_scale: f32,
    /// Spatio-temporal guidance scale (0.0 → выкл).
    pub stg_scale: f32,
    /// Сила rescale предикта к норме cond (0.0 → выкл).
    pub rescale_scale: f32,
    /// Isolated-modality guidance scale (1.0 → выкл).
    pub modality_scale: f32,
    /// Каждый `(skip_step+1)`-й шаг guidance выполняется (0 → каждый шаг).
    pub skip_step: u32,
    /// Индексы блоков DiT, в которых STG-проход пропускает self-attn.
    pub stg_blocks: Vec<usize>,
}

impl GuiderParams {
    /// Официальные video-дефолты distilled (constants.py PipelineParams).
    pub fn video_default() -> Self {
        Self { cfg_scale: 3.0, stg_scale: 1.0, rescale_scale: 0.7, modality_scale: 3.0, skip_step: 0, stg_blocks: vec![29] }
    }
    /// Официальные audio-дефолты distilled (cfg 7.0).
    pub fn audio_default() -> Self {
        Self { cfg_scale: 7.0, stg_scale: 1.0, rescale_scale: 0.7, modality_scale: 3.0, skip_step: 0, stg_blocks: vec![29] }
    }
    /// «Positive-only»: только cond-проход, `calculate` возвращает cond без изменений.
    pub fn positive_only() -> Self {
        Self { cfg_scale: 1.0, stg_scale: 0.0, rescale_scale: 0.0, modality_scale: 1.0, skip_step: 0, stg_blocks: Vec::new() }
    }

    /// Нужен ли uncond_text-проход (CFG).
    pub fn do_uncond(&self) -> bool {
        !approx_eq(self.cfg_scale, 1.0)
    }
    /// Нужен ли perturbed-проход (STG).
    pub fn do_perturbed(&self) -> bool {
        !approx_eq(self.stg_scale, 0.0)
    }
    /// Нужен ли isolated-modality-проход.
    pub fn do_isolated_modality(&self) -> bool {
        !approx_eq(self.modality_scale, 1.0)
    }
    /// Пропустить ли guidance на шаге `step` (skip_step-расписание).
    pub fn should_skip_step(&self, step: usize) -> bool {
        if self.skip_step == 0 {
            return false;
        }
        step % (self.skip_step as usize + 1) != 0
    }
    /// Сколько проходов DiT нужно на шаге (cond + опц. uncond/perturbed/modality).
    pub fn num_passes(&self) -> usize {
        1 + self.do_uncond() as usize + self.do_perturbed() as usize + self.do_isolated_modality() as usize
    }
}

/// Комбинировать проходы в финальный velocity/предикт (см. модульный докстринг).
/// `uncond_*` = `None` → соответствующий член зануляется (член `(scale-1)·(cond-uncond)`
/// требует, чтобы при отсутствии прохода scale был ≈нейтральным; гарантируется
/// тем, что [`GuiderParams::do_*`] и [`num_passes`] согласованы с передаваемыми проходами).
pub fn calculate(
    p: &GuiderParams,
    cond: &Tensor,
    uncond_text: Option<&Tensor>,
    uncond_perturbed: Option<&Tensor>,
    uncond_modality: Option<&Tensor>,
) -> R<Tensor> {
    let mut pred = cond.clone();
    if let Some(u) = uncond_text {
        // + (cfg-1)·(cond - u)
        pred = pred.add(&cond.sub(u)?.mul_scalar(p.cfg_scale - 1.0)?)?;
    }
    if let Some(u) = uncond_perturbed {
        // + stg·(cond - u)
        pred = pred.add(&cond.sub(u)?.mul_scalar(p.stg_scale)?)?;
    }
    if let Some(u) = uncond_modality {
        // + (mod-1)·(cond - u)
        pred = pred.add(&cond.sub(u)?.mul_scalar(p.modality_scale - 1.0)?)?;
    }
    if !approx_eq(p.rescale_scale, 0.0) {
        // factor = rescale·(cond.std/pred.std) + (1-rescale); pred *= factor
        let cs = std_all(cond)?;
        let ps = std_all(&pred)?;
        let factor = p.rescale_scale * (cs / ps) + (1.0 - p.rescale_scale);
        pred = pred.mul_scalar(factor)?;
    }
    Ok(pred)
}

/// Стандартное отклонение по всем элементам (population std, как `torch.std()` с
/// `unbiased=False`? — torch.std() по умолчанию unbiased=True; LTX использует
/// `tensor.std()` → unbiased=True, делитель N-1).
fn std_all(t: &Tensor) -> R<f32> {
    let v: Vec<f32> = t.to_dtype(synaptix_core::dtype::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let n = v.len() as f32;
    if n < 2.0 {
        return Ok(0.0);
    }
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (n - 1.0); // unbiased (torch default)
    Ok(var.sqrt())
}

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::device::Device;
    use synaptix_core::dtype::DType;

    fn reg() {
        synaptix_kernels_cpu::ensure_registered();
    }

    #[test]
    fn passes_and_flags() {
        let v = GuiderParams::video_default();
        assert!(v.do_uncond() && v.do_perturbed() && v.do_isolated_modality());
        assert_eq!(v.num_passes(), 4);
        let po = GuiderParams::positive_only();
        assert!(!po.do_uncond() && !po.do_perturbed() && !po.do_isolated_modality());
        assert_eq!(po.num_passes(), 1);
        // skip_step
        let mut s = GuiderParams::positive_only();
        s.skip_step = 1;
        assert!(!s.should_skip_step(0) && s.should_skip_step(1) && !s.should_skip_step(2));
    }

    #[test]
    fn positive_only_returns_cond() {
        reg();
        let cond = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], vec![2, 2], Device::Cpu).unwrap();
        let pred = calculate(&GuiderParams::positive_only(), &cond, None, None, None).unwrap();
        let a: Vec<f32> = pred.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cfg_only_matches_formula() {
        reg();
        // pred = cond + (cfg-1)(cond-uncond), без rescale
        let cond = Tensor::from_vec(vec![2f32, 4.0], vec![2], Device::Cpu).unwrap();
        let uncond = Tensor::from_vec(vec![1f32, 1.0], vec![2], Device::Cpu).unwrap();
        let mut p = GuiderParams::positive_only();
        p.cfg_scale = 3.0;
        let pred = calculate(&p, &cond, Some(&uncond), None, None).unwrap();
        let a: Vec<f32> = pred.flatten_all().unwrap().to_vec1().unwrap();
        // cond + 2*(cond-uncond) = [2+2*1, 4+2*3] = [4, 10]
        assert_eq!(a, vec![4.0, 10.0]);
    }

    #[test]
    fn full_combo_with_rescale_is_finite() {
        reg();
        let cond = Tensor::randn(vec![4, 8], Device::Cpu).unwrap();
        let ut = Tensor::randn(vec![4, 8], Device::Cpu).unwrap();
        let up = Tensor::randn(vec![4, 8], Device::Cpu).unwrap();
        let um = Tensor::randn(vec![4, 8], Device::Cpu).unwrap();
        let p = GuiderParams::video_default();
        let pred = calculate(&p, &cond, Some(&ut), Some(&up), Some(&um)).unwrap();
        let a: Vec<f32> = pred.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(a.iter().all(|x| x.is_finite()));
        assert_eq!(a.len(), 32);
    }
}
