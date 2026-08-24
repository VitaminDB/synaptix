use std::f64::consts::PI;

use synaptix_core::tensor::Tensor;

use crate::{err, Result, VibeVoiceError};

fn alpha_bar_cosine(t: f64) -> f64 {
    let x = (t + 0.008) / 1.008 * PI / 2.0;
    x.cos().powi(2)
}

fn betas_for_alpha_bar(n: usize, max_beta: f64) -> Vec<f64> {
    let mut betas = Vec::with_capacity(n);
    for i in 0..n {
        let t1 = i as f64 / n as f64;
        let t2 = (i + 1) as f64 / n as f64;
        let b = 1.0 - alpha_bar_cosine(t2) / alpha_bar_cosine(t1);
        betas.push(b.min(max_beta));
    }
    betas
}

fn interp(x: f64, xp_len: usize, fp: &[f64]) -> f64 {
    if x <= 0.0 {
        return fp[0];
    }
    if x >= (xp_len - 1) as f64 {
        return fp[xp_len - 1];
    }
    let lo = x.floor() as usize;
    let hi = lo + 1;
    let frac = x - lo as f64;
    fp[lo] * (1.0 - frac) + fp[hi] * frac
}

#[derive(Clone, Copy, Debug)]
pub struct PlanStep {
    pub convert_alpha: f32,
    pub convert_sigma: f32,
    pub first_order: bool,
    pub ca: f32,
    pub cb: f32,
    pub cc: f32,
    pub r0_inv: f32,
}

pub fn apply_plan_step(
    p: &PlanStep,
    eps: &Tensor,
    sample: &Tensor,
    prev: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let m0 = sample
        .mul_scalar(p.convert_alpha)
        .and_then(|a| {
            let b = eps.mul_scalar(p.convert_sigma)?;
            a.sub(&b)
        })
        .map_err(err)?;
    let mut x = sample
        .mul_scalar(p.ca)
        .and_then(|a| {
            let b = m0.mul_scalar(p.cb)?;
            a.add(&b)
        })
        .map_err(err)?;
    if !p.first_order {
        let m1 = prev.ok_or_else(|| {
            VibeVoiceError::Inference("plan: второй порядок без предыдущего выхода".into())
        })?;
        let d1 = m0
            .sub(m1)
            .and_then(|d| d.mul_scalar(p.r0_inv))
            .map_err(err)?;
        x = x.add(&d1.mul_scalar(p.cc).map_err(err)?).map_err(err)?;
    }
    Ok((x, m0))
}

pub struct DpmSolverMultistep {
    sigmas_train: Vec<f64>,
    num_train_timesteps: usize,
    solver_order: usize,
    pub sigmas: Vec<f64>,
    pub timesteps: Vec<f32>,
    model_outputs: Vec<Option<Tensor>>,
    lower_order_nums: usize,
    step_index: usize,
}

impl DpmSolverMultistep {
    pub fn new(num_train_timesteps: usize, beta_schedule: &str) -> Result<Self> {
        if beta_schedule != "cosine" && beta_schedule != "squaredcos_cap_v2" {
            return Err(VibeVoiceError::Config(format!(
                "beta_schedule '{beta_schedule}' unsupported"
            )));
        }
        let betas = betas_for_alpha_bar(num_train_timesteps, 0.999);
        let mut acp = Vec::with_capacity(num_train_timesteps);
        let mut running = 1.0f32;
        for b in &betas {
            running *= 1.0f32 - (*b as f32);
            acp.push(running);
        }
        let sigmas_train: Vec<f64> = acp
            .iter()
            .map(|a| (((1.0f32 - a) / a).sqrt()) as f64)
            .collect();
        Ok(Self {
            sigmas_train,
            num_train_timesteps,
            solver_order: 2,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            model_outputs: vec![None; 2],
            lower_order_nums: 0,
            step_index: 0,
        })
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        let last = self.num_train_timesteps as f64;
        let n = num_inference_steps;
        let mut lin = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let v = (last - 1.0) * (i as f64) / (n as f64);
            lin.push(v.round_ties_even());
        }
        lin.reverse();
        lin.truncate(n);
        self.timesteps = lin.iter().map(|v| *v as f32).collect();

        let mut sigmas: Vec<f64> = lin
            .iter()
            .map(|t| interp(*t, self.sigmas_train.len(), &self.sigmas_train))
            .collect();
        sigmas.push(0.0);
        self.sigmas = sigmas;

        self.model_outputs = vec![None; self.solver_order];
        self.lower_order_nums = 0;
        self.step_index = 0;
    }

    fn alpha_sigma(sigma: f64) -> (f64, f64) {
        let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
        (alpha_t, sigma * alpha_t)
    }

    fn convert(&self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let (alpha_t, sigma_t) = Self::alpha_sigma(sigma);
        let a = sample.mul_scalar(alpha_t as f32).map_err(err)?;
        let b = model_output.mul_scalar(sigma_t as f32).map_err(err)?;
        a.sub(&b).map_err(err)
    }

    fn first_order(&self, m0: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let (alpha_t, sigma_t) = Self::alpha_sigma(self.sigmas[self.step_index + 1]);
        let (alpha_s, sigma_s) = Self::alpha_sigma(self.sigmas[self.step_index]);
        let lambda_t = alpha_t.ln() - sigma_t.ln();
        let lambda_s = alpha_s.ln() - sigma_s.ln();
        let h = lambda_t - lambda_s;
        let ca = sigma_t / sigma_s;
        let cb = -(alpha_t * ((-h).exp() - 1.0));
        let a = sample.mul_scalar(ca as f32).map_err(err)?;
        let b = m0.mul_scalar(cb as f32).map_err(err)?;
        a.add(&b).map_err(err)
    }

    fn second_order(&self, m0: &Tensor, m1: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let (alpha_t, sigma_t) = Self::alpha_sigma(self.sigmas[self.step_index + 1]);
        let (alpha_s0, sigma_s0) = Self::alpha_sigma(self.sigmas[self.step_index]);
        let (alpha_s1, sigma_s1) = Self::alpha_sigma(self.sigmas[self.step_index - 1]);
        let lambda_t = alpha_t.ln() - sigma_t.ln();
        let lambda_s0 = alpha_s0.ln() - sigma_s0.ln();
        let lambda_s1 = alpha_s1.ln() - sigma_s1.ln();
        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let r0 = h0 / h;
        let base = alpha_t * ((-h).exp() - 1.0);
        let ca = sigma_t / sigma_s0;
        let d1 = m0.sub(m1).and_then(|t| t.mul_scalar((1.0 / r0) as f32)).map_err(err)?;
        let a = sample.mul_scalar(ca as f32).map_err(err)?;
        let b = m0.mul_scalar((-base) as f32).map_err(err)?;
        let c = d1.mul_scalar((-0.5 * base) as f32).map_err(err)?;
        a.add(&b).and_then(|t| t.add(&c)).map_err(err)
    }

    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let last = self.timesteps.len() - 1;
        let lower_order_final = self.step_index == last;
        let converted = self.convert(model_output, sample)?;
        for i in 0..self.solver_order - 1 {
            self.model_outputs[i] = self.model_outputs[i + 1].clone();
        }
        self.model_outputs[self.solver_order - 1] = Some(converted);

        let m0 = self.model_outputs[self.solver_order - 1]
            .clone()
            .ok_or_else(|| VibeVoiceError::Inference("scheduler: missing m0".into()))?;
        let prev = if self.lower_order_nums < 1 || lower_order_final {
            self.first_order(&m0, sample)?
        } else {
            let m1 = self.model_outputs[self.solver_order - 2]
                .clone()
                .ok_or_else(|| VibeVoiceError::Inference("scheduler: missing m1".into()))?;
            self.second_order(&m0, &m1, sample)?
        };

        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        Ok(prev)
    }

    pub fn plan(&mut self, num_inference_steps: usize) -> Vec<PlanStep> {
        self.set_timesteps(num_inference_steps);
        let last = self.timesteps.len() - 1;
        let mut out = Vec::with_capacity(self.timesteps.len());
        for i in 0..=last {
            let (alpha_i, sigma_i) = Self::alpha_sigma(self.sigmas[i]);
            let (alpha_t, sigma_t) = Self::alpha_sigma(self.sigmas[i + 1]);
            let (alpha_s0, sigma_s0) = Self::alpha_sigma(self.sigmas[i]);
            let lambda_t = alpha_t.ln() - sigma_t.ln();
            let lambda_s0 = alpha_s0.ln() - sigma_s0.ln();
            let h = lambda_t - lambda_s0;
            let base = alpha_t * ((-h).exp() - 1.0);
            let first_order = i == 0 || i == last;
            let (cc, r0_inv) = if first_order {
                (0.0, 0.0)
            } else {
                let (alpha_s1, sigma_s1) = Self::alpha_sigma(self.sigmas[i - 1]);
                let lambda_s1 = alpha_s1.ln() - sigma_s1.ln();
                let h0 = lambda_s0 - lambda_s1;
                (-0.5 * base, h / h0)
            };
            out.push(PlanStep {
                convert_alpha: alpha_i as f32,
                convert_sigma: sigma_i as f32,
                first_order,
                ca: (sigma_t / sigma_s0) as f32,
                cb: (-base) as f32,
                cc: cc as f32,
                r0_inv: r0_inv as f32,
            });
        }
        out
    }

    pub fn reset(&mut self) {
        self.model_outputs = vec![None; self.solver_order];
        self.lower_order_nums = 0;
        self.step_index = 0;
    }
}
