use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_infer::graph_capture::GraphCapturer;

use crate::generate::guide;
use crate::model::VibeVoiceModel;
use crate::schedule::{apply_plan_step, DpmSolverMultistep};
use crate::{err, Result, VibeVoiceError};

static GRAPHS_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static GRAPHS_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();

pub fn set_graphs_enabled(on: bool) {
    GRAPHS_INIT.get_or_init(|| ());
    GRAPHS_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn graphs_enabled() -> bool {
    GRAPHS_INIT.get_or_init(|| {
        if matches!(std::env::var("SYN_VV_GRAPH").as_deref(), Ok("1")) {
            GRAPHS_ON.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    GRAPHS_ON.load(std::sync::atomic::Ordering::Relaxed)
}

fn ordinal(device: Device) -> Option<usize> {
    match device {
        Device::Cuda(o) => Some(o),
        _ => None,
    }
}

pub struct SamplerGraph {
    capturer: GraphCapturer,
    x_in: Tensor,
    cond_in: Tensor,
    out: Tensor,
    _temb: Tensor,
    steps: usize,
    cfg_bits: u32,
    batch: usize,
}

impl SamplerGraph {
    pub fn matches(&self, batch: usize, steps: usize, cfg_scale: f32) -> bool {
        self.batch == batch && self.steps == steps && self.cfg_bits == cfg_scale.to_bits()
    }

    pub fn build(
        model: &VibeVoiceModel,
        scheduler: &mut DpmSolverMultistep,
        batch: usize,
        steps: usize,
        cfg_scale: f32,
    ) -> Result<Self> {
        let device = model.device;
        let dtype = model.dtype;
        let ord = ordinal(device)
            .ok_or_else(|| VibeVoiceError::Inference("graph: нужен CUDA".into()))?;
        let vae = model.config.acoustic_vae_dim();
        let hidden = model.lm.hidden_size();

        let x_in = Tensor::zeros(vec![batch, vae], dtype, device).map_err(err)?;
        let cond_in = Tensor::zeros(vec![batch * 2, hidden], dtype, device).map_err(err)?;
        let mut out = Tensor::zeros(vec![batch, vae], dtype, device).map_err(err)?;

        let plan = scheduler.plan(steps);
        let timesteps = scheduler.timesteps.clone();
        let mut temb_rows: Vec<Tensor> = Vec::with_capacity(steps);
        for t in &timesteps {
            let ts = vec![*t; batch * 2];
            temb_rows.push(model.head.time_embeddings(&ts)?);
        }
        let temb_refs: Vec<&Tensor> = temb_rows.iter().collect();
        let temb_all = Tensor::cat(&temb_refs, 0).map_err(err)?;
        drop(temb_rows);

        let stream = synaptix_core::device::cuda::default_stream(ord)
            .map_err(|e| VibeVoiceError::Inference(format!("graph stream: {e}")))?;
        if std::env::var("SYN_VV_GRAPH_DEBUG").is_ok() {
            eprintln!("[vv-graph] build batch={batch} steps={steps} cfg={cfg_scale}");
        }
        let mut capturer = GraphCapturer::new(1);
        {
            let temb = &temb_all;
            let xi = &x_in;
            let ci = &cond_in;
            let dst = &mut out;
            let plan_ref = &plan;
            capturer
                .capture_with(&stream, |_s| {
                    let mut x = xi.clone();
                    let mut prev: Option<Tensor> = None;
                    for (i, p) in plan_ref.iter().enumerate() {
                        let combined = Tensor::cat(&[&x, &x], 0)
                            .map_err(|e| infer_err(&e.to_string()))?;
                        let step_temb = temb
                            .narrow(0, i * batch * 2, batch * 2)
                            .and_then(|t| t.contiguous())
                            .map_err(|e| infer_err(&e.to_string()))?;
                        let eps = model
                            .head
                            .forward_with_temb(&combined, &step_temb, ci)
                            .map_err(|e| infer_err(&e.to_string()))?;
                        let guided = guide(&eps, cfg_scale, batch)
                            .map_err(|e| infer_err(&e.to_string()))?;
                        let (next, m0) = apply_plan_step(p, &guided, &x, prev.as_ref())
                            .map_err(|e| infer_err(&e.to_string()))?;
                        x = next;
                        prev = Some(m0);
                    }
                    dst.copy_from(&x).map_err(|e| infer_err(&e.to_string()))?;
                    Ok(())
                })
                .map_err(|e| VibeVoiceError::Inference(format!("graph capture: {e}")))?;
        }
        if let Some(g) = capturer.graph() {
            let _ = g.upload();
        }

        Ok(Self {
            capturer,
            x_in,
            cond_in,
            out,
            _temb: temb_all,
            steps,
            cfg_bits: cfg_scale.to_bits(),
            batch,
        })
    }

    pub fn run(&mut self, init_noise: &Tensor, cond: &Tensor) -> Result<Tensor> {
        self.x_in.copy_from(init_noise).map_err(err)?;
        self.cond_in.copy_from(cond).map_err(err)?;
        let g = self
            .capturer
            .graph()
            .ok_or_else(|| VibeVoiceError::Inference("graph: не захвачен".into()))?;
        g.launch()
            .map_err(|e| VibeVoiceError::Inference(format!("graph launch: {e:?}")))?;
        Ok(self.out.clone())
    }
}

fn infer_err(msg: &str) -> synaptix_infer::InferError {
    synaptix_infer::InferError::Other(msg.to_string())
}
