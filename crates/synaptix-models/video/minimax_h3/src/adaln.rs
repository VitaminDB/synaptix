use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};

use crate::config::{
    ADALN_CHUNKS, ADALN_MODALITIES, AUDIO_COND_TIMESTEP, FINAL_ADALN_CHUNKS,
    VISUAL_COND_TIMESTEP,
};
use crate::layout::{PackedLayout, SegmentKind};
use crate::scheduler::H3Scheduler;
use crate::H3Error;

type R<T> = Result<T, SynaptixError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeRole {
    Base,
    Audio,
    VisualCond,
    AudioCond,
}

#[derive(Debug, Clone)]
pub struct RoleTable {
    pub roles: Vec<TimeRole>,
}

impl RoleTable {
    pub fn for_layout(layout: &PackedLayout) -> Self {
        let mut roles = vec![TimeRole::Base, TimeRole::Audio];
        if layout.has_visual_cond() {
            roles.push(TimeRole::VisualCond);
        }
        if layout.has_audio_cond() {
            roles.push(TimeRole::AudioCond);
        }
        Self { roles }
    }

    pub fn len(&self) -> usize {
        self.roles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roles.is_empty()
    }

    pub fn index(&self, role: TimeRole) -> usize {
        self.roles.iter().position(|r| *r == role).unwrap_or(0)
    }

    pub fn role_for(&self, kind: SegmentKind) -> TimeRole {
        match kind {
            SegmentKind::Text | SegmentKind::Video => TimeRole::Base,
            SegmentKind::Audio => TimeRole::Audio,
            SegmentKind::Cond | SegmentKind::RefImg => {
                if self.roles.contains(&TimeRole::VisualCond) {
                    TimeRole::VisualCond
                } else {
                    TimeRole::Base
                }
            }
            SegmentKind::RefAudio => {
                if self.roles.contains(&TimeRole::AudioCond) {
                    TimeRole::AudioCond
                } else {
                    TimeRole::Audio
                }
            }
        }
    }

    pub fn timesteps(
        &self,
        sched: &H3Scheduler,
        step: usize,
        visual_cond_aug: f32,
        audio_cond_aug: f32,
    ) -> Vec<f32> {
        let t_v = sched.video_t(step) as f32;
        let t_a = sched.audio_t(step) as f32;
        self.roles
            .iter()
            .map(|r| match r {
                TimeRole::Base => t_v,
                TimeRole::Audio => t_a,
                TimeRole::VisualCond => t_v.max(visual_cond_aug),
                TimeRole::AudioCond => t_a.max(audio_cond_aug),
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ModSegment {
    pub start: usize,
    pub stop: usize,
    pub row: usize,
}

pub fn mod_segments(
    layout: &PackedLayout,
    roles: &RoleTable,
    text_tags: Option<&[u8]>,
) -> Vec<ModSegment> {
    let mut out = Vec::with_capacity(layout.segments.len() + 8);
    for seg in &layout.segments {
        if seg.is_empty() {
            continue;
        }
        let role_base = roles.index(roles.role_for(seg.kind)) * ADALN_MODALITIES;
        if seg.kind == SegmentKind::Text {
            if let Some(tags) = text_tags {
                let n = seg.len();
                let tags = &tags[..n.min(tags.len())];
                let mut run_start = 0usize;
                for i in 1..=tags.len() {
                    if i == tags.len() || tags[i] != tags[run_start] {
                        out.push(ModSegment {
                            start: seg.start + run_start,
                            stop: seg.start + i,
                            row: role_base + tags[run_start] as usize,
                        });
                        run_start = i;
                    }
                }
                if tags.len() < n {
                    out.push(ModSegment {
                        start: seg.start + tags.len(),
                        stop: seg.stop,
                        row: role_base + crate::config::MODALITY_TEXT,
                    });
                }
                continue;
            }
        }
        out.push(ModSegment {
            start: seg.start,
            stop: seg.stop,
            row: role_base + seg.kind.modality_tag(),
        });
    }
    out
}

pub fn timestep_sinusoidal(vals: &[f32], dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut emb = vec![0f32; vals.len() * dim];
    for (n, &t) in vals.iter().enumerate() {
        for i in 0..half {
            let freq = (-(10000f32.ln()) * (i as f32) / (half as f32)).exp();
            let ang = t * freq;
            emb[n * dim + i] = ang.cos();
            emb[n * dim + half + i] = ang.sin();
        }
    }
    emb
}

pub struct TimeEmbedder {
    proj_in_w: Tensor,
    proj_in_b: Tensor,
    proj_out_w: Tensor,
    proj_out_b: Tensor,
    freq_dim: usize,
}

impl TimeEmbedder {
    pub fn new(
        proj_in_w: Tensor,
        proj_in_b: Tensor,
        proj_out_w: Tensor,
        proj_out_b: Tensor,
        freq_dim: usize,
    ) -> Self {
        Self { proj_in_w, proj_in_b, proj_out_w, proj_out_b, freq_dim }
    }

    pub fn forward(&self, vals: &[f32], device: Device) -> R<Tensor> {
        let emb = timestep_sinusoidal(vals, self.freq_dim);
        let x = Tensor::from_vec(emb, vec![vals.len(), self.freq_dim], device)?;
        let h = x
            .matmul(&self.proj_in_w.transpose(0, 1)?.contiguous()?)?
            .broadcast_add(&self.proj_in_b)?;
        let h = h.silu()?;
        h.matmul(&self.proj_out_w.transpose(0, 1)?.contiguous()?)?
            .broadcast_add(&self.proj_out_b)
    }
}

pub struct AdalnProj {
    w: Tensor,
    b: Tensor,
    expand: usize,
    modalities: usize,
    hidden: usize,
}

impl AdalnProj {
    pub fn new(w: Tensor, b: Tensor, expand: usize, modalities: usize, hidden: usize) -> Self {
        Self { w, b, expand, modalities, hidden }
    }

    pub fn expand(&self) -> usize {
        self.expand
    }

    pub fn modalities(&self) -> usize {
        self.modalities
    }

    pub fn forward(&self, t_emb: &Tensor) -> R<Tensor> {
        let m = t_emb.dims()[0];
        let x = t_emb.silu()?;
        let x = if x.dtype() == self.w.dtype() { x } else { x.to_dtype(self.w.dtype())? };
        let y = x
            .matmul(&self.w.transpose(0, 1)?.contiguous()?)?
            .broadcast_add(&self.b)?;
        y.reshape(vec![m * self.modalities, self.expand, self.hidden])
    }
}

pub struct AdalnCache {
    blocks: Vec<Tensor>,
    final_layer: Tensor,
    pub rows: usize,
    pub final_rows: usize,
    pub hidden: usize,
    pub steps: usize,
}

impl AdalnCache {
    pub fn new(
        blocks: Vec<Tensor>,
        final_layer: Tensor,
        rows: usize,
        final_rows: usize,
        hidden: usize,
        steps: usize,
    ) -> Self {
        Self { blocks, final_layer, rows, final_rows, hidden, steps }
    }

    pub fn block_bytes(&self) -> usize {
        self.blocks
            .iter()
            .map(|t| t.dtype().bytes_for_numel(t.dims().iter().product::<usize>()))
            .sum::<usize>()
            + self
                .final_layer
                .dtype()
                .bytes_for_numel(self.final_layer.dims().iter().product::<usize>())
    }

    pub fn chunk(&self, block: usize, step: usize, chunk: usize) -> R<Tensor> {
        self.blocks[block]
            .narrow(0, step, 1)?
            .narrow(1, chunk, 1)?
            .contiguous()?
            .reshape(vec![self.rows, self.hidden])
    }

    pub fn row(&self, block: usize, step: usize, chunk: usize, row: usize) -> R<Tensor> {
        self.blocks[block]
            .narrow(0, step, 1)?
            .narrow(1, chunk, 1)?
            .narrow(2, row, 1)?
            .contiguous()?
            .reshape(vec![1, self.hidden])
    }

    pub fn final_chunk(&self, step: usize, chunk: usize) -> R<Tensor> {
        self.final_layer
            .narrow(0, step, 1)?
            .narrow(1, chunk, 1)?
            .contiguous()?
            .reshape(vec![self.final_rows, self.hidden])
    }

    pub fn final_row(&self, step: usize, chunk: usize, row: usize) -> R<Tensor> {
        self.final_layer
            .narrow(0, step, 1)?
            .narrow(2, row, 1)?
            .narrow(1, chunk, 1)?
            .contiguous()?
            .reshape(vec![1, self.hidden])
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

pub struct AdalnPlan {
    pub roles: RoleTable,
    pub timesteps: Vec<Vec<f32>>,
    pub visual_cond_aug: f32,
    pub audio_cond_aug: f32,
}

impl AdalnPlan {
    pub fn build(
        layout: &PackedLayout,
        sched: &H3Scheduler,
        visual_cond_aug: Option<f32>,
        audio_cond_aug: Option<f32>,
    ) -> Self {
        let roles = RoleTable::for_layout(layout);
        let vis = visual_cond_aug.unwrap_or(VISUAL_COND_TIMESTEP);
        let aud = audio_cond_aug.unwrap_or(AUDIO_COND_TIMESTEP);
        let timesteps = (0..sched.steps())
            .map(|s| roles.timesteps(sched, s, vis, aud))
            .collect();
        Self { roles, timesteps, visual_cond_aug: vis, audio_cond_aug: aud }
    }

    pub fn steps(&self) -> usize {
        self.timesteps.len()
    }

    pub fn rows(&self) -> usize {
        self.roles.len() * ADALN_MODALITIES
    }

    pub fn final_rows(&self) -> usize {
        self.roles.len()
    }

    pub fn estimated_bytes(&self, num_blocks: usize, hidden: usize, dtype: DType) -> usize {
        let per_block = self.steps() * ADALN_CHUNKS * self.rows() * hidden;
        let final_layer = self.steps() * FINAL_ADALN_CHUNKS * self.final_rows() * hidden;
        dtype.bytes_for_numel(per_block * num_blocks + final_layer)
    }
}

pub fn stack_steps(per_step: Vec<Tensor>, dtype: DType) -> Result<Tensor, H3Error> {
    let refs: Vec<&Tensor> = per_step.iter().collect();
    let t = Tensor::stack(&refs, 0)?;
    Ok(if t.dtype() == dtype { t } else { t.to_dtype(dtype)? })
}
