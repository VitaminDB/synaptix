use crate::rope::{audio_positions, text_positions, video_positions, video_t_span_total, FrameGrid, FRAME_RESCALE};
use crate::H3Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    Text,
    Cond,
    RefImg,
    RefAudio,
    Audio,
    Video,
}

impl SegmentKind {
    pub fn modality_tag(self) -> usize {
        match self {
            SegmentKind::Text => crate::config::MODALITY_TEXT,
            SegmentKind::Video | SegmentKind::Cond | SegmentKind::RefImg => {
                crate::config::MODALITY_VIDEO
            }
            SegmentKind::Audio | SegmentKind::RefAudio => crate::config::MODALITY_AUDIO,
        }
    }

    pub fn is_video_stream(self) -> bool {
        matches!(self, SegmentKind::Cond | SegmentKind::RefImg | SegmentKind::Video)
    }

    pub fn is_audio_stream(self) -> bool {
        matches!(self, SegmentKind::RefAudio | SegmentKind::Audio)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub start: usize,
    pub stop: usize,
    pub kind: SegmentKind,
}

impl Segment {
    pub fn len(&self) -> usize {
        self.stop - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.stop == self.start
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Keyframe {
    pub resolved_frame_index: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum RefBlock {
    Image { latent_h: usize, latent_w: usize },
    Audio { latent_t: usize },
    Video { latent_t: usize, latent_h: usize, latent_w: usize, audio_latent_t: usize },
}

#[derive(Debug, Clone)]
pub struct LayoutRequest {
    pub text_len: usize,
    pub latent_t: usize,
    pub latent_h: usize,
    pub latent_w: usize,
    pub audio_t: usize,
    pub frame_count: Option<usize>,
    pub keyframes: Vec<Keyframe>,
    pub refs: Vec<RefBlock>,
}

impl LayoutRequest {
    pub fn new(
        text_len: usize,
        latent_t: usize,
        latent_h: usize,
        latent_w: usize,
        audio_t: usize,
    ) -> Self {
        Self {
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            frame_count: None,
            keyframes: Vec::new(),
            refs: Vec::new(),
        }
    }

    pub fn with_frame_count(mut self, frames: usize) -> Self {
        self.frame_count = Some(frames);
        self
    }

    pub fn with_keyframes(mut self, kf: Vec<Keyframe>) -> Self {
        self.keyframes = kf;
        self
    }

    pub fn with_refs(mut self, refs: Vec<RefBlock>) -> Self {
        self.refs = refs;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LayoutSignature {
    pub text_len: usize,
    pub latent_t: usize,
    pub latent_h: usize,
    pub latent_w: usize,
    pub audio_t: usize,
    pub keyframes: Vec<usize>,
    pub refs: Vec<(u8, usize, usize, usize)>,
}

pub struct PackedLayout {
    pub seq_len: usize,
    pub positions: Vec<[f64; 3]>,
    pub segments: Vec<Segment>,
    pub img_rows: Vec<usize>,
    pub img_update: Vec<bool>,
    pub audio_rows: Vec<usize>,
    pub audio_update: Vec<bool>,
    pub frame_rows: usize,
    pub signature: LayoutSignature,
}

impl PackedLayout {
    pub fn build(req: &LayoutRequest) -> Result<Self, H3Error> {
        let frame = FrameGrid::new(req.latent_h, req.latent_w);
        let frame_rows = frame.len();
        if frame_rows == 0 {
            return Err(H3Error::Layout("пустая пространственная сетка".into()));
        }
        let (target_w_low, target_w_high) = frame.w_bounds();

        let mut segments: Vec<Segment> = Vec::new();
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let mut img_rows: Vec<usize> = Vec::new();
        let mut img_update: Vec<bool> = Vec::new();
        let mut audio_rows: Vec<usize> = Vec::new();
        let mut audio_update: Vec<bool> = Vec::new();
        let mut row = 0usize;

        let push = |kind: SegmentKind,
                        pos: Vec<[f64; 3]>,
                        segments: &mut Vec<Segment>,
                        positions: &mut Vec<[f64; 3]>,
                        row: &mut usize| {
            let n = pos.len();
            segments.push(Segment { start: *row, stop: *row + n, kind });
            positions.extend(pos);
            let start = *row;
            *row += n;
            (start, n)
        };

        push(
            SegmentKind::Text,
            text_positions(req.text_len),
            &mut segments,
            &mut positions,
            &mut row,
        );

        let mut cursor = req.text_len as f64;

        for kf in &req.keyframes {
            let cond_t = if kf.resolved_frame_index == 0 {
                req.text_len as f64
            } else if req.frame_count == Some(kf.resolved_frame_index + 1) {
                req.text_len as f64 + video_t_span_total(req.latent_t) - FRAME_RESCALE
            } else {
                return Err(H3Error::Layout(format!(
                    "keyframe поддержан только на первом/последнем кадре, получен индекс {}",
                    kf.resolved_frame_index
                )));
            };
            let pos: Vec<[f64; 3]> =
                frame.rows.iter().map(|hw| [cond_t, hw[0], hw[1]]).collect();
            let (start, n) =
                push(SegmentKind::Cond, pos, &mut segments, &mut positions, &mut row);
            img_rows.extend(start..start + n);
            img_update.extend(std::iter::repeat_n(false, n));
        }

        for blk in &req.refs {
            match *blk {
                RefBlock::Image { latent_h, latent_w } => {
                    let rf = FrameGrid::new(latent_h, latent_w);
                    let pos: Vec<[f64; 3]> =
                        rf.rows.iter().map(|hw| [cursor, hw[0], hw[1]]).collect();
                    let (start, n) =
                        push(SegmentKind::RefImg, pos, &mut segments, &mut positions, &mut row);
                    img_rows.extend(start..start + n);
                    img_update.extend(std::iter::repeat_n(false, n));
                    cursor += 1.0;
                }
                RefBlock::Audio { latent_t } => {
                    if latent_t > 0 {
                        let pos =
                            audio_positions(cursor, latent_t, target_w_low, target_w_high);
                        let (start, n) = push(
                            SegmentKind::RefAudio,
                            pos,
                            &mut segments,
                            &mut positions,
                            &mut row,
                        );
                        audio_rows.extend(start..start + n);
                        audio_update.extend(std::iter::repeat_n(false, n));
                    }
                    cursor += latent_t as f64;
                }
                RefBlock::Video { latent_t, latent_h, latent_w, audio_latent_t } => {
                    let rf = FrameGrid::new(latent_h, latent_w);
                    let (rw_low, rw_high) = rf.w_bounds();
                    if audio_latent_t > 0 {
                        let pos = audio_positions(cursor, audio_latent_t, rw_low, rw_high);
                        let (start, n) = push(
                            SegmentKind::RefAudio,
                            pos,
                            &mut segments,
                            &mut positions,
                            &mut row,
                        );
                        audio_rows.extend(start..start + n);
                        audio_update.extend(std::iter::repeat_n(false, n));
                    }
                    let pos = video_positions(latent_t, &rf, cursor);
                    let (start, n) =
                        push(SegmentKind::RefImg, pos, &mut segments, &mut positions, &mut row);
                    img_rows.extend(start..start + n);
                    img_update.extend(std::iter::repeat_n(false, n));
                    cursor += (audio_latent_t as f64).max(video_t_span_total(latent_t));
                }
            }
        }

        let pos = audio_positions(cursor, req.audio_t, target_w_low, target_w_high);
        let (start, n) =
            push(SegmentKind::Audio, pos, &mut segments, &mut positions, &mut row);
        audio_rows.extend(start..start + n);
        audio_update.extend(std::iter::repeat_n(true, n));

        let pos = video_positions(req.latent_t, &frame, cursor);
        let (start, n) =
            push(SegmentKind::Video, pos, &mut segments, &mut positions, &mut row);
        img_rows.extend(start..start + n);
        img_update.extend(std::iter::repeat_n(true, n));

        let signature = LayoutSignature {
            text_len: req.text_len,
            latent_t: req.latent_t,
            latent_h: req.latent_h,
            latent_w: req.latent_w,
            audio_t: req.audio_t,
            keyframes: req.keyframes.iter().map(|k| k.resolved_frame_index).collect(),
            refs: req
                .refs
                .iter()
                .map(|r| match *r {
                    RefBlock::Image { latent_h, latent_w } => (0u8, 0, latent_h, latent_w),
                    RefBlock::Audio { latent_t } => (1u8, latent_t, 0, 0),
                    RefBlock::Video { latent_t, latent_h, latent_w, audio_latent_t } => {
                        (2u8 + audio_latent_t.min(1) as u8, latent_t, latent_h, latent_w)
                    }
                })
                .collect(),
        };

        Ok(Self {
            seq_len: row,
            positions,
            segments,
            img_rows,
            img_update,
            audio_rows,
            audio_update,
            frame_rows,
            signature,
        })
    }

    pub fn segment(&self, kind: SegmentKind) -> Option<Segment> {
        self.segments.iter().rev().find(|s| s.kind == kind).copied()
    }

    pub fn has_visual_cond(&self) -> bool {
        self.segments
            .iter()
            .any(|s| matches!(s.kind, SegmentKind::Cond | SegmentKind::RefImg))
    }

    pub fn has_audio_cond(&self) -> bool {
        self.segments.iter().any(|s| s.kind == SegmentKind::RefAudio)
    }

    pub fn video_rows(&self) -> usize {
        self.img_rows.len()
    }

    pub fn audio_row_count(&self) -> usize {
        self.audio_rows.len()
    }

    pub fn target_video_rows(&self) -> usize {
        self.img_update.iter().filter(|u| **u).count()
    }

    pub fn target_audio_rows(&self) -> usize {
        self.audio_update.iter().filter(|u| **u).count()
    }
}
