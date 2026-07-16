use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq)]
pub enum AudioLmEvent {
    AudioTokens(Vec<u32>),
    TextTokens(Vec<u32>),
    Eos,
}

pub struct StreamingAudioLm {
    frame_size: usize,
    sample_rate: u32,
    audio_buffer: VecDeque<f32>,
    pending_events: VecDeque<AudioLmEvent>,
}

impl StreamingAudioLm {
    pub fn new(sample_rate: u32, frame_size: usize) -> Self {
        Self {
            frame_size,
            sample_rate,
            audio_buffer: VecDeque::new(),
            pending_events: VecDeque::new(),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn feed_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend(samples.iter().copied());
    }

    pub fn drain_frames(&mut self) -> Vec<Vec<f32>> {
        let mut frames = Vec::new();
        while self.audio_buffer.len() >= self.frame_size {
            let frame: Vec<f32> = self.audio_buffer.drain(..self.frame_size).collect();
            frames.push(frame);
        }
        frames
    }

    pub fn push_event(&mut self, event: AudioLmEvent) {
        self.pending_events.push_back(event);
    }

    pub fn next_event(&mut self) -> Option<AudioLmEvent> {
        self.pending_events.pop_front()
    }

    pub fn pending_audio_samples(&self) -> usize {
        self.audio_buffer.len()
    }
}
