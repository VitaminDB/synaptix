use crate::error::{Result, VisionError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOrder {
    Hwc,
    Chw,
}

#[derive(Debug, Clone)]
pub struct RgbImage {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub order: ChannelOrder,
}

impl RgbImage {
    pub fn new_hwc(data: Vec<f32>, width: usize, height: usize, channels: usize) -> Result<Self> {
        if data.len() != width * height * channels {
            return Err(VisionError::invalid_arg(format!(
                "RgbImage: data len {} != w*h*c = {}*{}*{} = {}",
                data.len(),
                width,
                height,
                channels,
                width * height * channels
            )));
        }
        Ok(Self { data, width, height, channels, order: ChannelOrder::Hwc })
    }

    pub fn zeros_hwc(width: usize, height: usize, channels: usize) -> Self {
        Self {
            data: vec![0.0; width * height * channels],
            width,
            height,
            channels,
            order: ChannelOrder::Hwc,
        }
    }

    pub fn pixel(&self, x: usize, y: usize, c: usize) -> f32 {
        match self.order {
            ChannelOrder::Hwc => self.data[(y * self.width + x) * self.channels + c],
            ChannelOrder::Chw => self.data[(c * self.height + y) * self.width + x],
        }
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, c: usize, v: f32) {
        match self.order {
            ChannelOrder::Hwc => self.data[(y * self.width + x) * self.channels + c] = v,
            ChannelOrder::Chw => self.data[(c * self.height + y) * self.width + x] = v,
        }
    }

    pub fn to_chw(&self) -> Self {
        if self.order == ChannelOrder::Chw {
            return self.clone();
        }
        let mut out = vec![0.0f32; self.data.len()];
        for c in 0..self.channels {
            for y in 0..self.height {
                for x in 0..self.width {
                    let src = (y * self.width + x) * self.channels + c;
                    let dst = (c * self.height + y) * self.width + x;
                    out[dst] = self.data[src];
                }
            }
        }
        Self {
            data: out,
            width: self.width,
            height: self.height,
            channels: self.channels,
            order: ChannelOrder::Chw,
        }
    }

    pub fn to_hwc(&self) -> Self {
        if self.order == ChannelOrder::Hwc {
            return self.clone();
        }
        let mut out = vec![0.0f32; self.data.len()];
        for c in 0..self.channels {
            for y in 0..self.height {
                for x in 0..self.width {
                    let src = (c * self.height + y) * self.width + x;
                    let dst = (y * self.width + x) * self.channels + c;
                    out[dst] = self.data[src];
                }
            }
        }
        Self {
            data: out,
            width: self.width,
            height: self.height,
            channels: self.channels,
            order: ChannelOrder::Hwc,
        }
    }
}
