use std::path::Path;
use ffmpeg_next as ffmpeg;
use ffmpeg_next::format::Pixel;
use ffmpeg_next::software::scaling::{Context as Scaler, Flags as SwsFlags};
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use crate::error::{IoError, Result};

pub fn init() -> Result<()> {
    ffmpeg::init().map_err(|e| IoError::Video(format!("ffmpeg init: {e}")))
}

pub struct VideoReader {
    pub width: u32,
    pub height: u32,
    pub fps_num: i32,
    pub fps_den: i32,
    frames: Vec<Vec<u8>>,
}

impl VideoReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        init()?;
        let path = path.as_ref();
        let mut ictx = ffmpeg::format::input(path)
            .map_err(|e| IoError::Video(format!("open input {:?}: {e}", path)))?;
        let stream = ictx.streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| IoError::Video("no video stream".into()))?;
        let stream_idx = stream.index();
        let tb = stream.time_base();
        let r = stream.avg_frame_rate();
        let (fps_num, fps_den) = if r.denominator() != 0 {
            (r.numerator(), r.denominator())
        } else {
            (25, 1)
        };

        let codec_ctx = ffmpeg::codec::Context::from_parameters(stream.parameters())
            .map_err(|e| IoError::Video(format!("codec context: {e}")))?;
        let mut decoder = codec_ctx.decoder().video()
            .map_err(|e| IoError::Video(format!("video decoder: {e}")))?;

        let mut scaler = Scaler::get(
            decoder.format(), decoder.width(), decoder.height(),
            Pixel::RGB24, decoder.width(), decoder.height(),
            SwsFlags::BILINEAR,
        ).map_err(|e| IoError::Video(format!("scaler: {e}")))?;

        let width = decoder.width();
        let height = decoder.height();
        let mut frames: Vec<Vec<u8>> = Vec::new();

        let packets: Vec<_> = ictx.packets().collect();
        for (stream, packet) in &packets {
            if stream.index() != stream_idx { continue; }
            decoder.send_packet(packet)
                .map_err(|e| IoError::Video(format!("send_packet: {e}")))?;
            let mut decoded = VideoFrame::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                let mut rgb = VideoFrame::new(Pixel::RGB24, width, height);
                scaler.run(&decoded, &mut rgb)
                    .map_err(|e| IoError::Video(format!("scaler.run: {e}")))?;
                let row_stride = rgb.stride(0);
                let row_w = (width * 3) as usize;
                let mut buf = vec![0u8; (height as usize) * row_w];
                for row in 0..height as usize {
                    let src = &rgb.data(0)[row * row_stride..row * row_stride + row_w];
                    buf[row * row_w..(row + 1) * row_w].copy_from_slice(src);
                }
                frames.push(buf);
            }
        }
        decoder.send_eof().ok();
        let mut decoded = VideoFrame::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut rgb = VideoFrame::new(Pixel::RGB24, width, height);
            scaler.run(&decoded, &mut rgb).ok();
            let row_stride = rgb.stride(0);
            let row_w = (width * 3) as usize;
            let mut buf = vec![0u8; (height as usize) * row_w];
            for row in 0..height as usize {
                let src = &rgb.data(0)[row * row_stride..row * row_stride + row_w];
                buf[row * row_w..(row + 1) * row_w].copy_from_slice(src);
            }
            frames.push(buf);
        }

        Ok(Self { width, height, fps_num, fps_den, frames })
    }

    pub fn num_frames(&self) -> usize { self.frames.len() }

    pub fn frame_rgb24(&self, idx: usize) -> Option<&[u8]> {
        self.frames.get(idx).map(|v| v.as_slice())
    }

    pub fn frame_tensor(&self, idx: usize, device: Device) -> Result<Tensor> {
        let rgb = self.frames.get(idx)
            .ok_or_else(|| IoError::Video(format!("frame {idx} out of range")))?;
        rgb24_to_tensor(rgb, self.height as usize, self.width as usize, device)
    }

    pub fn into_tensors(self, device: Device) -> Result<Vec<Tensor>> {
        let (h, w) = (self.height as usize, self.width as usize);
        self.frames.iter()
            .map(|rgb| rgb24_to_tensor(rgb, h, w, device))
            .collect()
    }
}

pub fn rgb24_to_tensor(rgb: &[u8], h: usize, w: usize, device: Device) -> Result<Tensor> {
    let mut chw = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            chw[y * w + x]             = rgb[src]     as f32 / 255.0;
            chw[h * w + y * w + x]     = rgb[src + 1] as f32 / 255.0;
            chw[2 * h * w + y * w + x] = rgb[src + 2] as f32 / 255.0;
        }
    }
    let raw: Vec<u8> = chw.iter().flat_map(|f| f.to_le_bytes()).collect();
    Tensor::from_raw_bytes(raw, vec![3, h, w], DType::F32, device)
        .map_err(IoError::Core)
}

pub fn tensor_to_rgb24(tensor: &Tensor) -> Result<Vec<u8>> {
    let dims = tensor.dims();
    if dims.len() != 3 || dims[0] != 3 {
        return Err(IoError::Video(format!("expected [3, H, W], got {:?}", dims)));
    }
    let (h, w) = (dims[1], dims[2]);
    let flat = tensor.flatten_all().map_err(IoError::Core)?
        .to_vec1::<f32>().map_err(IoError::Core)?;
    let mut out = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let dst = (y * w + x) * 3;
            out[dst]     = (flat[y * w + x]             .clamp(0.0, 1.0) * 255.0) as u8;
            out[dst + 1] = (flat[h * w + y * w + x]     .clamp(0.0, 1.0) * 255.0) as u8;
            out[dst + 2] = (flat[2 * h * w + y * w + x] .clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    Ok(out)
}
