use std::path::Path;
use ffmpeg_next as ffmpeg;
use ffmpeg_next::codec::{self, Context as CodecCtx};
use ffmpeg_next::format::{self, Pixel};
use ffmpeg_next::software::scaling::{Context as Scaler, Flags as SwsFlags};
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use ffmpeg_next::Rational;
use synaptix_core::tensor::Tensor;
use crate::error::{IoError, Result};
use super::ffmpeg::{init, tensor_to_rgb24};

pub fn write_h264(frames: &[Tensor], path: impl AsRef<Path>, fps: f32, crf: u8) -> Result<()> {
    if frames.is_empty() {
        return Err(IoError::Video("write_h264: no frames".into()));
    }
    init()?;
    let path = path.as_ref();
    let dims = frames[0].dims();
    if dims.len() != 3 || dims[0] != 3 {
        return Err(IoError::Video(format!("expected [3,H,W] frames, got {:?}", dims)));
    }
    let (height, width) = (dims[1] as u32, dims[2] as u32);
    let fps_i = fps.round() as i32;

    let mut octx = format::output(path)
        .map_err(|e| IoError::Video(format!("open output {:?}: {e}", path)))?;

    let encoder_codec = ffmpeg::encoder::find(codec::Id::H264)
        .or_else(|| ffmpeg::encoder::find_by_name("libx264"))
        .or_else(|| ffmpeg::encoder::find(codec::Id::MPEG4))
        .ok_or_else(|| IoError::Video("no H264/MPEG4 encoder".into()))?;

    let mut enc = CodecCtx::new_with_codec(encoder_codec)
        .encoder().video()
        .map_err(|e| IoError::Video(format!("encoder.video: {e}")))?;

    enc.set_width(width);
    enc.set_height(height);
    enc.set_format(Pixel::YUV420P);
    enc.set_time_base(Rational(1, fps_i));
    enc.set_frame_rate(Some(Rational(fps_i, 1)));
    enc.set_bit_rate(8_000_000);
    enc.set_gop(12);

    if octx.format().flags().contains(format::flag::Flags::GLOBAL_HEADER) {
        enc.set_flags(codec::Flags::GLOBAL_HEADER);
    }

    let mut opts = ffmpeg::Dictionary::new();
    opts.set("preset", "fast");
    opts.set("crf", &crf.to_string());

    let mut enc = enc.open_with(opts)
        .map_err(|e| IoError::Video(format!("encoder open: {e}")))?;

    let stream_idx = {
        let mut st = octx.add_stream(encoder_codec)
            .map_err(|e| IoError::Video(format!("add_stream: {e}")))?;
        st.set_parameters(&enc);
        st.set_time_base(Rational(1, fps_i));
        st.index()
    };

    octx.write_header()
        .map_err(|e| IoError::Video(format!("write_header: {e}")))?;

    let stream_tb = octx.stream(stream_idx).unwrap().time_base();

    let mut scaler = Scaler::get(
        Pixel::RGB24, width, height,
        Pixel::YUV420P, width, height,
        SwsFlags::BILINEAR,
    ).map_err(|e| IoError::Video(format!("scaler: {e}")))?;

    for (ti, frame_tensor) in frames.iter().enumerate() {
        let rgb = tensor_to_rgb24(frame_tensor)?;
        let mut src = VideoFrame::new(Pixel::RGB24, width, height);
        src.data_mut(0)[..rgb.len()].copy_from_slice(&rgb);
        let mut dst = VideoFrame::new(Pixel::YUV420P, width, height);
        scaler.run(&src, &mut dst)
            .map_err(|e| IoError::Video(format!("scaler.run {ti}: {e}")))?;
        dst.set_pts(Some(ti as i64));
        enc.send_frame(&dst)
            .map_err(|e| IoError::Video(format!("send_frame {ti}: {e}")))?;
        flush_packets(&mut enc, &mut octx, stream_idx, Rational(1, fps_i), stream_tb)?;
    }

    enc.send_eof()
        .map_err(|e| IoError::Video(format!("send_eof: {e}")))?;
    flush_packets(&mut enc, &mut octx, stream_idx, Rational(1, fps_i), stream_tb)?;

    octx.write_trailer()
        .map_err(|e| IoError::Video(format!("write_trailer: {e}")))?;
    Ok(())
}

fn flush_packets(
    enc: &mut ffmpeg::encoder::video::Video,
    octx: &mut ffmpeg::format::context::Output,
    stream_idx: usize,
    enc_tb: Rational,
    stream_tb: Rational,
) -> Result<()> {
    let mut pkt = ffmpeg::Packet::empty();
    while enc.receive_packet(&mut pkt).is_ok() {
        pkt.set_stream(stream_idx);
        pkt.rescale_ts(enc_tb, stream_tb);
        pkt.write_interleaved(octx)
            .map_err(|e| IoError::Video(format!("write packet: {e}")))?;
    }
    Ok(())
}
