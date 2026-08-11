use synaptix_video_minimax_h3::layout::{
    Keyframe, LayoutRequest, PackedLayout, RefBlock, SegmentKind,
};
use synaptix_video_minimax_h3::rope::{
    axis_from_sqrt_area, video_t_grid, video_t_span_total, FrameGrid, FRAME_RESCALE,
};

fn t2va(text: usize, lt: usize, lh: usize, lw: usize, at: usize) -> PackedLayout {
    PackedLayout::build(&LayoutRequest::new(text, lt, lh, lw, at)).unwrap()
}

#[test]
fn t2va_order_is_text_audio_video() {
    let l = t2va(16, 4, 8, 8, 10);
    let kinds: Vec<SegmentKind> = l.segments.iter().map(|s| s.kind).collect();
    assert_eq!(kinds, vec![SegmentKind::Text, SegmentKind::Audio, SegmentKind::Video]);
    let frame_rows = (8 / 2) * (8 / 2);
    assert_eq!(l.frame_rows, frame_rows);
    assert_eq!(l.seq_len, 16 + 10 * 2 + 4 * frame_rows);
    assert_eq!(l.positions.len(), l.seq_len);
}

#[test]
fn all_target_rows_are_updatable() {
    let l = t2va(16, 4, 8, 8, 10);
    assert!(l.img_update.iter().all(|u| *u));
    assert!(l.audio_update.iter().all(|u| *u));
    assert_eq!(l.target_video_rows(), 4 * l.frame_rows);
    assert_eq!(l.target_audio_rows(), 20);
}

#[test]
fn text_positions_are_linear_on_t_only() {
    let l = t2va(5, 2, 4, 4, 3);
    for i in 0..5 {
        assert_eq!(l.positions[i], [i as f64, 0.0, 0.0]);
    }
}

#[test]
fn audio_rows_are_channel_major_with_extreme_w() {
    let (lh, lw, at) = (8usize, 8usize, 4usize);
    let l = t2va(2, 2, lh, lw, at);
    let frame = FrameGrid::new(lh, lw);
    let (lo, hi) = frame.w_bounds();
    let seg = l.segment(SegmentKind::Audio).unwrap();
    for i in 0..at {
        let left = l.positions[seg.start + i];
        let right = l.positions[seg.start + at + i];
        assert_eq!(left[1], 0.0);
        assert_eq!(right[1], 0.0);
        assert_eq!(left[2], lo);
        assert_eq!(right[2], hi);
        assert_eq!(left[0], right[0]);
    }
}

#[test]
fn video_time_grid_uses_1_4_4_4_4_pattern() {
    let g = video_t_grid(6, 0.0);
    let r = FRAME_RESCALE;
    let expect = [0.0, r, r + 4.0 * r, r + 8.0 * r, r + 12.0 * r, r + 16.0 * r];
    for (a, b) in g.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
    assert!((video_t_span_total(1) - r).abs() < 1e-12);
    assert!((video_t_span_total(2) - 5.0 * r).abs() < 1e-12);
}

#[test]
fn spatial_axis_is_area_normalized_endpoint_exclusive() {
    let axis = axis_from_sqrt_area(8, 2, 8.0);
    assert_eq!(axis.len(), 4);
    assert!((axis[0] - 0.0).abs() < 1e-12);
    assert!((axis[1] - 8.0).abs() < 1e-12);
    assert!((axis[3] - 24.0).abs() < 1e-12);

    let wide = axis_from_sqrt_area(4, 2, 8.0);
    assert_eq!(wide.len(), 2);
    assert!((wide[0] - 8.0).abs() < 1e-12);
    assert!((wide[1] - 16.0).abs() < 1e-12);
}

#[test]
fn keyframes_add_cond_segments_before_audio() {
    let req = LayoutRequest::new(4, 3, 8, 8, 5)
        .with_frame_count(9)
        .with_keyframes(vec![
            Keyframe { resolved_frame_index: 0 },
            Keyframe { resolved_frame_index: 8 },
        ]);
    let l = PackedLayout::build(&req).unwrap();
    let kinds: Vec<SegmentKind> = l.segments.iter().map(|s| s.kind).collect();
    assert_eq!(
        kinds,
        vec![
            SegmentKind::Text,
            SegmentKind::Cond,
            SegmentKind::Cond,
            SegmentKind::Audio,
            SegmentKind::Video
        ]
    );
    assert!(l.has_visual_cond());
    let cond_rows = 2 * l.frame_rows;
    assert_eq!(l.img_update.iter().filter(|u| !**u).count(), cond_rows);

    let first = l.segments[1];
    assert_eq!(l.positions[first.start][0], 4.0);
    let last = l.segments[2];
    let expect = 4.0 + video_t_span_total(3) - FRAME_RESCALE;
    assert!((l.positions[last.start][0] - expect).abs() < 1e-12);
}

#[test]
fn keyframe_in_the_middle_is_rejected() {
    let req = LayoutRequest::new(4, 3, 8, 8, 5)
        .with_frame_count(9)
        .with_keyframes(vec![Keyframe { resolved_frame_index: 4 }]);
    assert!(PackedLayout::build(&req).is_err());
}

#[test]
fn reference_blocks_advance_the_cursor() {
    let req = LayoutRequest::new(3, 2, 8, 8, 4).with_refs(vec![
        RefBlock::Image { latent_h: 8, latent_w: 8 },
        RefBlock::Audio { latent_t: 6 },
        RefBlock::Video { latent_t: 2, latent_h: 8, latent_w: 8, audio_latent_t: 3 },
    ]);
    let l = PackedLayout::build(&req).unwrap();
    let kinds: Vec<SegmentKind> = l.segments.iter().map(|s| s.kind).collect();
    assert_eq!(
        kinds,
        vec![
            SegmentKind::Text,
            SegmentKind::RefImg,
            SegmentKind::RefAudio,
            SegmentKind::RefAudio,
            SegmentKind::RefImg,
            SegmentKind::Audio,
            SegmentKind::Video
        ]
    );
    assert!(l.has_visual_cond());
    assert!(l.has_audio_cond());

    let ref_img = l.segments[1];
    assert_eq!(l.positions[ref_img.start][0], 3.0);
    let ref_audio = l.segments[2];
    assert_eq!(l.positions[ref_audio.start][0], 4.0);
    let ref_video = l.segments[4];
    assert_eq!(l.positions[ref_video.start][0], 10.0);
}

#[test]
fn signature_distinguishes_conditioning() {
    let plain = t2va(4, 2, 8, 8, 4);
    let with_kf = PackedLayout::build(
        &LayoutRequest::new(4, 2, 8, 8, 4)
            .with_frame_count(5)
            .with_keyframes(vec![Keyframe { resolved_frame_index: 0 }]),
    )
    .unwrap();
    assert_ne!(plain.signature, with_kf.signature);
}
