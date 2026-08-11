use synaptix_video_minimax_h3::adaln::{mod_segments, AdalnPlan, RoleTable, TimeRole};
use synaptix_video_minimax_h3::config::{
    ADALN_MODALITIES, MODALITY_AUDIO, MODALITY_TEXT, MODALITY_VIDEO,
};
use synaptix_video_minimax_h3::layout::{Keyframe, LayoutRequest, PackedLayout, SegmentKind};
use synaptix_video_minimax_h3::scheduler::H3Scheduler;

fn plain_layout() -> PackedLayout {
    PackedLayout::build(&LayoutRequest::new(6, 2, 8, 8, 4)).unwrap()
}

fn keyframe_layout() -> PackedLayout {
    PackedLayout::build(
        &LayoutRequest::new(6, 2, 8, 8, 4)
            .with_frame_count(5)
            .with_keyframes(vec![Keyframe { resolved_frame_index: 0 }]),
    )
    .unwrap()
}

#[test]
fn plain_run_has_two_time_roles() {
    let l = plain_layout();
    let roles = RoleTable::for_layout(&l);
    assert_eq!(roles.roles, vec![TimeRole::Base, TimeRole::Audio]);
    assert_eq!(roles.index(TimeRole::Base), 0);
    assert_eq!(roles.index(TimeRole::Audio), 1);
}

#[test]
fn visual_conditioning_adds_a_role() {
    let l = keyframe_layout();
    let roles = RoleTable::for_layout(&l);
    assert_eq!(roles.roles, vec![TimeRole::Base, TimeRole::Audio, TimeRole::VisualCond]);
    assert_eq!(roles.role_for(SegmentKind::Cond), TimeRole::VisualCond);
    assert_eq!(roles.role_for(SegmentKind::Video), TimeRole::Base);
    assert_eq!(roles.role_for(SegmentKind::Text), TimeRole::Base);
}

#[test]
fn mod_rows_encode_role_times_modality() {
    let l = plain_layout();
    let roles = RoleTable::for_layout(&l);
    let segs = mod_segments(&l, &roles, None);
    assert_eq!(segs.len(), 3);
    assert_eq!(segs[0].row, 0 * ADALN_MODALITIES + MODALITY_TEXT);
    assert_eq!(segs[1].row, 1 * ADALN_MODALITIES + MODALITY_AUDIO);
    assert_eq!(segs[2].row, 0 * ADALN_MODALITIES + MODALITY_VIDEO);
}

#[test]
fn segments_tile_the_sequence_without_gaps() {
    let l = keyframe_layout();
    let roles = RoleTable::for_layout(&l);
    let segs = mod_segments(&l, &roles, None);
    let mut cursor = 0usize;
    for s in &segs {
        assert_eq!(s.start, cursor, "разрыв перед {}", s.start);
        assert!(s.stop > s.start);
        cursor = s.stop;
    }
    assert_eq!(cursor, l.seq_len);
}

#[test]
fn text_tags_split_the_text_span_into_runs() {
    let l = plain_layout();
    let roles = RoleTable::for_layout(&l);
    let tags = vec![1u8, 1, 0, 0, 0, 1];
    let segs = mod_segments(&l, &roles, Some(&tags));
    let text_runs: Vec<_> = segs.iter().filter(|s| s.start < 6).collect();
    assert_eq!(text_runs.len(), 3);
    assert_eq!((text_runs[0].start, text_runs[0].stop), (0, 2));
    assert_eq!(text_runs[0].row, MODALITY_TEXT);
    assert_eq!((text_runs[1].start, text_runs[1].stop), (2, 5));
    assert_eq!(text_runs[1].row, MODALITY_VIDEO);
    assert_eq!((text_runs[2].start, text_runs[2].stop), (5, 6));
    assert_eq!(text_runs[2].row, MODALITY_TEXT);
}

#[test]
fn cond_timesteps_are_pinned_near_one() {
    let l = keyframe_layout();
    let sched = H3Scheduler::new(4, 12.0, 3.0);
    let plan = AdalnPlan::build(&l, &sched, None, None);
    assert_eq!(plan.steps(), 4);
    assert_eq!(plan.rows(), 3 * ADALN_MODALITIES);
    assert_eq!(plan.final_rows(), 3);
    for step in 0..plan.steps() {
        let ts = &plan.timesteps[step];
        assert_eq!(ts.len(), 3);
        assert!(ts[2] >= 0.999 - 1e-6, "step {step}: {}", ts[2]);
        assert!((ts[0] - (1.0 - sched.video_sigma(step)) as f32).abs() < 1e-6);
        assert!((ts[1] - (1.0 - sched.audio_sigma(step)) as f32).abs() < 1e-6);
    }
}

#[test]
fn cache_size_scales_with_steps_and_roles() {
    let l = plain_layout();
    let sched = H3Scheduler::new(8, 12.0, 3.0);
    let plan = AdalnPlan::build(&l, &sched, None, None);
    let bytes = plan.estimated_bytes(50, 5376, synaptix_core::dtype::DType::BF16);
    let per_block = 8 * 6 * 6 * 5376 * 2;
    let final_layer = 8 * 2 * 2 * 5376 * 2;
    assert_eq!(bytes, per_block * 50 + final_layer);
    assert!(bytes < 200 * 1024 * 1024, "кэш {bytes} байт слишком велик");
}

#[test]
fn cache_slices_are_reshapable_at_every_step_and_chunk() {
    synaptix_kernels_cpu::ensure_registered();
    use synaptix_core::device::Device;
    use synaptix_core::tensor::Tensor;
    use synaptix_video_minimax_h3::adaln::AdalnCache;
    use synaptix_video_minimax_h3::config::{ADALN_CHUNKS, FINAL_ADALN_CHUNKS};

    let (steps, rows, final_rows, hidden, num_blocks) = (3usize, 6usize, 2usize, 8usize, 2usize);
    let fill = |chunks: usize, r: usize| -> Tensor {
        let n = steps * chunks * r * hidden;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::from_vec(data, vec![steps, chunks, r, hidden], Device::Cpu).unwrap()
    };
    let blocks: Vec<Tensor> = (0..num_blocks).map(|_| fill(ADALN_CHUNKS, rows)).collect();
    let cache = AdalnCache::new(
        blocks,
        fill(FINAL_ADALN_CHUNKS, final_rows),
        rows,
        final_rows,
        hidden,
        steps,
    );

    for b in 0..num_blocks {
        for step in 0..steps {
            for c in 0..ADALN_CHUNKS {
                let t = cache.chunk(b, step, c).expect("chunk");
                assert_eq!(t.dims(), &[rows, hidden]);
                let got = t.reshape(vec![rows * hidden]).unwrap().to_vec1::<f32>().unwrap();
                let base = ((step * ADALN_CHUNKS + c) * rows * hidden) as f32;
                assert_eq!(got[0], base, "блок {b} шаг {step} чанк {c}");
                assert_eq!(got[rows * hidden - 1], base + (rows * hidden - 1) as f32);
                for r in 0..rows {
                    let one = cache.row(b, step, c, r).expect("row");
                    assert_eq!(one.dims(), &[1, hidden]);
                    assert_eq!(
                        one.reshape(vec![hidden]).unwrap().to_vec1::<f32>().unwrap()[0],
                        base + (r * hidden) as f32
                    );
                }
            }
        }
    }

    for step in 0..steps {
        for c in 0..FINAL_ADALN_CHUNKS {
            assert_eq!(cache.final_chunk(step, c).expect("final").dims(), &[final_rows, hidden]);
            for r in 0..final_rows {
                assert_eq!(cache.final_row(step, c, r).expect("final row").dims(), &[1, hidden]);
            }
        }
    }
}
