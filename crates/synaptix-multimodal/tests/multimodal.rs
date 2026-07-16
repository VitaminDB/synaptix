use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_multimodal::{
    fuse_image_features, mlp_projector, pack_any_res_tokens, AudioLmEvent, FusionPlan, FusionSpan,
    Modality, ModalityRouter, MlpProjectorWeights, StreamingAudioLm,
};

fn setup() {
    ensure_registered();
}

fn const_t(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape.to_vec(), Device::Cpu).unwrap()
}

#[test]
fn fusion_plan_splits_text_by_marker() {
    let marker = 99u32;
    let tokens = vec![1u32, 2, marker, 3, 4, marker, 5];
    let plan = FusionPlan::from_text_with_image_marker(&tokens, marker, &[5, 7]).unwrap();
    assert_eq!(plan.spans.len(), 5);
    match &plan.spans[0] {
        FusionSpan::Text { token_ids } => assert_eq!(token_ids, &vec![1, 2]),
        _ => panic!(),
    }
    match &plan.spans[1] {
        FusionSpan::ImageFeature { token_count, source_idx } => {
            assert_eq!(*token_count, 5);
            assert_eq!(*source_idx, 0);
        }
        _ => panic!(),
    }
    match &plan.spans[2] {
        FusionSpan::Text { token_ids } => assert_eq!(token_ids, &vec![3, 4]),
        _ => panic!(),
    }
    match &plan.spans[3] {
        FusionSpan::ImageFeature { token_count, source_idx } => {
            assert_eq!(*token_count, 7);
            assert_eq!(*source_idx, 1);
        }
        _ => panic!(),
    }
    match &plan.spans[4] {
        FusionSpan::Text { token_ids } => assert_eq!(token_ids, &vec![5]),
        _ => panic!(),
    }
    assert_eq!(plan.total_tokens(), 2 + 5 + 2 + 7 + 1);
}

#[test]
fn fusion_plan_rejects_count_mismatch() {
    let marker = 99u32;
    let tokens = vec![marker, marker];
    let r = FusionPlan::from_text_with_image_marker(&tokens, marker, &[5]);
    assert!(r.is_err());
}

#[test]
fn fuse_image_features_concatenates_correctly() {
    setup();
    let dim = 3usize;
    let vocab = 4usize;
    let embed: Vec<f32> = (1..=(vocab * dim) as i32).map(|x| x as f32).collect();
    let embed_t = const_t(embed, &[vocab, dim]);
    let img_feat = const_t(vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0], &[2, 3]);
    let tokens = vec![0u32, 1, 99, 2];
    let plan = FusionPlan::from_text_with_image_marker(&tokens, 99, &[2]).unwrap();
    let fused = fuse_image_features(&plan, &embed_t, &[img_feat]).unwrap();
    assert_eq!(fused.dims(), &[5, 3]);
}

#[test]
fn pack_any_res_2x2_tiles() {
    setup();
    let t = |v: f32| const_t(vec![v; 6], &[3, 2]);
    let tiles = vec![t(1.0), t(2.0), t(3.0), t(4.0)];
    let (packed, plan) = pack_any_res_tokens(&tiles, 2, 2).unwrap();
    assert_eq!(packed.dims(), &[12, 2]);
    assert_eq!(plan.positions.len(), 4);
    assert_eq!(plan.positions[0].row, 0);
    assert_eq!(plan.positions[0].col, 0);
    assert_eq!(plan.positions[3].row, 1);
    assert_eq!(plan.positions[3].col, 1);
    assert_eq!(plan.tokens_per_tile, 3);
}

#[test]
fn modality_router_basic() {
    let mut r = ModalityRouter::new();
    r.route(Modality::Text, "qwen36").route(Modality::Vision, "siglip");
    assert_eq!(r.lookup(Modality::Text), Some("qwen36"));
    assert_eq!(r.lookup(Modality::Vision), Some("siglip"));
    assert_eq!(r.lookup(Modality::Audio), None);
}

#[test]
fn mlp_projector_output_shape() {
    setup();
    let in_dim = 4usize;
    let hidden = 6usize;
    let out_dim = 3usize;
    let weights = MlpProjectorWeights {
        fc1_weight: const_t(vec![0.1; in_dim * hidden], &[in_dim, hidden]),
        fc1_bias: Some(const_t(vec![0.0; hidden], &[1, hidden])),
        fc2_weight: const_t(vec![0.1; hidden * out_dim], &[hidden, out_dim]),
        fc2_bias: Some(const_t(vec![0.0; out_dim], &[1, out_dim])),
    };
    let input = const_t(vec![1.0; 8 * in_dim], &[8, in_dim]);
    let out = mlp_projector(&input, &weights).unwrap();
    assert_eq!(out.dims(), &[8, out_dim]);
}

#[test]
fn streaming_audio_lm_drains_frames() {
    let mut lm = StreamingAudioLm::new(16000, 4);
    lm.feed_audio(&[1.0, 2.0, 3.0]);
    assert!(lm.drain_frames().is_empty());
    lm.feed_audio(&[4.0, 5.0, 6.0, 7.0, 8.0]);
    let frames = lm.drain_frames();
    assert_eq!(frames, vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]);
    assert_eq!(lm.pending_audio_samples(), 0);
}

#[test]
fn streaming_audio_lm_event_queue() {
    let mut lm = StreamingAudioLm::new(16000, 4);
    lm.push_event(AudioLmEvent::TextTokens(vec![1, 2, 3]));
    lm.push_event(AudioLmEvent::Eos);
    assert_eq!(lm.next_event(), Some(AudioLmEvent::TextTokens(vec![1, 2, 3])));
    assert_eq!(lm.next_event(), Some(AudioLmEvent::Eos));
    assert_eq!(lm.next_event(), None);
}
