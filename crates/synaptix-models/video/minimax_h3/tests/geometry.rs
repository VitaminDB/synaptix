use synaptix_video_minimax_h3::config::{
    audio_latent_frames, frames_for_duration, latent_frames, latent_grid, snap_frame_count,
};
use synaptix_video_minimax_h3::pipeline::Geometry;

#[test]
fn frame_grid_snaps_to_17k_plus_5() {
    assert_eq!(snap_frame_count(5), 5);
    assert_eq!(snap_frame_count(6), 22);
    assert_eq!(snap_frame_count(22), 22);
    assert_eq!(snap_frame_count(23), 39);
    assert_eq!(snap_frame_count(124), 124);
    assert_eq!(snap_frame_count(125), 141);
    assert_eq!(snap_frame_count(362), 362);
    for k in 0..24 {
        let f = 17 * k + 5;
        assert_eq!(snap_frame_count(f), f, "k={k}");
    }
}

#[test]
fn five_seconds_is_124_frames() {
    assert_eq!(frames_for_duration(5.0), 124);
    assert_eq!(frames_for_duration(5.0) as f64 / 24.0, 124.0 / 24.0);
}

#[test]
fn latent_dimensions_follow_16x_4x() {
    assert_eq!(latent_frames(124), 31);
    assert_eq!(latent_grid(1344, 768), (48, 84));
    assert_eq!(latent_grid(1280, 720), (45, 80));
}

#[test]
fn audio_latents_run_at_40hz() {
    assert_eq!(audio_latent_frames(124), 207);
    assert_eq!(audio_latent_frames(24), 40);
    assert_eq!(audio_latent_frames(240), 400);
}

#[test]
fn geometry_video_tokens_match_patching() {
    let g = Geometry::new(1344, 768, 124);
    assert_eq!(g.frame_count, 124);
    assert_eq!(g.latent_t, 31);
    assert_eq!(g.latent_h, 48);
    assert_eq!(g.latent_w, 84);
    assert_eq!(g.video_tokens([1, 2, 2]), 31 * 24 * 42);
}

#[test]
fn geometry_from_duration_matches_frames() {
    let a = Geometry::from_duration(1344, 768, 5.0);
    let b = Geometry::new(1344, 768, 124);
    assert_eq!(a.frame_count, b.frame_count);
    assert_eq!(a.audio_t, b.audio_t);
}
