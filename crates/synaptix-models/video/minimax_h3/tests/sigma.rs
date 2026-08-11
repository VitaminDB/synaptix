use synaptix_video_minimax_h3::scheduler::{shift_sigma, time_shift_sigma, unshift_sigma, H3Scheduler};

#[test]
fn shift_and_unshift_round_trip() {
    for shift in [3.0f64, 12.0] {
        for base in [0.0f64, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0] {
            let s = shift_sigma(base, shift);
            let back = unshift_sigma(s, shift);
            assert!((back - base).abs() < 1e-12, "shift={shift} base={base} back={back}");
        }
    }
}

#[test]
fn time_shift_maps_between_schedules() {
    for sigma in [0.05f64, 0.3, 0.7, 0.95] {
        let a = time_shift_sigma(sigma, 12.0, 3.0);
        let back = time_shift_sigma(a, 3.0, 12.0);
        assert!((back - sigma).abs() < 1e-12, "sigma={sigma} back={back}");
    }
}

#[test]
fn endpoints_are_fixed() {
    assert!((shift_sigma(1.0, 12.0) - 1.0).abs() < 1e-12);
    assert!(shift_sigma(0.0, 12.0).abs() < 1e-12);
    assert!((time_shift_sigma(1.0, 12.0, 3.0) - 1.0).abs() < 1e-12);
}

#[test]
fn audio_schedule_is_less_shifted_than_video() {
    let s = H3Scheduler::new(20, 12.0, 3.0);
    for step in 1..s.steps() {
        let v = s.video_sigma(step);
        let a = s.audio_sigma(step);
        assert!(a <= v + 1e-12, "step={step} audio={a} video={v}");
    }
}

#[test]
fn schedule_is_monotonically_decreasing_to_zero() {
    let s = H3Scheduler::new(8, 12.0, 3.0);
    assert_eq!(s.steps(), 8);
    assert!((s.video_sigma(0) - 1.0).abs() < 1e-12);
    assert!(s.video_sigma(8).abs() < 1e-12);
    for i in 0..s.steps() {
        assert!(s.video_sigma(i) > s.video_sigma(i + 1), "step {i}");
        assert!(s.video_dt(i) < 0.0);
        assert!(s.audio_dt(i) < 0.0);
    }
}

#[test]
fn timesteps_are_one_minus_sigma() {
    let s = H3Scheduler::new(4, 12.0, 3.0);
    for i in 0..=s.steps() {
        assert!((s.video_t(i) - (1.0 - s.video_sigma(i))).abs() < 1e-12);
        assert!((s.audio_t(i) - (1.0 - s.audio_sigma(i))).abs() < 1e-12);
    }
}
