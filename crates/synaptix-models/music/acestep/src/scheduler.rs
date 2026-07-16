
pub fn timestep_schedule(infer_steps: usize, shift: f32) -> Vec<f32> {
    let n = infer_steps.max(1);
    let mut t: Vec<f32> = (0..=n).map(|i| 1.0 - i as f32 / n as f32).collect();
    if (shift - 1.0).abs() > 1e-9 {
        for v in t.iter_mut() {
            *v = shift * *v / (1.0 + (shift - 1.0) * *v);
        }
    }
    if let Some(first) = t.first_mut() {
        *first = 1.0;
    }
    if let Some(last) = t.last_mut() {
        *last = 0.0;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_no_shift() {
        let t = timestep_schedule(8, 1.0);
        assert_eq!(t.len(), 9);
        assert!((t[0] - 1.0).abs() < 1e-6);
        assert!((t[8] - 0.0).abs() < 1e-6);
        assert!((t[4] - 0.5).abs() < 1e-6);
        for w in t.windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn schedule_shift_monotonic() {
        let t = timestep_schedule(16, 3.0);
        assert_eq!(t.len(), 17);
        assert!((t[0] - 1.0).abs() < 1e-6);
        assert!((t[16] - 0.0).abs() < 1e-6);
        for w in t.windows(2) {
            assert!(w[0] > w[1], "schedule must be strictly decreasing");
        }
    }
}
