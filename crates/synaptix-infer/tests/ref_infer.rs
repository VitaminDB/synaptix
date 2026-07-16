use synaptix_infer::sampling::{
    GreedySampler, LogitProcessor, ProcessorContext, Sampler, TemperatureProcessor, TopKProcessor,
    TopPProcessor,
};
use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::rng::Philox4x32;
use synaptix_test_utils::load_case;

fn setup() {
    ensure_registered();
}

fn ctx() -> ProcessorContext {
    ProcessorContext {
        input_ids: Vec::new(),
        step: 0,
        batch_idx: 0,
    }
}

#[test]
fn t14_1_greedy_argmax() {
    setup();
    let t = load_case("infer", "greedy_argmax");
    let logits_2d: Vec<Vec<f32>> = t["logits"].to_vec2().unwrap();
    let expected: Vec<i64> = t["tokens"].to_vec1().unwrap();
    let mut sampler = GreedySampler;
    let mut rng = Philox4x32::new(0);
    for (i, row) in logits_2d.iter().enumerate() {
        let tok = sampler.sample(row, &mut rng).unwrap();
        assert_eq!(tok as i64, expected[i], "row {} mismatch", i);
    }
}

#[test]
fn t14_2_temperature_scaling() {
    setup();
    let t = load_case("infer", "temperature_scaling");
    let logits_2d: Vec<Vec<f32>> = t["logits"].to_vec2().unwrap();
    let temperature = t["temperature"].to_vec1::<f32>().unwrap()[0];
    let expected_scaled: Vec<Vec<f32>> = t["scaled_logits"].to_vec2().unwrap();
    let mut proc = TemperatureProcessor { temperature };
    for (i, row) in logits_2d.iter().enumerate() {
        let mut buf = row.clone();
        proc.process(&mut buf, &ctx()).unwrap();
        let exp = &expected_scaled[i];
        for (j, (a, b)) in buf.iter().zip(exp.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(diff < 1e-5, "[{},{}] diff {} > 1e-5", i, j, diff);
        }
    }
}

#[test]
fn t14_3_top_k_filter() {
    setup();
    let t = load_case("infer", "top_k_filter");
    let logits_2d: Vec<Vec<f32>> = t["logits"].to_vec2().unwrap();
    let top_k = t["top_k"].to_vec1::<i64>().unwrap()[0] as usize;
    let expected: Vec<Vec<f32>> = t["filtered_logits"].to_vec2().unwrap();
    let mut proc = TopKProcessor { k: top_k };
    for (i, row) in logits_2d.iter().enumerate() {
        let mut buf = row.clone();
        proc.process(&mut buf, &ctx()).unwrap();
        let exp = &expected[i];
        for (j, (a, b)) in buf.iter().zip(exp.iter()).enumerate() {
            if b.is_infinite() {
                assert!(a.is_infinite(), "[{},{}] expected -inf, got {}", i, j, a);
            } else {
                let diff = (a - b).abs();
                assert!(diff < 1e-5, "[{},{}] diff {} > 1e-5", i, j, diff);
            }
        }
    }
}

#[test]
fn t14_4_top_p_filter() {
    setup();
    let t = load_case("infer", "top_p_filter");
    let logits_2d: Vec<Vec<f32>> = t["logits"].to_vec2().unwrap();
    let top_p = t["top_p"].to_vec1::<f32>().unwrap()[0];
    let expected: Vec<Vec<f32>> = t["filtered_logits"].to_vec2().unwrap();
    let mut proc = TopPProcessor { p: top_p };
    for (i, row) in logits_2d.iter().enumerate() {
        let mut buf = row.clone();
        proc.process(&mut buf, &ctx()).unwrap();
        let exp = &expected[i];
        let kept_ours: usize = buf.iter().filter(|v| v.is_finite()).count();
        let kept_exp: usize = exp.iter().filter(|v| v.is_finite()).count();
        let diff_count = (kept_ours as i64 - kept_exp as i64).abs();
        assert!(
            diff_count <= 2,
            "row {} kept count diff {} (ours={}, exp={}), boundary float drift > 2",
            i, diff_count, kept_ours, kept_exp
        );
        let mismatch: usize = buf
            .iter()
            .zip(exp.iter())
            .filter(|(a, b)| a.is_infinite() != b.is_infinite())
            .count();
        assert!(
            mismatch <= 2,
            "row {} kept/dropped mismatch {} > 2 (float boundary)",
            i, mismatch
        );
    }
}

#[test]
#[ignore = "slow: 100k samples per row, ~5 min debug build; run via --include-ignored"]
fn t14_5_combined_sampling_distribution() {
    setup();
    let t = load_case("infer", "combined_sampling");
    let probs_2d: Vec<Vec<f32>> = t["final_probs"].to_vec2().unwrap();
    let vocab = probs_2d[0].len();
    let n_samples = 100_000usize;

    let mut sampler = synaptix_infer::sampling::MultinomialSampler;
    let mut rng = Philox4x32::new(42);

    for (row_idx, probs) in probs_2d.iter().enumerate() {
        let logits: Vec<f32> = probs
            .iter()
            .map(|&p| if p > 0.0 { p.ln() } else { f32::NEG_INFINITY })
            .collect();
        let mut counts = vec![0u64; vocab];
        for _ in 0..n_samples {
            let tok = sampler.sample(&logits, &mut rng).unwrap() as usize;
            counts[tok] += 1;
        }
        let mut violations: Vec<(usize, f64, f64)> = Vec::new();
        for (i, &p) in probs.iter().enumerate() {
            let empirical = counts[i] as f64 / n_samples as f64;
            let expected = p as f64;
            if expected > 0.001 {
                let diff = (empirical - expected).abs();
                if diff > 0.01 {
                    violations.push((i, empirical, expected));
                }
            } else if expected == 0.0 {
                assert_eq!(
                    counts[i], 0,
                    "row {} token {} sampled {} раз, expected prob=0",
                    row_idx, i, counts[i]
                );
            }
        }
        assert!(
            violations.len() <= 2,
            "row {} имеет {} токенов с расхождением > 0.01; первые: {:?}",
            row_idx, violations.len(), &violations[..violations.len().min(5)]
        );
        let total_in_support: u64 = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, _)| counts[i])
            .sum();
        assert_eq!(
            total_in_support, n_samples as u64,
            "row {} sampled tokens вне support",
            row_idx
        );
    }
}
