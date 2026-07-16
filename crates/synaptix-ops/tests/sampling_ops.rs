//! Аналитические unit-тесты для sampling-операций над логитами (без torch-эталонов).

use synaptix_ops::sampling_ops::{
    dry::apply_dry, epsilon::apply_epsilon, eta::apply_eta, xtc::apply_xtc,
};

const NINF: f32 = f32::NEG_INFINITY;

#[test]
fn epsilon_masks_low_prob() {
    // softmax([5,0,0]) ≈ [0.986, 0.0067, 0.0067]; eps=0.1 → маскируются 1,2
    let mut logits = vec![5.0f32, 0.0, 0.0];
    apply_epsilon(&mut logits, 0.1);
    assert_eq!(logits[0], 5.0);
    assert_eq!(logits[1], NINF);
    assert_eq!(logits[2], NINF);
}

#[test]
fn epsilon_keeps_argmax_when_all_below() {
    // равномерное [0,0,0,0]: каждая prob=0.25 < 0.3 → всё бы замаскировалось,
    // но argmax восстанавливается (max_by при равенстве → последний индекс)
    let mut logits = vec![0.0f32; 4];
    apply_epsilon(&mut logits, 0.3);
    let kept: Vec<usize> = (0..4).filter(|&i| logits[i].is_finite()).collect();
    assert_eq!(kept.len(), 1, "ровно один токен должен остаться");
}

#[test]
fn eta_masks_low_prob_peaked() {
    // пиковое распределение: threshold ≈ eta=0.1 → маскируются хвостовые 1,2
    let mut logits = vec![5.0f32, 0.0, 0.0];
    apply_eta(&mut logits, 0.1);
    assert_eq!(logits[0], 5.0);
    assert_eq!(logits[1], NINF);
    assert_eq!(logits[2], NINF);
}

#[test]
fn eta_noop_when_zero() {
    let mut logits = vec![3.0f32, 1.0, -2.0];
    apply_eta(&mut logits, 0.0);
    assert_eq!(logits, vec![3.0, 1.0, -2.0]);
}

#[test]
fn xtc_excludes_top_above_threshold() {
    // probs ≈ [0.665, 0.245, 0.090, ~0]; threshold=0.05 → above={0,1,2},
    // оставляем наименее вероятный из них (2), маскируем 0,1; токен 3 (ниже порога) цел
    let mut logits = vec![3.0f32, 2.0, 1.0, -5.0];
    apply_xtc(&mut logits, 1.0, 0.05);
    assert_eq!(logits[0], NINF);
    assert_eq!(logits[1], NINF);
    assert_eq!(logits[2], 1.0);
    assert_eq!(logits[3], -5.0);
}

#[test]
fn xtc_noop_when_probability_zero() {
    let mut logits = vec![3.0f32, 2.0, 1.0];
    apply_xtc(&mut logits, 0.0, 0.05);
    assert_eq!(logits, vec![3.0, 2.0, 1.0]);
}

#[test]
fn dry_penalizes_repeat_continuation() {
    // история [1,2,3,1,2]: суффикс "1,2" повторяется → следующий токен 3 штрафуется.
    // l=2, allowed=2 → pen = multiplier·base^0 = 1.0
    let mut logits = vec![0.0f32; 4];
    apply_dry(&mut logits, 1.0, 2.0, 2, &[1, 2, 3, 1, 2]);
    assert_eq!(logits[3], -1.0);
    assert_eq!(logits[0], 0.0);
    assert_eq!(logits[1], 0.0);
    assert_eq!(logits[2], 0.0);
}

#[test]
fn dry_noop_below_allowed_length() {
    // совпадение длины 2, но allowed_length=3 → штрафа нет
    let mut logits = vec![0.0f32; 4];
    apply_dry(&mut logits, 1.0, 2.0, 3, &[1, 2, 3, 1, 2]);
    assert_eq!(logits, vec![0.0; 4]);
}
