//! C5: n-gram (prompt-lookup) draft + интеграция с verify_tokens.

use synaptix_infer::sampling::speculative::draft_model::NgramDraftModel;
use synaptix_infer::sampling::speculative::lookahead::LookaheadDecoder;
use synaptix_infer::sampling::speculative::{ngram_lookup, verify_tokens, DraftModel};
use synaptix_ops::rng::Philox4x32;

#[test]
fn t42_1_ngram_lookup_longest_match() {
    // Суффикс [1,2,3] встречался ранее с продолжением 4,1.
    let toks = [1, 2, 3, 4, 1, 2, 3];
    assert_eq!(ngram_lookup(&toks, 3, 2), vec![4, 1]);
    assert_eq!(ngram_lookup(&toks, 3, 1), vec![4]);
}

#[test]
fn t42_2_ngram_lookup_falls_back_to_shorter() {
    // Длинные суффиксы не совпадают; срабатывает биграмма [1,2] → продолжение 9.
    let toks = [5, 1, 2, 9, 1, 2];
    assert_eq!(ngram_lookup(&toks, 4, 1), vec![9]);
}

#[test]
fn t42_3_ngram_lookup_no_match() {
    assert_eq!(ngram_lookup(&[1, 2, 3], 3, 2), Vec::<u32>::new());
    assert_eq!(ngram_lookup(&[], 3, 2), Vec::<u32>::new());
}

#[test]
fn t42_4_draft_model_and_logits_shape() {
    let mut m = NgramDraftModel::new(3, 16);
    let toks = [1, 2, 3, 4, 1, 2, 3];
    let drafted = m.draft(&toks, 2).unwrap();
    assert_eq!(drafted, vec![4, 1]);

    let logits = m.draft_logits(&toks, 2).unwrap();
    assert_eq!(logits.len(), 2);
    assert_eq!(logits[0].len(), 16);
    // Пик на предложенном токене.
    assert_eq!(argmax(&logits[0]), 4);
    assert_eq!(argmax(&logits[1]), 1);
}

#[test]
fn t42_5_verify_accepts_when_target_agrees() {
    let mut m = NgramDraftModel::new(3, 16);
    let toks = [1, 2, 3, 4, 1, 2, 3];
    let drafted = m.draft(&toks, 2).unwrap();
    let draft_logits = m.draft_logits(&toks, 2).unwrap();
    // Таргет согласен (пик на тех же токенах) → принимаются все.
    let target_logits = draft_logits.clone();
    let mut rng = Philox4x32::new(0);
    let out = verify_tokens(&drafted, &draft_logits, &target_logits, &mut rng);
    assert_eq!(out.accepted, vec![4, 1]);
    assert_eq!(out.rejected_at, None);
}

#[test]
fn t42_6_verify_rejects_when_target_disagrees() {
    let mut m = NgramDraftModel::new(3, 16);
    let toks = [1, 2, 3, 4, 1, 2, 3];
    let drafted = m.draft(&toks, 2).unwrap();
    let draft_logits = m.draft_logits(&toks, 2).unwrap();
    // Таргет уверенно хочет другой токен на позиции 0 → отказ на 0.
    let mut target_logits = vec![vec![0.0f32; 16]; 2];
    target_logits[0][15] = 30.0;
    target_logits[1][1] = 30.0;
    let mut rng = Philox4x32::new(0);
    let out = verify_tokens(&drafted, &draft_logits, &target_logits, &mut rng);
    assert_eq!(out.accepted, Vec::<u32>::new());
    assert_eq!(out.rejected_at, Some(0));
}

#[test]
fn t42_7_lookahead_window() {
    let dec = LookaheadDecoder::new(8, 2);
    // Биграмма [1,2] раньше → продолжение 3.
    let toks = [9, 9, 1, 2, 3, 1, 2];
    assert_eq!(dec.propose(&toks, 1), vec![3]);
    // Узкое окно отрезает раннее вхождение → пусто.
    let narrow = LookaheadDecoder::new(3, 2);
    assert_eq!(narrow.propose(&toks, 1), Vec::<u32>::new());
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap()
}
