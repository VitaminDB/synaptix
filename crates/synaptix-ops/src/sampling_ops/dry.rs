use std::collections::HashMap;

/// DRY (Don't Repeat Yourself) repetition penalty. Для каждой позиции `j` в
/// истории, где `input_ids[j] == last_token`, измеряется длина обратного совпадения
/// суффиксов `l` (сколько токенов подряд совпадает с суффиксом, кончающимся на
/// последней позиции). Токен `input_ids[j+1]`, который исторически следовал за
/// таким контекстом, штрафуется: при `l ≥ allowed_length`
///   `штраф = multiplier · base^(l − allowed_length)` (берётся максимум по токену).
/// Штраф вычитается из логита соответствующего токена.
pub fn apply_dry(
    logits: &mut [f32],
    multiplier: f32,
    base: f32,
    allowed_length: usize,
    input_ids: &[u32],
) {
    let n = input_ids.len();
    if n < 2 || multiplier <= 0.0 {
        return;
    }
    let last = input_ids[n - 1];
    let vocab = logits.len();

    // макс. длина совпадения по каждому токену-кандидату (следующему)
    let mut max_len: HashMap<u32, usize> = HashMap::new();
    for j in 0..n - 1 {
        if input_ids[j] != last {
            continue;
        }
        // расширяем совпадение назад: input_ids[j-l] == input_ids[n-1-l]
        let mut l = 1usize;
        while j >= l && (n - 1) >= l && input_ids[j - l] == input_ids[n - 1 - l] {
            l += 1;
        }
        let next_tok = input_ids[j + 1];
        let e = max_len.entry(next_tok).or_insert(0);
        if l > *e {
            *e = l;
        }
    }

    for (tok, l) in max_len {
        if l >= allowed_length {
            let pen = multiplier * base.powi((l - allowed_length) as i32);
            let idx = tok as usize;
            if idx < vocab {
                logits[idx] -= pen;
            }
        }
    }
}
