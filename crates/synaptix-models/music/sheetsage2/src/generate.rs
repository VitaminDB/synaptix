//! Жадная генерация событий под грамматикой (`constrained_prompt_generate`).
//!
//! На каждом шаге логиты вне разрешённого грамматикой множества отбрасываются
//! и берётся argmax (первый при равенстве, как у `torch.argmax`). Окно, за
//! которым идёт следующее, останавливается на первой метке времени за
//! `stop_time`: дальше — правый контекст, его транскрибирует следующее окно.

use crate::decoder::{Decoder, DecoderState};
use crate::grammar::GrammarState;
use crate::vocab::{Vocab, EOS, OUT, SOS};
use crate::SheetError;

/// Выданные токены (префикс в начале, `<eos>` в конце) и признак того, что
/// упёрлись в предел длины.
pub struct Generated {
    pub tokens: Vec<u32>,
    pub hit_limit: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn constrained_generate(
    decoder: &Decoder,
    state: &mut DecoderState,
    vocab: &Vocab,
    prefix: &[u32],
    max_len: usize,
    stop_time: Option<f64>,
    cancel: &dyn Fn() -> bool,
    on_token: &mut dyn FnMut(usize),
) -> Result<Generated, SheetError> {
    let mut prefix = prefix.to_vec();
    if prefix.first() != Some(&SOS) {
        return Err(SheetError::Sequence("префикс генерации должен начинаться с <|sos|>".into()));
    }
    if prefix.last() == Some(&EOS) {
        prefix.pop();
    }
    let out_index = prefix
        .iter()
        .position(|&t| t == OUT)
        .ok_or_else(|| SheetError::Sequence("в префиксе нет <|out|>".into()))?;
    let mut grammar = GrammarState::default();
    for &t in &prefix[out_index + 1..] {
        grammar.update(vocab, t)?;
    }
    let mut output = prefix.clone();
    let mut input = prefix;
    let mut mask = Vec::new();
    let mut finished = false;
    while output.len() < max_len {
        if cancel() {
            return Err(SheetError::Cancelled("декодер"));
        }
        let logits = decoder.step(state, &input)?;
        grammar.allowed(vocab, &mut mask);
        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, (&l, &ok)) in logits.iter().zip(&mask).enumerate() {
            // NaN не выбирается никогда; -inf за маской — как masked_fill релиза.
            if ok && l > best_val {
                best_val = l;
                best = i;
            }
        }
        let token = best as u32;
        output.push(token);
        finished = grammar.update(vocab, token)?;
        if !finished {
            if let (Some(stop), Some(id)) = (stop_time, vocab.token_to_time_id(token)) {
                if id as f64 / vocab.time_hz as f64 >= stop {
                    output.push(EOS);
                    finished = true;
                }
            }
        }
        on_token(output.len());
        if finished {
            break;
        }
        input = vec![token];
    }
    let hit_limit = !finished;
    if output.last() != Some(&EOS) {
        output.push(EOS);
    }
    Ok(Generated { tokens: output, hit_limit })
}
