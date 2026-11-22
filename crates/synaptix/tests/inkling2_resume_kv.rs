//! Точка возврата сессии Inkling-2 через фасад — так, как зовёт служба:
//! перед ходом `set_resume_at(history)` (конец истории перед приглашением к
//! ответу), клиент убирает из истории размышления прошлых ответов. Новый
//! промпт тогда продолжает прошлый промпт, но не сгенерированный ответ, и
//! кэш берётся от снимка в точке возврата — весь прошлый промпт до истории.
//!
//! Шаблон Inkling-2 приглашения к ответу не дописывает (промпт кончается
//! `<|end_message|>` реплики пользователя, `<|message_model|>` модель пишет
//! сама), поэтому история совпадает с промптом и `history_tokens` даёт `None`:
//! точка возврата — на конце промпта, он же конец истории.
//!
//!   SYN_INKLING2_DIR=<бандл> cargo test -p synaptix --profile fast-release \
//!     --test inkling2_resume_kv -- --nocapture --test-threads 1
//!
//! Длинный прогон (~60K токенов, ход с размышлениями, затем без них) — ещё и
//! `SYN_INKLING2_LONG=<корпус.txt>` (`SYN_INKLING2_LONG_TOKENS`, 60000).
//! Без `SYN_INKLING2_DIR` тесты проходят как no-op.

use std::path::Path;
use std::time::Instant;

use synaptix::facade::chat::{history_tokens, prompt_for_model};
use synaptix::facade::llm::{load_llm_full, load_llm_with_policy, Device, GenerationOptions, LlmGeneration, LlmTokenizer, Message, QuantPolicy, RuntimeOptions};

fn opts(max_seq_len: usize, max_new: usize) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens: max_new,
        max_seq_len,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 7,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
    }
}

/// Промпт с размышлениями и граница истории — как в службе.
fn render(tok: &LlmTokenizer, msgs: &[Message]) -> (Vec<u32>, Option<usize>) {
    let prompt = tok.apply_chat_template_reasoning(msgs, true, true, None, None).expect("template");
    let prompt = prompt_for_model(tok, true, prompt);
    let ids = tok.encode(&prompt).expect("encode");
    let history = history_tokens(tok, msgs, true, None, None, &prompt, &ids);
    (ids, history)
}

/// Разбор ответа по спецтокенам: (размышления, ответ).
fn split_answer(tok: &LlmTokenizer, out: &[u32]) -> (String, String) {
    let single = |s: &str| tok.encode(s).expect("encode")[0];
    let (message, thinking, text, end) =
        (single("<|message_model|>"), single("<|content_thinking|>"), single("<|content_text|>"), single("<|end_message|>"));
    let (mut reasoning, mut answer, mut cur) = (Vec::new(), Vec::new(), None::<bool>);
    for &id in out {
        if id == thinking {
            cur = Some(true);
        } else if id == text {
            cur = Some(false);
        } else if id == end || id == message {
            cur = None;
        } else if let Some(t) = cur {
            if t { reasoning.push(id) } else { answer.push(id) }
        }
    }
    (tok.decode(&reasoning).unwrap_or_default(), tok.decode(&answer).unwrap_or_default())
}

/// Размышления выкинуты из истории → из кэша весь прошлый промпт до границы;
/// ответ тот же, что у хода с нуля. Сессия заводится на малый контекст
/// (кольца SWA ещё линейные), ход 2 её доращивает — снимок от раскладки
/// кольца не зависит.
#[test]
fn inkling2_turn_without_reasoning_reuses_prompt() {
    let Ok(path) = std::env::var("SYN_INKLING2_DIR") else {
        eprintln!("SYN_INKLING2_DIR не задан — пропускаем");
        return;
    };
    let (model, tok) = load_llm_with_policy(Path::new(&path), QuantPolicy::balance(), &Device::Cuda(0)).expect("load");
    let max_new = 1500usize;

    let mut msgs = vec![Message::user("Сколько будет 17 * 23? Ответь одним числом.")];
    let (ids1, history1) = render(&tok, &msgs);
    let h1 = history1.unwrap_or(ids1.len());
    println!("ход 1: промпт {} ток., граница службы {history1:?}, точка {h1}", ids1.len());
    // Малая сессия: SWA-слоты линейные (ёмкость меньше кольца 1024 + 512).
    let mut session = model.new_kv_session(1024, 600).expect("session").expect("Inkling-2 умеет префикс-KV");
    session.set_resume_at(history1);
    let mut out1 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(1024, 600));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let reused = r
        .generate_streaming_cached(&mut session, &ids1, &tok, |id, _| {
            out1.push(id);
            true
        })
        .expect("ход 1");
    assert_eq!(reused, 0);
    let (reasoning, answer) = split_answer(&tok, &out1);
    println!("ход 1: размышлений {} симв., ответ {answer:?}", reasoning.len());
    assert!(!reasoning.is_empty(), "ход 1 должен размышлять");
    assert!(answer.contains("391"), "{answer}");

    // Клиент убирает размышления из истории.
    msgs.push(Message::assistant_turn(answer, None, Vec::new()));
    msgs.push(Message::user("А теперь прибавь 9. Ответь одним числом."));
    let (ids2, history2) = render(&tok, &msgs);
    assert_eq!(&ids2[..h1], &ids1[..h1], "история хода 1 — префикс промпта 2");
    let head: Vec<u32> = ids1.iter().chain(&out1).copied().collect();
    assert_ne!(&ids2[..head.len().min(ids2.len())], &head[..head.len().min(ids2.len())], "промпт 2 не продолжает сгенерированное");
    assert_eq!(session.reusable(&ids2), h1, "из кэша — весь промпт до границы истории");

    // Служба доращивает сессию под ход: SWA-слоты из линейных становятся кольцами.
    model.grow_kv_session(&mut session, 4096, max_new).expect("рост сессии");
    assert!(session.ctx_tokens() > 1536, "сессия доросла (кольца SWA)");
    assert_eq!(session.reusable(&ids2), h1, "рост сессии точку возврата не теряет");
    session.set_resume_at(history2);
    let mut cached2 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(4096, max_new));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let t = Instant::now();
    let reused = r
        .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
            cached2.push(id);
            true
        })
        .expect("ход 2");
    let t_cached = t.elapsed().as_millis();
    assert_eq!(reused, h1, "ход 2 должен взять из кэша весь промпт до истории");

    let mut fresh = model.new_kv_session(4096, max_new).expect("session").expect("префикс-KV");
    fresh.set_resume_at(history2);
    let mut fresh2 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(4096, max_new));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let t = Instant::now();
    let r0 = r
        .generate_streaming_cached(&mut fresh, &ids2, &tok, |id, _| {
            fresh2.push(id);
            true
        })
        .expect("ход 2 с нуля");
    let t_fresh = t.elapsed().as_millis();
    assert_eq!(r0, 0);
    let (_, ans2) = split_answer(&tok, &cached2);
    println!("ход 2: промпт {} ток., из кэша {reused}; {t_cached} мс с кэшем, {t_fresh} мс с нуля; ответ {ans2:?}", ids2.len());
    // Префилл от снимка даёт логиты бит в бит (ink2_resume), декод детерминирован.
    assert_eq!(cached2, fresh2, "ход 2 от точки возврата и с нуля разошлись");
    assert!(ans2.contains("400"), "{ans2}");

    // Ход 3 без границы службы: точка — на конце промпта 2, и ход 3, продолжающий
    // промпт 2 (а не ответ), берёт его целиком.
    let (_, answer2) = split_answer(&tok, &cached2);
    msgs.push(Message::assistant_turn(answer2, None, Vec::new()));
    msgs.push(Message::user("Спасибо!"));
    let (ids3, _) = render(&tok, &msgs);
    let reuse3 = session.reusable(&ids3);
    println!("ход 3: промпт {} ток., из кэша возьмётся {reuse3} (история хода 2 — {:?})", ids3.len(), history2);
    assert_eq!(reuse3, history2.unwrap_or(ids2.len()), "точка возврата хода 2 — на границе его истории");
}

/// Длинный промпт (~60K) как в службе: ход с размышлениями, затем ход с
/// историей без них. Из кэша — весь прошлый промпт до истории; служба сессию
/// не выбрасывает (`reusable * 2 >= len`); время хода до/после.
#[test]
fn inkling2_long_turn_resumes_without_reasoning() {
    let (Ok(path), Ok(corpus)) = (std::env::var("SYN_INKLING2_DIR"), std::env::var("SYN_INKLING2_LONG")) else {
        eprintln!("SYN_INKLING2_DIR / SYN_INKLING2_LONG не заданы — пропускаем");
        return;
    };
    let tokens: usize = std::env::var("SYN_INKLING2_LONG_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(60_000);
    // Окно — как у службы (`load_llm_full` с её окном), `SYN_INKLING2_LONG_CTX` (131072).
    let window: usize = std::env::var("SYN_INKLING2_LONG_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(131_072);
    let (model, tok) = load_llm_full(Path::new(&path), QuantPolicy::balance(), &Device::Cuda(0), Some(window), RuntimeOptions::default()).expect("load");
    let text = std::fs::read_to_string(&corpus).expect("корпус");
    let mut ids = tok.encode(&text).expect("encode");
    ids.truncate(tokens);
    let text = tok.decode(&ids).expect("decode");
    let max_new = 2048usize;

    let mut msgs = vec![Message::user(format!("Вот материалы:\n\n{text}\n\nНазови три темы, которые в них встречаются, коротко."))];
    let (ids1, history1) = render(&tok, &msgs);
    let h1 = history1.unwrap_or(ids1.len());
    // Сессия — как в службе: на ход с запасом, шаг 8K.
    let want = (ids1.len() + max_new + 16).div_ceil(8192) * 8192;
    let mut session = model.new_kv_session(want, max_new).expect("session").expect("префикс-KV");
    session.set_resume_at(history1);
    let mut out1 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(want, max_new));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let t = Instant::now();
    let mut first1 = None;
    let reused = r
        .generate_streaming_cached(&mut session, &ids1, &tok, |id, _| {
            first1.get_or_insert_with(|| t.elapsed().as_secs_f64());
            out1.push(id);
            true
        })
        .expect("ход 1");
    let t1 = t.elapsed().as_secs_f64();
    assert_eq!(reused, 0);
    let (reasoning, answer) = split_answer(&tok, &out1);
    println!(
        "ход 1: промпт {} ток. (точка {h1}), первый токен через {:.2} с, {} ток. ответа ({} симв. размышлений) за {t1:.1} с; ответ {:?}",
        ids1.len(),
        first1.unwrap_or(0.0),
        out1.len(),
        reasoning.len(),
        answer.chars().take(200).collect::<String>()
    );

    msgs.push(Message::assistant_turn(if answer.is_empty() { "Темы перечислены.".into() } else { answer }, None, Vec::new()));
    msgs.push(Message::user("Какая из них занимает больше всего места? Одним предложением."));
    let (ids2, history2) = render(&tok, &msgs);
    let reusable = session.reusable(&ids2);
    println!("ход 2: промпт {} ток., reusable {reusable}, ёмкость сессии {}", ids2.len(), session.ctx_tokens());
    assert_eq!(reusable, h1, "из кэша — весь прошлый промпт до истории");
    assert!(reusable * 2 >= ids2.len(), "служба выбросила бы сессию (reusable·2 < len)");
    // Ход длиннее сессии — служба доращивает её.
    let need = ids2.len() + max_new + 16;
    if session.ctx_tokens() < need {
        model.grow_kv_session(&mut session, need.div_ceil(8192) * 8192, max_new).expect("рост сессии");
    }
    session.set_resume_at(history2);
    let mut out2 = Vec::new();
    let mut r = LlmGeneration::new(&model, opts(session.ctx_tokens(), max_new));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let t = Instant::now();
    let mut first = None;
    let reused = r
        .generate_streaming_cached(&mut session, &ids2, &tok, |id, _| {
            first.get_or_insert_with(|| t.elapsed().as_secs_f64());
            out2.push(id);
            true
        })
        .expect("ход 2");
    let t2 = t.elapsed().as_secs_f64();
    assert_eq!(reused, h1);
    let (_, ans2) = split_answer(&tok, &out2);
    println!(
        "ход 2: из кэша {reused} из {}, первый токен через {:.2} с, {} ток. за {t2:.1} с; ответ {:?}",
        ids2.len(),
        first.unwrap_or(0.0),
        out2.len(),
        ans2.chars().take(200).collect::<String>()
    );

    // Как было до точки возврата: ход 2 — полный префилл. Граница чанка — на той же
    // точке, что у хода от снимка: число частей flash-внимания префилла зависит от
    // длины чанка (`attn_fa_splits`), и на 60K другая нарезка (хвост 158 строк против
    // 289) меняет логиты в младших разрядах (KL ~2e-10, ink2_resume) — greedy-ответ
    // через сотни токенов может уйти иначе. С той же нарезкой — бит в бит.
    drop(session);
    let mut fresh = model.new_kv_session(need.div_ceil(8192) * 8192, max_new).expect("session").expect("префикс-KV");
    fresh.set_resume_at(Some(h1));
    let mut r = LlmGeneration::new(&model, opts(fresh.ctx_tokens(), max_new));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let t = Instant::now();
    let mut first0 = None;
    let mut out0 = Vec::new();
    r.generate_streaming_cached(&mut fresh, &ids2, &tok, |id, _| {
        first0.get_or_insert_with(|| t.elapsed().as_secs_f64());
        out0.push(id);
        true
    })
    .expect("ход 2 с нуля");
    println!(
        "ход 2 с нуля (как раньше): первый токен через {:.2} с, всего {:.1} с; ответы совпали: {}",
        first0.unwrap_or(0.0),
        t.elapsed().as_secs_f64(),
        out0 == out2
    );
    assert_eq!(out0, out2, "ход 2 от точки возврата и с нуля разошлись");
}
