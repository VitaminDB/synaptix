//! Префикс-KV Qwen4Exp с картинкой в истории: ход, продолжающий диалог с
//! вложением, обязан переиспользовать посчитанный префикс, а подмена
//! картинки при тех же токенах — сбросить его.
//!
//! До 11.09.2026 медиа-промпт выключал сессию целиком (и в движке, и в
//! synthos): чат с одной картинкой в истории платил полным префиллом всей
//! истории на каждом ходу (55k токенов — 40 с на ход).
//!
//! Запуск: `SYN_QWEN4EXP_BUNDLE=… SYN_QWEN4EXP_IMAGE=/path/to.png
//! SYN_QWEN4EXP_EXPERT_CACHE_GB=6 cargo test -p synaptix --release --test
//! qwen4exp_prefix_kv_media -- --nocapture --test-threads=1`.

use std::path::Path;

use synaptix::facade::llm::{
    load_llm_with_policy, optimal_profile, Device, GenerationOptions, LlmGeneration,
    MediaEmbedding, Message,
};

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

#[test]
fn qwen4exp_prefix_kv_survives_media() {
    let (Ok(path), Ok(image)) = (
        std::env::var("SYN_QWEN4EXP_BUNDLE"),
        std::env::var("SYN_QWEN4EXP_IMAGE"),
    ) else {
        eprintln!("SYN_QWEN4EXP_BUNDLE / SYN_QWEN4EXP_IMAGE не заданы — пропускаем");
        return;
    };
    synaptix::facade::llm::cuda_release_kernel_caches();
    synaptix::facade::llm::cuda_trim_pool(0);
    let policy = optimal_profile(Path::new(&path)).policy;
    let (model, tok) =
        load_llm_with_policy(Path::new(&path), policy, &Device::Cuda(0)).expect("load");
    assert!(model.supports_media(), "бандл без башни зрения");
    assert!(model.ensure_media_tower().expect("башня"), "башня не загрузилась");
    let media = model.encode_image(Path::new(&image), Some(512)).expect("encode_image");
    model.release_media_tower();
    eprintln!("картинка: {} vision-токенов", media.tokens());

    let msgs = [
        Message::system("Отвечай кратко, одним предложением."),
        Message::user(&format!("{}Что изображено на картинке?", media.prompt_block)),
    ];
    let prompt1 = tok
        .apply_chat_template_ex_tools(&msgs, true, false, None)
        .expect("chat template");
    let ids1 = tok.encode(&prompt1).expect("encode");

    let ctx = 8192usize;
    let max_new = 48usize;
    let mut session = model
        .new_kv_session(ctx, max_new)
        .expect("session")
        .expect("Qwen4Exp умеет префикс-KV");

    let run = |session: &mut _, ids: &[u32], media: &MediaEmbedding| -> (Vec<u32>, String, usize) {
        let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
        r.set_stop_tokens(tok.eos_ids().to_vec());
        let mut got = Vec::new();
        let mut text = String::new();
        let reused = r
            .generate_streaming_cached_media(session, ids, &tok, &[media], |id, s| {
                got.push(id);
                text.push_str(s);
                true
            })
            .expect("turn");
        (got, text, reused)
    };

    // Ход 1: полный префилл, точка возврата на конце промпта.
    let (a1, t1, reused1) = run(&mut session, &ids1, &media);
    eprintln!("ход 1: промпт {} ток, из кэша {reused1} → {t1:?}", ids1.len());
    assert_eq!(reused1, 0);
    assert!(!t1.trim().is_empty(), "модель ничего не сказала");

    // Ход 2: та же история + ответ + новый вопрос — префикс обязан
    // переиспользоваться целиком.
    let mut ids2 = ids1.clone();
    ids2.extend_from_slice(&a1);
    ids2.extend(
        tok.encode("<|im_end|>\n<|im_start|>user\nНазови главный цвет на ней одним словом.<|im_end|>\n<|im_start|>assistant\n")
            .expect("encode tail"),
    );
    let (a2, t2, reused2) = run(&mut session, &ids2, &media);
    eprintln!("ход 2: промпт {} ток, из кэша {reused2} → {t2:?}", ids2.len());
    assert_eq!(reused2, ids1.len(), "ход с картинкой в истории не переиспользовал префикс");
    assert!(!t2.trim().is_empty());

    // Эталон хода 2 без сессии: полный префилл по тем же эмбеддингам.
    let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
    r.set_stop_tokens(tok.eos_ids().to_vec());
    let mut a2_ref = Vec::new();
    let mut t2_ref = String::new();
    r.generate_streaming_media(&ids2, &tok, &[&media], |id, s| {
        a2_ref.push(id);
        t2_ref.push_str(s);
        true
    })
    .expect("fresh turn");
    let diff = a2.iter().zip(&a2_ref).position(|(x, y)| x != y);
    eprintln!("эталон без сессии → {t2_ref:?}; первое расхождение: {diff:?}");
    // Стек не бит-детерминирован, но начало ответа обязано совпасть.
    assert!(
        diff.map_or(true, |d| d >= 4),
        "кэшированный ход разошёлся с полным префиллом с токена {diff:?}"
    );

    // Ход 3: те же токены, но другая картинка на месте заполнителей —
    // префикс недействителен, кэш обязан это заметить.
    let other = MediaEmbedding {
        kind: media.kind,
        embeds: media.embeds.mul_scalar(0.5).expect("scale"),
        tokens: media.tokens,
        prompt_block: media.prompt_block.clone(),
        grid_hw: media.grid_hw,
        blocks: media.blocks,
    };
    let (_, t3, reused3) = run(&mut session, &ids2, &other);
    eprintln!("ход 3 (другая картинка): из кэша {reused3} → {t3:?}");
    assert_eq!(reused3, 0, "подменённая картинка не сбросила префикс");
}
