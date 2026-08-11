use std::path::Path;

use synaptix::facade::llm::{
    load_llm_with_policy, Device, GenerationOptions, LlmGeneration, QuantPolicy,
};

const MODEL: &str = "/home/master/models/Qwen3.6-27B-Fable-Fusion-711-MTP.syn";

fn opts(max_seq_len: usize) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens: 24,
        max_seq_len,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 1234,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
    }
}

#[test]
#[ignore]
fn prefill_chunking_matches_full() {
    synaptix::facade::llm::set_mtp_enabled(false);
    let prompt = "Опиши подробно устройство HTTP-протокола: методы, заголовки, коды ответов, \
        отличия HTTP/1.1 от HTTP/2 и HTTP/3, механику keep-alive, чанковую передачу, \
        кэширование, условные запросы и согласование содержимого. "
        .repeat(6);

    synaptix::facade::llm::set_prefill_chunk_size(0);
    let (model, tok) =
        load_llm_with_policy(Path::new(MODEL), QuantPolicy::balance(), &Device::Cuda(0))
            .expect("load");
    let ids = tok.encode(&prompt).expect("encode");
    println!("prompt tokens = {}", ids.len());

    let mut full = Vec::new();
    let mut run1 = LlmGeneration::new(&model, opts(ids.len() + 64));
    run1.generate_streaming(&ids, &tok, |t, _s| {
        full.push(t);
        true
    })
    .expect("gen full");

    synaptix::facade::llm::set_prefill_chunk_size(64);
    let mut chunked = Vec::new();
    let mut run2 = LlmGeneration::new(&model, opts(ids.len() + 64));
    run2.generate_streaming(&ids, &tok, |t, _s| {
        chunked.push(t);
        true
    })
    .expect("gen chunked");

    println!("full    = {full:?}");
    println!("chunk64 = {chunked:?}");
    assert_eq!(
        full, chunked,
        "greedy-выход при чанкованном prefill отличается от цельного"
    );
}
