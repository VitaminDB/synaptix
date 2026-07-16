//! Bit-exact-проверка нативного CLIP-токенайзера против HF `CLIPTokenizer`.
//!
//! Reference: `scripts/reference/gen_sdxl_clip.py` дампит `input_ids [1,77]`
//! (i32) для фиксированного промпта — той же токенизацией, что кормит энкодеры.
//! Здесь грузим vocab.json/merges.txt SDXL и сверяем, что наш `encode` даёт
//! ровно те же id (bos + BPE + eos, паддинг eos до 77). Пропускается без весов.

use std::path::Path;

use synaptix_image_sdxl::ClipTokenizer;

const SDXL: &str = "models/stabilityai/stable-diffusion-xl-base-1.0";
const PROMPT: &str = "a photograph of an astronaut riding a horse";

fn run_case(case: &str, tok_subdir: &str) {
    let dir = format!("{SDXL}/{tok_subdir}");
    if !Path::new(&dir).exists() {
        eprintln!("SKIP {case}: нет токенайзера {dir}");
        return;
    }
    let ref_path =
        synaptix_test_utils::reference_data_path("sdxl_clip", &format!("{case}.safetensors"));
    if !ref_path.exists() {
        eprintln!("SKIP {case}: нет reference {ref_path:?} (запусти gen_sdxl_clip.py)");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);
    let expected: Vec<u32> = refs["input_ids"].to_vec2::<i32>().unwrap()[0]
        .iter()
        .map(|&v| v as u32)
        .collect();

    let tok = ClipTokenizer::from_dir(&dir).unwrap();
    let got = tok.encode(PROMPT, 77);

    eprintln!("{case}: got[..12]={:?}", &got[..12.min(got.len())]);
    assert_eq!(got.len(), expected.len(), "{case}: длина {} != {}", got.len(), expected.len());
    assert_eq!(got, expected, "{case}: токенизация не bit-exact к HF");
}

#[test]
fn clip_l_tokenizer_bit_exact() {
    run_case("clip_l", "tokenizer");
}

#[test]
fn clip_bigg_tokenizer_bit_exact() {
    run_case("clip_bigg", "tokenizer_2");
}
