//! Файлы HF-каталога из текстового кодера LTX-2.5 (Comfy-раскладка):
//! `config.json` — из метаданных `gemma_config`, `tokenizer.json` — из
//! тензора `tokenizer_json`, остальное — из `hf_asset__*`.
//! `ltx25_te_assets TE.safetensors OUT_DIR`
use std::path::Path;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader as _;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let l = SafetensorsLoader::open(Path::new(&a[0])).expect("safetensors");
    let out = Path::new(&a[1]);
    std::fs::create_dir_all(out).expect("dir");
    let cfg = l.metadata().get("gemma_config").expect("gemma_config");
    std::fs::write(out.join("config.json"), cfg).expect("config");
    for n in l.names() {
        let file = if n == "tokenizer_json" {
            "tokenizer.json".to_string()
        } else if let Some(f) = n.strip_prefix("hf_asset__") {
            f.to_string()
        } else {
            continue;
        };
        let (b, _, _) = l.raw_bytes(n).expect("bytes");
        std::fs::write(out.join(&file), b).expect("write");
        println!("{file}: {} байт", b.len());
    }
}
