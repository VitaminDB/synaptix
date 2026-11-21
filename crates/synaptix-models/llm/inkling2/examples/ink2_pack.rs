//! Упаковка HF-каталога в `.syn`:
//! `ink2_pack <hf-dir> <out.syn> [--mxfp8-experts keep|nvfp4] [--attn keep|nvfp4|mxfp8] [--head keep|nvfp4|mxfp8]`.

use synaptix_llm_inkling2::pack::{pack, Fmt, PackOptions};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let mut opts = PackOptions { mxfp8_experts: Fmt::Keep, attn: Fmt::Keep, head: Fmt::Keep };
    let mut i = 3;
    while i + 1 < a.len() {
        let f = Fmt::parse(&a[i + 1])?;
        match a[i].as_str() {
            "--mxfp8-experts" => opts.mxfp8_experts = f,
            "--attn" => opts.attn = f,
            "--head" => opts.head = f,
            o => return Err(format!("ключ {o}")),
        }
        i += 2;
    }
    synaptix_kernels_cpu::ensure_registered();
    let t = std::time::Instant::now();
    pack(std::path::Path::new(&a[1]), std::path::Path::new(&a[2]), opts)?;
    let size = std::fs::metadata(&a[2]).map(|m| m.len()).unwrap_or(0);
    println!("{} — {:.2} ГБ за {:.0} с ({opts:?})", a[2], size as f64 / 1e9, t.elapsed().as_secs_f64());
    Ok(())
}
