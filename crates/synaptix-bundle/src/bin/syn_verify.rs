//! syn-verify — walk every alive chunk and verify CRC32C.
//! With `--strict` and the `sha256` / `blake3` features: also verifies
//! per-chunk cryptographic hashes and the bundle's manifest hash-of-hashes.

use std::path::PathBuf;

use synaptix_bundle::Bundle;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("usage: syn-verify <bundle.syn> [--strict]");
        return Ok(());
    }

    let mut bundle: Option<PathBuf> = None;
    let mut strict = false;
    for a in args {
        match a.as_str() {
            "--strict" => strict = true,
            _ if bundle.is_none() => bundle = Some(PathBuf::from(a)),
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    let bundle = bundle.ok_or_else(|| anyhow::anyhow!("missing <bundle.syn>"))?;

    let b = Bundle::open(&bundle)?;
    let total = b.cdir().entries.iter().filter(|e| e.is_alive()).count();
    b.verify_full()?;
    eprintln!("ok: {total} alive chunks, all CRC32C match");

    if strict {
        verify_strict(&b)?;
    }
    Ok(())
}

#[cfg_attr(
    not(any(feature = "sha256", feature = "blake3")),
    allow(unused_variables, unused_mut)
)]
fn verify_strict(b: &Bundle) -> anyhow::Result<()> {
    let mut ran_any = false;

    #[cfg(feature = "blake3")]
    {
        let t = std::time::Instant::now();
        match b.verify_blake3() {
            Ok(true) => {
                ran_any = true;
                eprintln!("strict: Blake3 per-chunk + manifest verified ({:.2}s)", t.elapsed().as_secs_f64());
            }
            Ok(false) => {}
            Err(e) => anyhow::bail!("strict (blake3) failed: {e}"),
        }
    }

    #[cfg(feature = "sha256")]
    {
        let t = std::time::Instant::now();
        match b.verify_sha256() {
            Ok(true) => {
                ran_any = true;
                eprintln!("strict: SHA-256 per-chunk + manifest verified ({:.2}s)", t.elapsed().as_secs_f64());
            }
            Ok(false) => {}
            Err(e) => anyhow::bail!("strict (sha256) failed: {e}"),
        }
    }

    if !ran_any {
        #[cfg(not(any(feature = "sha256", feature = "blake3")))]
        anyhow::bail!("--strict requires syn-verify built with `--features sha256` or `--features blake3`");
        #[cfg(any(feature = "sha256", feature = "blake3"))]
        eprintln!("strict: bundle has no manifest_sha256/manifest_blake3 (built without --sha256/--blake3); skipping");
    }
    Ok(())
}
