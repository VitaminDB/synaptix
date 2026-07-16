use clap::Parser;
use synaptix_debug::{compare::compare_dumps, load_from_file};

#[derive(Parser, Debug)]
#[command(name = "synaptix-diff", about = "cos_sim/L1/L2 между двумя TensorDump (.syndump)")]
struct Args {
    #[arg(value_name = "A")]
    a: std::path::PathBuf,
    #[arg(value_name = "B")]
    b: std::path::PathBuf,
}

fn main() {
    let args = Args::parse();
    let a = match load_from_file(&args.a) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error loading {}: {e}", args.a.display());
            std::process::exit(2);
        }
    };
    let b = match load_from_file(&args.b) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error loading {}: {e}", args.b.display());
            std::process::exit(2);
        }
    };
    if a.dims != b.dims {
        eprintln!("shape mismatch: {:?} vs {:?}", a.dims, b.dims);
        std::process::exit(2);
    }
    match compare_dumps(&a, &b) {
        Ok(r) => {
            println!(
                "numel={} dtype_a={:?} dtype_b={:?} cos_sim={:.10} max_abs={:.6e} rel_err={:.6e} l1={:.6e} l2={:.6e}",
                r.numel, a.dtype, b.dtype, r.cos_sim, r.max_abs, r.rel_err, r.l1, r.l2
            );
        }
        Err(e) => {
            eprintln!("compare failed: {e}");
            std::process::exit(2);
        }
    }
}
