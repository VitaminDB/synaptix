use clap::Parser;
use synaptix_debug::compare::dump_to_f64;
use synaptix_debug::{load_from_file, nan_detector::scan_finite};

#[derive(Parser, Debug)]
#[command(name = "synaptix-dump-inspect", about = "Показать структуру TensorDump (.syndump)")]
struct Args {
    #[arg(value_name = "FILE")]
    file: std::path::PathBuf,

    #[arg(long, default_value_t = 8)]
    head: usize,
}

fn main() {
    let args = Args::parse();
    let dump = match load_from_file(&args.file) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    println!("name:  {}", dump.name);
    println!("dtype: {:?}", dump.dtype);
    println!("dims:  {:?} (numel={})", dump.dims, dump.numel());
    let data = match dump_to_f64(&dump) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("decode failed: {e}");
            std::process::exit(2);
        }
    };
    let mut nan = 0usize;
    let mut pinf = 0usize;
    let mut ninf = 0usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &x in &data {
        if x.is_nan() {
            nan += 1;
            continue;
        }
        if x == f64::INFINITY {
            pinf += 1;
            continue;
        }
        if x == f64::NEG_INFINITY {
            ninf += 1;
            continue;
        }
        sum += x;
        sum_sq += x * x;
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    let n = (data.len() - nan - pinf - ninf) as f64;
    let mean = if n > 0.0 { sum / n } else { 0.0 };
    let var = if n > 0.0 { sum_sq / n - mean * mean } else { 0.0 };
    println!("min:   {min:?}");
    println!("max:   {max:?}");
    println!("mean:  {mean:?}");
    println!("std:   {:?}", var.max(0.0).sqrt());
    println!("nan={nan} +inf={pinf} -inf={ninf}");
    let h = args.head.min(data.len());
    if h > 0 {
        println!("head[{h}]: {:?}", &data[..h]);
    }
    let _ = scan_finite;
}
