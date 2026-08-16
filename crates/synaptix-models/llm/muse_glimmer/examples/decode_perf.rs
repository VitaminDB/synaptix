use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_muse_glimmer::MusePipeline;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bundle = args.first().expect("usage: decode_perf <bundle.syn> [ctx] [n_decode]");
    let ctx: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1024);
    let n_decode: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);
    let max_seq = (ctx + n_decode + 8).max(2048);
    let t0 = std::time::Instant::now();
    let p = MusePipeline::load_with_precision(bundle, device, PrecisionConfig::nvfp4(), Some(max_seq))
        .expect("load");
    eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let mut kv = p.model.make_kv_cache(1, max_seq).expect("kv");
    let ids: Vec<u32> = (0..ctx as u32).map(|i| 2000 + (i * 37) % 50000).collect();
    let t0 = std::time::Instant::now();
    let chunk = 1024;
    let mut off = 0;
    let mut logits = None;
    while off < ctx {
        let end = (off + chunk).min(ctx);
        let t = Tensor::from_vec(ids[off..end].to_vec(), vec![1usize, end - off], device).unwrap();
        logits = Some(no_grad(|| p.model.forward(&t, &mut kv)).expect("prefill"));
        off = end;
    }
    let prefill_s = t0.elapsed().as_secs_f32();
    eprintln!("prefill {ctx} tok in {prefill_s:.2}s ({:.0} tok/s)", ctx as f32 / prefill_s);

    let mut cur = 2025u32;
    let _ = logits;
    for warm in 0..3 {
        let t = Tensor::from_vec(vec![cur + warm], vec![1usize, 1], device).unwrap();
        let _ = no_grad(|| p.model.forward(&t, &mut kv)).expect("warmup");
    }

    let mut step = |cur: &mut u32, i: u32| {
        let t = Tensor::from_vec(vec![*cur], vec![1usize, 1], device).unwrap();
        let lg = no_grad(|| p.model.forward(&t, &mut kv)).expect("decode");
        let v = lg.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        *cur = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(2025 + i);
    };

    let t0 = std::time::Instant::now();
    for i in 0..n_decode as u32 {
        step(&mut cur, i);
    }
    let dt = t0.elapsed().as_secs_f32();
    eprintln!(
        "clean decode {n_decode} tok @ctx~{ctx}: {dt:.2}s ({:.2} tok/s, {:.2} ms/tok)",
        n_decode as f32 / dt,
        dt * 1000.0 / n_decode as f32
    );

    synaptix_llm_common::model::set_decode_prof(true);
    for i in 0..(n_decode as u32).min(24) {
        step(&mut cur, i);
    }
    synaptix_llm_common::model::set_decode_prof(false);
    eprintln!("{}", synaptix_llm_common::model::decode_prof_report_and_clear());
}
