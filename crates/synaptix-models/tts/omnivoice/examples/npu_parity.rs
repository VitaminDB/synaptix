use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_omnivoice::OmniVoicePipeline;

fn host(t: synaptix_core::tensor::Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = &args[1];
    let text = args.get(2).map(String::as_str).unwrap_or("Привет! Это проверка синтеза речи на нейропроцессоре.");
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_npu::register().expect("NPU");
    let cpu = OmniVoicePipeline::from_syn(model, Device::Cpu, DType::F32).unwrap();
    let npu = OmniVoicePipeline::from_syn(model, Device::Npu(0), DType::F16).unwrap();
    let (ids, mask, c) = cpu.prepare(text, Some("ru")).unwrap();
    let sb = npu.backbone().bucket_for(c).unwrap_or(c);
    let hd = 1024;
    let layers: Vec<usize> = std::env::var("LAYERS").ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or(vec![1, 2, 4, 8, 14, 20, 28]);
    for n in layers {
        let a = host(cpu.backbone().forward_padded_upto(&[(&ids, &mask, c)], sb, Some(n)).unwrap());
        let b = host(npu.backbone().forward_padded_upto(&[(&ids, &mask, c)], sb, Some(n)).unwrap());
        let (mut num, mut den, mut amax, mut bmax, mut nan) = (0f64, 0f64, 0f32, 0f32, 0usize);
        for i in 0..c * hd {
            let (x, y) = (a[i], b[i]);
            if !y.is_finite() {
                nan += 1;
                continue;
            }
            num += ((x - y) as f64).powi(2);
            den += (x as f64).powi(2);
            amax = amax.max(x.abs());
            bmax = bmax.max(y.abs());
        }
        println!("блоков {n:>2}: отн. ошибка {:.4}, max|cpu| {amax:.1}, max|npu| {bmax:.1}, не-чисел {nan}", (num / den.max(1e-12)).sqrt());
    }
    let t = c * 3 / 4;
    let uid: Vec<i64> = (0..8).flat_map(|cb| ids[cb * c + (c - t)..cb * c + c].to_vec()).collect();
    let um: Vec<u8> = mask[c - t..].to_vec();
    let pair = std::env::var_os("PAIR").is_some();
    let seqs: Vec<(&[i64], &[u8], usize)> = if pair { vec![(&ids, &mask, c), (&uid, &um, t)] } else { vec![(&ids, &mask, c)] };
    let a = host(cpu.backbone().forward_padded(&seqs, sb).unwrap());
    let b = host(npu.backbone().forward_padded(&seqs, sb).unwrap());
    let v = 1025;
    let (mut agree, mut rows, mut nan) = (0usize, 0usize, 0usize);
    for cb in 0..8 {
        for j in 0..c {
            let o = (cb * sb + j) * v;
            let am = |r: &[f32]| r.iter().enumerate().fold((0, f32::NEG_INFINITY), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0;
            if b[o..o + v].iter().any(|x| !x.is_finite()) {
                nan += 1;
            }
            agree += (am(&a[o..o + v]) == am(&b[o..o + v])) as usize;
            rows += 1;
        }
    }
    println!("логиты: argmax совпал {:.1}% ({rows} строк), строк с не-числами {nan}; cpu[0..4] {:?} npu[0..4] {:?}", 100.0 * agree as f64 / rows as f64, &a[..4], &b[..4]);
}
