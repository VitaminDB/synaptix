//! Проверка операций постоянной формы на NPU против CPU: `npu_ops [op]`
//! (op: rope | append | attn | all), T строк = `$T` (по умолчанию 32).
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn rnd(shape: Vec<usize>, seed: u64) -> Tensor {
    Tensor::randn_seeded(shape, seed, Device::Cpu).unwrap().to_dtype(DType::F16).unwrap()
}

fn cmp(name: &str, a: &Tensor, b: &Tensor) {
    let a = a.to_device(Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.to_device(Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        d += ((x - y) as f64).powi(2);
        n += (*x as f64).powi(2);
    }
    println!("{name}: отн. ошибка {:.2e}", (d / n.max(1e-30)).sqrt());
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_npu::register().expect("NPU");
    let op = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    let t: usize = std::env::var("T").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let (cap, nkv, nh, hd, start) = (1024usize, 2usize, 16usize, 128usize, 37usize);
    let npu = Device::Npu(0);
    let pos_c = Tensor::from_vec(vec![start as u32], vec![1usize], Device::Cpu).unwrap();
    let pos_n = pos_c.to_device(npu).unwrap();
    let tc_c = Tensor::from_vec(vec![(start + t) as u32], vec![1usize], Device::Cpu).unwrap();
    let tc_n = tc_c.to_device(npu).unwrap();
    if op == "rope" || op == "all" {
        let x = rnd(vec![1, nh, t, hd], 1);
        let cos = rnd(vec![4096, hd], 2);
        let sin = rnd(vec![4096, hd], 3);
        let c = x.rope_apply_dev(&cos, &sin, &pos_c, hd).unwrap();
        let n = x.to_device(npu).unwrap().rope_apply_dev(&cos.to_device(npu).unwrap(), &sin.to_device(npu).unwrap(), &pos_n, hd).unwrap();
        cmp("rope", &c, &n);
    }
    if op == "append" || op == "all" {
        let buf = rnd(vec![1, nkv, cap, hd], 4);
        let k = rnd(vec![1, nkv, t, hd], 5);
        let mut bc = rnd(vec![1, nkv, cap, hd], 4);
        bc.kv_append_dev(&k, &pos_c).unwrap();
        let mut bn = buf.to_device(npu).unwrap();
        bn.kv_append_dev(&k.to_device(npu).unwrap(), &pos_n).unwrap();
        cmp("append", &bc, &bn);
    }
    if op == "attn" || op == "all" {
        let q = rnd(vec![1, nh, t, hd], 6);
        let k = rnd(vec![1, nkv, cap, hd], 7);
        let v = rnd(vec![1, nkv, cap, hd], 8);
        let c = q.flash_attention_dev(&k, &v, &tc_c, 0.088, true).unwrap();
        let n = q.to_device(npu).unwrap().flash_attention_dev(&k.to_device(npu).unwrap(), &v.to_device(npu).unwrap(), &tc_n, 0.088, true).unwrap();
        cmp("attn", &c, &n);
    }
}
