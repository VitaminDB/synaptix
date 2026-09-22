//! Сверка квант-путей эмбеддинга GGUF (gather и связанная голова) с плотным.
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::SynBundleLoader;

fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu).unwrap().to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn main() {
    synaptix::init().unwrap();
    let path = std::env::args().nth(1).unwrap();
    let src = SynBundleLoader::open(std::path::Path::new(&path)).unwrap();
    let src = src.gguf().unwrap();
    let name = "model.embed_tokens.weight";
    let dev = Device::Cuda(0);
    let qw = src.load_quant(name, dev).unwrap().unwrap();
    let (n, k) = (qw.n(), qw.k());
    println!("embed {:?} n={n} k={k} row_bytes={:?}", qw.dtype(), qw.block_row_bytes());
    let dense = src.load_to(name, Device::Cpu, DType::F32).unwrap();
    let dv = to_vec(&dense);
    let ids: Vec<u32> = vec![818, 5279, 2, 0, (n - 1) as u32, 106];
    let ids_t = Tensor::from_vec(ids.clone(), vec![ids.len()], dev).unwrap();
    let g = qw.embed_gather(&ids_t).unwrap();
    let gv = to_vec(&g);
    for (i, id) in ids.iter().enumerate() {
        let want = &dv[*id as usize * k..(*id as usize + 1) * k];
        let got = &gv[i * k..(i + 1) * k];
        let maxd = want.iter().zip(got).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("gather id {id}: maxdiff {maxd:.5} want {:?} got {:?}", &want[..4], &got[..4]);
    }
    // Голова: x [1, k] случайная → logits [1, n] через квант против плотного.
    let x: Vec<f32> = (0..k).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * 0.1).collect();
    let xt = Tensor::from_vec(x.clone(), vec![1, k], dev).unwrap();
    for dt in [DType::F16, DType::BF16] {
        let xq = xt.to_dtype(dt).unwrap();
        let y = xq.linear_quant(&qw).unwrap();
        let yv = to_vec(&y);
        let mut maxd = 0f32;
        let mut worst = 0usize;
        for row in [0usize, 1, 818, 5279, n / 2, n - 1] {
            let want: f32 = dv[row * k..(row + 1) * k].iter().zip(&x).map(|(a, b)| a * b).sum();
            let d = (want - yv[row]).abs();
            if d > maxd { maxd = d; worst = row; }
            println!("head {dt:?} row {row}: want {want:.5} got {:.5}", yv[row]);
        }
        println!("head {dt:?}: len {} maxdiff {maxd:.5} (row {worst})", yv.len());
    }
}
