//! Сверка частей MiniMax Music 3 с эталоном: каждая часть считается на
//! заданном входе `a·sin(w·i + p)` (i — плоский индекс, f64 → f32), выходы —
//! сырыми f32 LE в каталог. `music3_check BUNDLE.syn OUT_DIR`
use std::path::Path;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_minimax3::{ar::Ar, flow::{ConditionEncoder, FlowTransformer}, loader::Comp, vocoder::Vocoder, Music3};

fn wave(shape: Vec<usize>, a: f64, w: f64, p: f64, dev: Device) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|i| (a * (w * i as f64 + p).sin()) as f32).collect();
    Tensor::from_vec(v, shape, dev).unwrap()
}

fn dump(dir: &Path, name: &str, t: &Tensor) {
    let v = t.contiguous().unwrap().to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let dims: Vec<String> = t.dims().iter().map(|d| d.to_string()).collect();
    std::fs::write(dir.join(format!("{name}.{}.f32", dims.join("x"))), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    println!("{name}: {:?}", t.dims());
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let (bundle, dir) = (Path::new(&a[0]), Path::new(&a[1]));
    std::fs::create_dir_all(dir).unwrap();
    synaptix_core::grad::no_grad(|| {
        // 1. Токены промпта.
        let m = Music3::open(bundle, dev).unwrap();
        let (ids, _) = m.prompt_ids("# Style\n**Genre**: acoustic pop <|bpm 96|>\n- soft *female* lead", "[Verse] x\nMorning light filtering through the pine\n[Chorus]\nSoftly the world begins to breathe").unwrap();
        std::fs::write(dir.join("prompt_ids.txt"), ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" ")).unwrap();
        println!("prompt_ids: {}", ids.len());
        // 2. Вокодер: латенты [1,128,40].
        let voc = Vocoder::load(&Comp::open(bundle, "vocoder", dev).unwrap()).unwrap();
        dump(dir, "vocoder", &voc.decode(&wave(vec![1, 128, 40], 0.5, 0.37, 0.0, dev)).unwrap());
        drop(voc);
        // 3. Кодер условий: кадры [1,10,8·4096].
        let enc = ConditionEncoder::load(&Comp::open(bundle, "condition_encoder", dev).unwrap(), dev).unwrap();
        dump(dir, "condition", &enc.forward(&wave(vec![1, 10, 8 * 4096], 0.5, 0.013, 0.0, dev)).unwrap());
        // 4. Трансформер: lat [50,128] (токен-мажор), cond [50,2048], t = 0.4.
        let tr = FlowTransformer::load(&Comp::open(bundle, "transformer", dev).unwrap(), dev).unwrap();
        let rope = tr.rope(51).unwrap();
        dump(dir, "velocity", &tr.forward(&wave(vec![50, 128], 0.5, 0.11, 0.0, dev), &wave(vec![50, 2048], 0.3, 0.07, 1.0, dev), 0.4, &rope).unwrap());
        drop(tr);
        // 5. Глубинный декодер: seq [2,5,4096]; 6. LM: логиты последней позиции промпта.
        let ar = Ar::open(bundle, dev, None).unwrap();
        dump(dir, "depth", &ar.depth.forward(&wave(vec![2, 5, 4096], 0.5, 0.017, 0.0, dev).to_dtype(DType::BF16).unwrap()).unwrap());
        let mut kv = ar.model.make_kv_cache(1, 256).unwrap();
        let t = Tensor::from_vec(ids.clone(), vec![1usize, ids.len()], dev).unwrap();
        let hid = ar.model.forward_trunk(&t, &mut kv).unwrap();
        let last = ar.model.normed_at(&hid, ids.len() - 1).unwrap().contiguous().unwrap();
        dump(dir, "lm_last_hidden", &last);
        dump(dir, "lm_logits_codes", &ar.model.lm_head_forward(&last).unwrap().narrow(1, 151_670, 16_384 + 5).unwrap().contiguous().unwrap());
    });
}
