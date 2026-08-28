//! MoE-FFN на GPU: квантованные эксперты обязаны считать то же, что плотные.
//!
//! Здесь проверяется не арифметика роутинга (это делает `moe_ffn.rs` на CPU),
//! а то, что квант-путь `QLinear::Quant` подключён к экспертам правильно:
//! перепутанные gate/up или сбитая ведущая ось стопки дадут не «немного
//! другой» ответ, а качественно иной.

use std::collections::HashMap;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::ModelError;
use synaptix_llm_common::moe::{MoeConfig, MoeFfn};
use synaptix_llm_common::weights::WeightSource;

// Кратности под MXFP8 (K % 32 == 0) и NVFP4 (N, K % 64 == 0).
const H: usize = 128;
const I: usize = 64;
const E: usize = 4;
const K: usize = 2;
const T: usize = 7;

struct Weights {
    data: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl WeightSource for Weights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        let (v, shape) = self
            .data
            .get(key)
            .ok_or_else(|| ModelError::Load(format!("нет тензора {key}")))?;
        Tensor::from_vec::<_, f32>(v.clone(), shape.clone(), device)
            .and_then(|t| t.to_dtype(dtype))
            .map_err(|e| ModelError::Load(e.to_string()))
    }

    fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.4
        })
        .collect()
}

fn weights() -> Weights {
    let mut data = HashMap::new();
    data.insert("mlp.gate.weight".into(), (noise(1, E * H), vec![E, H]));
    data.insert(
        "mlp.experts.gate_up_proj".into(),
        (noise(2, E * 2 * I * H), vec![E, 2 * I, H]),
    );
    data.insert("mlp.experts.down_proj".into(), (noise(3, E * H * I), vec![E, H, I]));
    data.insert("mlp.shared_expert.gate_proj.weight".into(), (noise(4, I * H), vec![I, H]));
    data.insert("mlp.shared_expert.up_proj.weight".into(), (noise(5, I * H), vec![I, H]));
    data.insert("mlp.shared_expert.down_proj.weight".into(), (noise(6, H * I), vec![H, I]));
    data.insert("mlp.shared_expert_gate.weight".into(), (noise(7, H), vec![1, H]));
    Weights { data }
}

fn cfg() -> MoeConfig {
    MoeConfig {
        hidden_size: H,
        moe_intermediate_size: I,
        num_experts: E,
        num_experts_per_tok: K,
        shared_intermediate_size: I,
        norm_topk_prob: true,
        chunk: 4,
    }
}

fn setup() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

fn run(device: Device, compute: DType, quant: DType, x: &[f32]) -> Vec<f32> {
    let w = weights();
    let moe = MoeFfn::load(&w, "mlp", cfg(), device, compute, quant).expect("сборка MoE");
    let xt = Tensor::from_vec::<_, f32>(x.to_vec(), vec![T, H], device)
        .and_then(|t| t.to_dtype(compute))
        .unwrap();
    moe.forward(&xt)
        .expect("forward")
        .to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap()
}

/// Относительная L2 между двумя выходами.
fn l2_rel(got: &[f32], want: &[f32]) -> f32 {
    let num: f64 = got.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|v| (*v as f64).powi(2)).sum();
    (num / den.max(1e-12)).sqrt() as f32
}

#[test]
fn quantized_experts_match_dense_within_quant_noise() {
    if !setup() {
        return;
    }
    let x = noise(42, T * H);
    let dense_cpu = run(Device::Cpu, DType::F32, DType::F32, &x);
    let dense_gpu = run(Device::Cuda(0), DType::F16, DType::F16, &x);
    let mxfp8 = run(Device::Cuda(0), DType::F16, DType::MXFP8, &x);
    let nvfp4 = run(Device::Cuda(0), DType::F16, DType::NVFP4, &x);

    // Плотный GPU-путь отличается от CPU только точностью F16.
    assert!(l2_rel(&dense_gpu, &dense_cpu) < 0.02, "dense GPU: {}", l2_rel(&dense_gpu, &dense_cpu));
    // Квант шумит сильнее, но остаётся тем же ответом, а не другим.
    let e8 = l2_rel(&mxfp8, &dense_cpu);
    let e4 = l2_rel(&nvfp4, &dense_cpu);
    println!("l2_rel: mxfp8={e8} nvfp4={e4}");
    assert!(e8 < 0.10, "MXFP8 разошёлся: {e8}");
    // На случайных весах FP4 — худший случай: 4 бита мантиссы на блок из 16.
    assert!(e4 < 0.30, "NVFP4 разошёлся: {e4}");
}

/// Разбиение на чанки меняет только то, сколько строк уходит в GEMM за раз.
/// Совпадения бит в бит тут не будет — при разном числе строк у эксперта
/// работают разные ядра (GEMV на одной строке, GEMM на нескольких), — поэтому
/// проверяется главное: и целиком, и по кускам ответ остаётся тем же с
/// точностью до квант-шума. Перепутанные при перестановке токены дали бы
/// расхождение в разы, а не в проценты.
#[test]
fn chunking_keeps_the_answer_within_quant_noise() {
    if !setup() {
        return;
    }
    let x = noise(11, T * H);
    let w = weights();
    let device = Device::Cuda(0);
    let xt = Tensor::from_vec::<_, f32>(x.clone(), vec![T, H], device)
        .and_then(|t| t.to_dtype(DType::F16))
        .unwrap();
    let out = |chunk: usize, quant: DType| -> Vec<f32> {
        let mut c = cfg();
        c.chunk = chunk;
        MoeFfn::load(&w, "mlp", c, device, DType::F16, quant)
            .expect("сборка")
            .forward(&xt)
            .expect("forward")
            .to_device(Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .unwrap()
    };

    let reference = {
        let moe = MoeFfn::load(&w, "mlp", cfg(), Device::Cpu, DType::F32, DType::F32).unwrap();
        let xc = Tensor::from_vec::<_, f32>(x.clone(), vec![T, H], Device::Cpu).unwrap();
        moe.forward(&xc).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };

    // Плотный путь: чанк вообще не должен влиять на численность заметно.
    let dense_whole = out(T, DType::F16);
    let dense_split = out(2, DType::F16);
    assert!(l2_rel(&dense_split, &dense_whole) < 0.02, "dense: чанк изменил выход");

    // Квант: оба варианта — тот же ответ в пределах шума кванта.
    for chunk in [T, 3, 2, 1] {
        let got = out(chunk, DType::MXFP8);
        let err = l2_rel(&got, &reference);
        println!("chunk={chunk} l2_rel={err}");
        assert!(err < 0.10, "chunk={chunk}: разошлось с эталоном, L2={err}");
    }
}
