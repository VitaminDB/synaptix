//! MoE-FFN Qwen-раскладки против прямого расчёта по формуле.
//!
//! Эталон здесь считается независимо от движка — на `f32` в тесте, циклами,
//! ровно как записано в `Qwen2MoeSparseMoeBlock`. Совпадение с ним и есть
//! проверка: маршрутизация, сортировка токенов по экспертам, обратная
//! перестановка и shared expert.

use std::collections::HashMap;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::ModelError;
use synaptix_llm_common::moe::{MoeConfig, MoeFfn};
use synaptix_llm_common::weights::WeightSource;

const H: usize = 8;
const I: usize = 6;
const E: usize = 4;
const K: usize = 2;
const T: usize = 5;

/// Источник плотных весов в памяти.
struct Weights {
    data: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Weights {
    fn get(&self, key: &str) -> Option<&(Vec<f32>, Vec<usize>)> {
        self.data.get(key)
    }
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

/// Детерминированный «шум» — свой у каждого тензора.
fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn weights(with_shared: bool) -> Weights {
    let mut data = HashMap::new();
    data.insert("mlp.gate.weight".to_string(), (noise(1, E * H), vec![E, H]));
    data.insert(
        "mlp.experts.gate_up_proj".to_string(),
        (noise(2, E * 2 * I * H), vec![E, 2 * I, H]),
    );
    data.insert(
        "mlp.experts.down_proj".to_string(),
        (noise(3, E * H * I), vec![E, H, I]),
    );
    if with_shared {
        data.insert(
            "mlp.shared_expert.gate_proj.weight".to_string(),
            (noise(4, I * H), vec![I, H]),
        );
        data.insert(
            "mlp.shared_expert.up_proj.weight".to_string(),
            (noise(5, I * H), vec![I, H]),
        );
        data.insert(
            "mlp.shared_expert.down_proj.weight".to_string(),
            (noise(6, H * I), vec![H, I]),
        );
        data.insert(
            "mlp.shared_expert_gate.weight".to_string(),
            (noise(7, H), vec![1, H]),
        );
    }
    Weights { data }
}

fn cfg(with_shared: bool, norm_topk_prob: bool) -> MoeConfig {
    MoeConfig {
        hidden_size: H,
        moe_intermediate_size: I,
        num_experts: E,
        num_experts_per_tok: K,
        shared_intermediate_size: if with_shared { I } else { 0 },
        norm_topk_prob,
        chunk: 3, // меньше T — заодно проверяется склейка чанков
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `y = W · x`, `W:[out, in]`.
fn matvec(w: &[f32], x: &[f32], out: usize, inp: usize) -> Vec<f32> {
    (0..out)
        .map(|o| (0..inp).map(|i| w[o * inp + i] * x[i]).sum())
        .collect()
}

/// Прямой расчёт блока по формуле Qwen-MoE.
fn reference(w: &Weights, x: &[f32], with_shared: bool, norm_topk_prob: bool) -> Vec<f32> {
    let gate = &w.get("mlp.gate.weight").unwrap().0;
    let gate_up = &w.get("mlp.experts.gate_up_proj").unwrap().0;
    let down = &w.get("mlp.experts.down_proj").unwrap().0;
    let mut out = vec![0f32; T * H];

    for t in 0..T {
        let xt = &x[t * H..(t + 1) * H];
        let logits = matvec(gate, xt, E, H);

        let mut order: Vec<usize> = (0..E).collect();
        order.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
        let top = &order[..K];
        let max = top.iter().map(|j| logits[*j]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|j| (logits[*j] - max).exp()).collect();
        let denom: f32 = if norm_topk_prob {
            exps.iter().sum()
        } else {
            logits.iter().map(|v| (v - max).exp()).sum()
        };

        for (slot, expert) in top.iter().enumerate() {
            let gu = &gate_up[expert * 2 * I * H..(expert + 1) * 2 * I * H];
            let dn = &down[expert * H * I..(expert + 1) * H * I];
            let gu = matvec(gu, xt, 2 * I, H);
            let h: Vec<f32> = (0..I).map(|i| silu(gu[i]) * gu[I + i]).collect();
            let y = matvec(dn, &h, H, I);
            let weight = exps[slot] / denom;
            for (o, v) in y.iter().enumerate() {
                out[t * H + o] += weight * v;
            }
        }

        if with_shared {
            let sg = &w.get("mlp.shared_expert.gate_proj.weight").unwrap().0;
            let su = &w.get("mlp.shared_expert.up_proj.weight").unwrap().0;
            let sd = &w.get("mlp.shared_expert.down_proj.weight").unwrap().0;
            let sgate = &w.get("mlp.shared_expert_gate.weight").unwrap().0;
            let g = matvec(sg, xt, I, H);
            let u = matvec(su, xt, I, H);
            let h: Vec<f32> = (0..I).map(|i| silu(g[i]) * u[i]).collect();
            let y = matvec(sd, &h, H, I);
            let gate_scalar = 1.0 / (1.0 + (-matvec(sgate, xt, 1, H)[0]).exp());
            for (o, v) in y.iter().enumerate() {
                out[t * H + o] += gate_scalar * v;
            }
        }
    }
    out
}

fn run(with_shared: bool, norm_topk_prob: bool) -> (Vec<f32>, Vec<f32>) {
    synaptix_kernels_cpu::ensure_registered();
    let w = weights(with_shared);
    let x = noise(42, T * H);
    let moe = MoeFfn::load(
        &w,
        "mlp",
        cfg(with_shared, norm_topk_prob),
        Device::Cpu,
        DType::F32,
        DType::F32, // не квантованный dtype → плотный путь
    )
    .expect("сборка MoE");
    let xt = Tensor::from_vec::<_, f32>(x.clone(), vec![T, H], Device::Cpu).unwrap();
    let got = moe
        .forward(&xt)
        .expect("forward")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    (got, reference(&w, &x, with_shared, norm_topk_prob))
}

fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "[{i}] {g} != {w} (допуск {tol})"
        );
    }
}

#[test]
fn matches_reference_with_shared_expert() {
    let (got, want) = run(true, true);
    assert_close(&got, &want, 1e-5);
}

#[test]
fn matches_reference_without_shared_expert() {
    let (got, want) = run(false, true);
    assert_close(&got, &want, 1e-5);
}

/// Без нормировки веса берутся из общей softmax и в сумме меньше единицы —
/// это другой результат, а не тот же с точностью до масштаба.
#[test]
fn unnormalised_routing_differs_and_matches_its_own_reference() {
    let (got, want) = run(false, false);
    assert_close(&got, &want, 1e-5);
    let (normalised, _) = run(false, true);
    let diff: f32 = got.iter().zip(&normalised).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-3, "нормировка top-k обязана менять выход");
}

/// Один токен — путь decode: сортировка по экспертам вырождается, но
/// перестановки обязаны остаться согласованными.
#[test]
fn single_token_matches_reference() {
    synaptix_kernels_cpu::ensure_registered();
    let w = weights(true);
    let x = noise(7, H);
    let moe = MoeFfn::load(&w, "mlp", cfg(true, true), Device::Cpu, DType::F32, DType::F32)
        .expect("сборка MoE");
    let xt = Tensor::from_vec::<_, f32>(x.clone(), vec![1usize, H], Device::Cpu).unwrap();
    let got = moe.forward(&xt).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Тот же эталон, но на одном токене.
    let mut full = vec![0f32; T * H];
    full[..H].copy_from_slice(&x);
    let want = reference(&w, &full, true, true);
    assert_close(&got, &want[..H], 1e-5);
}

/// Расхождение формы стопки с конфигом — ошибка загрузки, а не тихое чтение
/// чужих байт.
#[test]
fn wrong_expert_shape_is_rejected() {
    synaptix_kernels_cpu::ensure_registered();
    let mut w = weights(false);
    w.data.insert(
        "mlp.experts.down_proj".to_string(),
        (noise(3, E * H * (I + 1)), vec![E, H, I + 1]),
    );
    let err = match MoeFfn::load(&w, "mlp", cfg(false, true), Device::Cpu, DType::F32, DType::F32) {
        Ok(_) => panic!("форма стопки не совпадает с конфигом — загрузка обязана упасть"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("down_proj"), "{err}");
}
