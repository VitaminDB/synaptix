use std::collections::HashMap;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::ModelError;
use synaptix_llm_common::moe::{ExpertCache, MoeConfig, MoeFfn};
use synaptix_llm_common::weights::WeightSource;

const H: usize = 8;
const I: usize = 6;
const E: usize = 16;
const K: usize = 3;
const T: usize = 12;
const LAYERS: usize = 3;

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
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn weights(layer: usize) -> Weights {
    let seed = 1 + layer as u64 * 100;
    let mut data = HashMap::new();
    data.insert("mlp.gate.weight".to_string(), (noise(seed, E * H), vec![E, H]));
    data.insert(
        "mlp.experts.gate_up_proj".to_string(),
        (noise(seed + 1, E * 2 * I * H), vec![E, 2 * I, H]),
    );
    data.insert(
        "mlp.experts.down_proj".to_string(),
        (noise(seed + 2, E * H * I), vec![E, H, I]),
    );
    data.insert(
        "mlp.shared_expert.gate_proj.weight".to_string(),
        (noise(seed + 3, I * H), vec![I, H]),
    );
    data.insert(
        "mlp.shared_expert.up_proj.weight".to_string(),
        (noise(seed + 4, I * H), vec![I, H]),
    );
    data.insert(
        "mlp.shared_expert.down_proj.weight".to_string(),
        (noise(seed + 5, H * I), vec![H, I]),
    );
    data.insert(
        "mlp.shared_expert_gate.weight".to_string(),
        (noise(seed + 6, H), vec![1, H]),
    );
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
        chunk: 5,
        skip_below: 0.0,
    }
}

fn host_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .and_then(|x| x.to_dtype(DType::F32))
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .unwrap()
}

#[test]
fn offloaded_matches_resident() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec::<_, f32>(noise(42, T * H), vec![T, H], Device::Cpu).unwrap();

    // Кэш заведомо меньше нужного: 4 эксперта на все три слоя, значит
    // вытеснение сработает не раз за один прогон.
    let one_expert = (2 * I * H + H * I) * std::mem::size_of::<f32>();
    let cache = ExpertCache::new(Device::Cpu, one_expert * 4);

    for layer in 0..LAYERS {
        let w = weights(layer);
        let resident = MoeFfn::load(&w, "mlp", cfg(), Device::Cpu, DType::F32, DType::F32).unwrap();
        let offloaded = MoeFfn::load_offloaded(
            &w,
            "mlp",
            cfg(),
            Device::Cpu,
            DType::F32,
            DType::F32,
            cache.clone(),
            layer,
        )
        .unwrap();

        let want = host_vec(&resident.forward(&x).unwrap());
        let got = host_vec(&offloaded.forward(&x).unwrap());
        assert_eq!(want.len(), got.len());
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-6, "оффлоад разошёлся: {a} vs {b}");
        }
        assert!(offloaded.cache_stats().is_some());
    }

    let stats = cache.stats();
    assert!(stats.misses > 0, "кэш не заполнялся");
    assert!(
        stats.bytes <= one_expert * 4,
        "кэш вырос за ёмкость: {} > {}",
        stats.bytes,
        one_expert * 4
    );
    assert!(stats.resident <= 4, "резидентов больше ёмкости: {}", stats.resident);
}

#[test]
fn skip_below_drops_only_weak_pairs() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec::<_, f32>(noise(11, T * H), vec![T, H], Device::Cpu).unwrap();
    let one_expert = (2 * I * H + H * I) * std::mem::size_of::<f32>();
    let cache = ExpertCache::new(Device::Cpu, one_expert * 2);

    let w = weights(0);
    let exact = MoeFfn::load_offloaded(
        &w,
        "mlp",
        cfg(),
        Device::Cpu,
        DType::F32,
        DType::F32,
        cache.clone(),
        0,
    )
    .unwrap();
    let want = host_vec(&exact.forward(&x).unwrap());

    cache.clear();
    let mut approx_cfg = cfg();
    // Порог выше веса последнего слота top-k, но ниже первого: отбрасывается
    // хвост роутинга, основной вклад остаётся.
    approx_cfg.skip_below = 0.34;
    let approx = MoeFfn::load_offloaded(
        &w,
        "mlp",
        approx_cfg,
        Device::Cpu,
        DType::F32,
        DType::F32,
        cache.clone(),
        1,
    )
    .unwrap();
    let got = host_vec(&approx.forward(&x).unwrap());

    let stats = cache.stats();
    assert!(stats.skipped > 0, "клапан не сработал");
    let max_diff = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    assert!(max_diff > 0.0, "выход не изменился — клапан ничего не отбросил");
    assert!(
        max_diff < scale,
        "клапан снёс больше, чем весит сам выход: {max_diff} против {scale}"
    );
}

#[test]
fn cache_hits_on_repeat() {
    synaptix_kernels_cpu::ensure_registered();
    let x = Tensor::from_vec::<_, f32>(noise(7, T * H), vec![T, H], Device::Cpu).unwrap();
    let one_expert = (2 * I * H + H * I) * std::mem::size_of::<f32>();
    // Ёмкости хватает на всех экспертов слоя — второй прогон должен быть
    // полностью по кэшу.
    let cache = ExpertCache::new(Device::Cpu, one_expert * E);
    let w = weights(0);
    let moe = MoeFfn::load_offloaded(
        &w,
        "mlp",
        cfg(),
        Device::Cpu,
        DType::F32,
        DType::F32,
        cache.clone(),
        0,
    )
    .unwrap();

    let first = host_vec(&moe.forward(&x).unwrap());
    let after_first = cache.stats();
    let second = host_vec(&moe.forward(&x).unwrap());
    let after_second = cache.stats();

    assert_eq!(first, second);
    assert_eq!(
        after_second.misses, after_first.misses,
        "второй прогон снова читал эксперты из системной памяти"
    );
    assert!(after_second.hits > after_first.hits);
}
