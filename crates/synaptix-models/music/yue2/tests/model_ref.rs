//! Сверка AR-логитов и NAR-скоростей с эталоном релиза (FP32, CPU).
//!
//! Входы фиксированы и повторены в `scratchpad/ref/model_ref.py`: тот же
//! префикс, те же семантические токены чанка, то же состояние ODE. Эталонный
//! прогон заодно показал, что оптимизированный путь релиза (`CachedNAR`, его и
//! повторяет наш `nar`) сходится с каноническим `nar_velocity` до 2.4e-6 —
//! значит сверять чанкованный путь корректно.

use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_yue2::ar::{run_phase, ArSession, Phase, Yue2Ar};
use synaptix_music_yue2::nar::{prefill_chunk, Yue2Nar};
use synaptix_music_yue2::protocol::{Sampling, CODEC_OFFSET, MUSIC_END};

/// Префикс эталона: запрос «english, pop, female vocal» с короткой партитурой.
const PREFIX: [u32; 60] = [
    151643, 31115, 264, 43221, 12, 3401, 657, 19360, 45840, 11, 1221, 6923, 4627, 448, 34647,
    11211, 504, 279, 2661, 4682, 624, 58, 15930, 921, 29120, 11, 2420, 11, 8778, 25407, 198, 58,
    47312, 6198, 921, 58, 4450, 921, 89143, 358, 2776, 34347, 271, 151847, 55, 25, 16, 198, 42,
    55992, 198, 47576, 34, 96946, 17, 384, 17, 7360, 151848, 151851,
];
const CODEC: [u32; 8] = [0, 100, 1000, 5000, 12345, 20000, 32767, 7];
const FRAMES: usize = 8;
const LATENT: usize = 64;

/// Эталонные логиты последней позиции префикса.
const REF_ARGMAX: u32 = 163_899;
const REF_TOP3: [(u32, f32); 3] =
    [(163_899, 19.034_857), (169_677, 16.076_710), (158_642, 12.547_413)];
const REF_CODEC_ARGMAX: u32 = 12_046;
const REF_CODEC_MEAN: f32 = -0.174_131_89;
const REF_TEXT_MEAN: f32 = -5.042_542_5;

fn bundle() -> Option<PathBuf> {
    let p = std::env::var("SYN_YUE2_BUNDLE").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("Storage/syn_models/yue2-3b.syn")
    });
    p.exists().then_some(p)
}

/// То же состояние ODE, что у эталонного скрипта.
fn det_state() -> Vec<f32> {
    let mut x = vec![0f32; FRAMES * LATENT];
    for t in 0..FRAMES {
        for c in 0..LATENT {
            x[t * LATENT + c] = (0.21 * (t as f32 + 1.0) + 0.13 * c as f32).sin() * 0.5;
        }
    }
    x
}

#[test]
#[ignore = "грузит 3,6B весов в FP32 на CPU — запускать точечно"]
fn ar_logits_match_reference() {
    let Some(path) = bundle() else {
        eprintln!("[yue2-ar] бандла нет — тест пропущен");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    let ar = Yue2Ar::open(&path, Device::Cpu, DType::F32, None, 4096).expect("открыть AR");
    assert_eq!(ar.config.num_hidden_layers, 28);
    assert_eq!(ar.config.vocab_size, 184_704);

    let mut session = ArSession::new(&ar, PREFIX.len() + 8).expect("сессия");
    session.feed(&PREFIX).expect("префилл");
    let logits: Vec<f32> = session
        .logits_snapshot()
        .expect("логиты")
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(logits.len(), 184_704);

    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    eprintln!(
        "[yue2-ar] argmax={argmax} ({:.4}), эталон {REF_ARGMAX} ({:.4})",
        logits[argmax as usize], REF_TOP3[0].1
    );
    assert_eq!(argmax, REF_ARGMAX, "argmax логитов расходится с эталоном");
    for (id, want) in REF_TOP3 {
        let got = logits[id as usize];
        assert!((got - want).abs() < 0.05, "логит {id}: {got} против эталона {want}");
    }

    let text = &logits[..151_643];
    let codec = &logits[CODEC_OFFSET as usize..CODEC_OFFSET as usize + 32_768];
    let text_mean = text.iter().sum::<f32>() / text.len() as f32;
    let codec_mean = codec.iter().sum::<f32>() / codec.len() as f32;
    let codec_argmax = codec
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    assert!((text_mean - REF_TEXT_MEAN).abs() < 0.02, "среднее текстовой части {text_mean}");
    assert!((codec_mean - REF_CODEC_MEAN).abs() < 0.02, "среднее кодек-части {codec_mean}");
    assert_eq!(codec_argmax, REF_CODEC_ARGMAX, "лучший семантический токен");
}

/// Ожидания NAR: `(raw_t, mean, std, min, max, первые 4 канала кадра 0)`.
const REF_NAR: [(f32, f32, f32, f32, f32, [f32; 4]); 3] = [
    (20.0, 0.166_705, 0.728_235, -2.385_17, 2.435_07, [2.074_06, -0.349_86, 0.693_17, 0.246_37]),
    (0.0, 0.136_342, 0.846_200, -1.997_28, 3.472_63, [2.522_04, -1.065_87, -0.344_03, 0.479_56]),
    (
        -1.098_612_3,
        0.286_887,
        1.048_588,
        -1.818_66,
        4.804_71,
        [2.797_79, -0.775_29, 0.282_26, 0.708_14],
    ),
];

#[test]
#[ignore = "грузит 3,6B весов в FP32 на CPU — запускать точечно"]
fn nar_velocity_matches_reference() {
    let Some(path) = bundle() else {
        eprintln!("[yue2-nar] бандла нет — тест пропущен");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    let ar = Yue2Ar::open(&path, Device::Cpu, DType::F32, None, 4096).expect("открыть AR");
    let nar = Yue2Nar::open(&path, Device::Cpu, DType::F32, None).expect("открыть NAR");

    let mut ar_tokens: Vec<u32> = PREFIX.to_vec();
    ar_tokens.extend(CODEC.iter().map(|&c| c + CODEC_OFFSET));
    ar_tokens.push(MUSIC_END);
    let mut ctx = prefill_chunk(&ar, &ar_tokens, FRAMES).expect("AR-контекст чанка");
    assert_eq!(ctx.ar_len(), ar_tokens.len());
    assert_eq!(ctx.nar_len(), FRAMES + 2);

    let state = Tensor::from_vec(det_state(), vec![FRAMES, LATENT], Device::Cpu).expect("состояние");
    for (raw_t, mean_ref, std_ref, min_ref, max_ref, head) in REF_NAR {
        let v = nar.velocity(&mut ctx, &state, raw_t).expect("скорость");
        assert_eq!(v.dims(), &[FRAMES, LATENT]);
        let values: Vec<f32> = v.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let var = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / values.len() as f32;
        let std = var.sqrt();
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!(
            "[yue2-nar] raw_t={raw_t}: mean={mean:.6} (эталон {mean_ref:.6}) std={std:.6} (эталон {std_ref:.6})"
        );
        assert!((mean - mean_ref).abs() < 5e-3, "среднее при raw_t={raw_t}: {mean}");
        assert!((std - std_ref).abs() < 5e-3, "разброс при raw_t={raw_t}: {std}");
        assert!((min - min_ref).abs() < 2e-2, "минимум при raw_t={raw_t}: {min}");
        assert!((max - max_ref).abs() < 2e-2, "максимум при raw_t={raw_t}: {max}");
        for (i, want) in head.iter().enumerate() {
            let got = values[i];
            assert!((got - want).abs() < 1e-2, "кадр 0, канал {i}: {got} против эталона {want}");
        }
    }
}

/// Детерминированный (argmax) эталон AR-цикла: 100 семантических токенов на том
/// же префиксе при `temperature = 0`, штрафе 1.2 и окне 50. Сверяет не только
/// веса, но и порядок операций сэмплинга — маску фазы, частотный штраф и запрет
/// конца фазы до `min_tokens`.
const REF_GREEDY: [u32; 100] = [
    12046, 4196, 14883, 24487, 31383, 23931, 23108, 32517, 29542, 345, 17417, 7187,
    18217, 3380, 6270, 31485, 12268, 26622, 9007, 8979, 24527, 21020, 6048, 14110,
    20360, 1944, 14713, 751, 7506, 24849, 26866, 11654, 3578, 28344, 21436, 4583,
    4271, 6205, 12851, 12168, 11240, 32519, 13191, 26610, 16079, 32225, 6215, 9525,
    19802, 21723, 12526, 607, 30404, 5642, 17128, 3288, 19827, 26615, 21050, 8927,
    11372, 18122, 15289, 22831, 14649, 5637, 29478, 648, 27337, 26777, 18078, 8092,
    17701, 30071, 11879, 21354, 1583, 32551, 20368, 22497, 21050, 18576, 3864, 4290,
    4225, 14976, 16127, 10909, 29683, 28354, 9133, 19170, 13678, 11251, 7554, 23945,
    2378, 3687, 11556, 5358,
];

#[test]
#[ignore = "100 шагов декода на CPU в FP32 — запускать точечно"]
fn greedy_semantic_matches_reference() {
    let Some(path) = bundle() else {
        eprintln!("[yue2-ar] бандла нет — тест пропущен");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    let ar = Yue2Ar::open(&path, Device::Cpu, DType::F32, None, 4096).expect("открыть AR");
    let sampling = Sampling {
        temperature: 0.0,
        top_p: 0.95,
        top_k: 100,
        repetition_penalty: 1.2,
        penalty_window: 50,
        min_tokens: 100,
        max_tokens: 100,
    };
    let mut session = ArSession::new(&ar, PREFIX.len() + sampling.max_tokens + 2).expect("сессия");
    session.feed(&PREFIX).expect("префилл");
    let out = run_phase(
        &mut session,
        None,
        Phase::Semantic,
        &sampling,
        1.0,
        7,
        &|| false,
        &|_, _, _| {},
    )
    .expect("фаза музыки");
    assert!(out.truncated, "фаза должна упереться в лимит, а не закончиться сама");
    let codec: Vec<u32> = out.tokens.iter().map(|t| t - CODEC_OFFSET).collect();
    assert_eq!(codec.len(), REF_GREEDY.len());
    let first_diff = codec.iter().zip(REF_GREEDY.iter()).position(|(a, b)| a != b);
    eprintln!(
        "[yue2-ar] greedy: первые 8 {:?}, эталон {:?}, первое расхождение {:?}",
        &codec[..8],
        &REF_GREEDY[..8],
        first_diff
    );
    assert_eq!(codec, REF_GREEDY.to_vec(), "argmax-последовательность разошлась с эталоном");
}
