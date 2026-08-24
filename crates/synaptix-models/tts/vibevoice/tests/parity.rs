use std::path::PathBuf;

use safetensors::SafeTensors;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tts_vibevoice::config::GenerationConfig;
use synaptix_tts_vibevoice::generate::SpeechGenerator;
use synaptix_tts_vibevoice::loader::VibeVoiceCheckpoint;
use synaptix_tts_vibevoice::model::VibeVoiceModel;
use synaptix_tts_vibevoice::processor::VibeVoiceProcessor;
use synaptix_tts_vibevoice::schedule::DpmSolverMultistep;

struct Ref {
    bytes: Vec<u8>,
}

impl Ref {
    fn open(path: &PathBuf) -> Self {
        Self {
            bytes: std::fs::read(path).expect("reference safetensors"),
        }
    }

    fn get(&self, name: &str) -> (Vec<f32>, Vec<usize>) {
        let st = SafeTensors::deserialize(&self.bytes).expect("deserialize ref");
        let v = st.tensor(name).unwrap_or_else(|_| panic!("ref tensor {name}"));
        let shape = v.shape().to_vec();
        let data = match v.dtype() {
            safetensors::Dtype::F32 => v
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            safetensors::Dtype::I64 => v
                .data()
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            safetensors::Dtype::U8 => v.data().iter().map(|b| *b as f32).collect(),
            other => panic!("ref dtype {other:?}"),
        };
        (data, shape)
    }

    fn tensor(&self, name: &str, device: Device) -> Tensor {
        let (data, shape) = self.get(name);
        Tensor::from_vec(data, shape, device).expect("ref tensor upload")
    }

    fn ids(&self, name: &str) -> Vec<i64> {
        let st = SafeTensors::deserialize(&self.bytes).expect("deserialize ref");
        let v = st.tensor(name).unwrap_or_else(|_| panic!("ref tensor {name}"));
        v.data()
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }
}

fn env_paths() -> Option<(PathBuf, PathBuf)> {
    let bundle = std::env::var("SYN_VV_BUNDLE").ok()?;
    let reference = std::env::var("SYN_VV_REF").ok()?;
    Some((PathBuf::from(bundle), PathBuf::from(reference)))
}

fn device() -> Device {
    match std::env::var("SYN_VV_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ => Device::Cuda(0),
    }
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("to host")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()))
}

fn report(name: &str, got: &[f32], want: &[f32], min_cos: f64) {
    let c = cosine(got, want);
    let m = max_abs(got, want);
    println!("{name}: cos={c:.8} max_abs={m:.6e} n={}", got.len());
    assert!(c >= min_cos, "{name}: cos {c:.8} < {min_cos}");
}

struct Fixture {
    model: VibeVoiceModel,
    processor: VibeVoiceProcessor,
    reference: Ref,
    device: Device,
}

fn init_backends() {
    synaptix_kernels_cpu::ensure_registered();
    if std::env::var("SYN_VV_DEVICE").as_deref() != Ok("cpu") {
        synaptix_kernels_cuda::ensure_registered();
    }
}

fn setup() -> Option<&'static Fixture> {
    static FIXTURE: std::sync::OnceLock<Option<Fixture>> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(build_fixture).as_ref()
}

fn build_fixture() -> Option<Fixture> {
    init_backends();
    let (bundle, refpath) = match env_paths() {
        Some(v) => v,
        None => {
            eprintln!("SYN_VV_BUNDLE / SYN_VV_REF не заданы — тест пропущен");
            return None;
        }
    };
    let device = device();
    let ckpt = VibeVoiceCheckpoint::open(&bundle, device, DType::F32).expect("open checkpoint");
    let processor =
        VibeVoiceProcessor::new(&ckpt.tokenizer_json, &ckpt.preprocessor).expect("processor");
    let model = VibeVoiceModel::load(&ckpt, 4096).expect("load model");
    Some(Fixture {
        model,
        processor,
        reference: Ref::open(&refpath),
        device,
    })
}

#[test]
fn scheduler_matches_reference() {
    let Some(fx) = setup() else { return };
    let mut sched = DpmSolverMultistep::new(1000, "cosine").expect("scheduler");
    sched.set_timesteps(20);
    let (ts, _) = fx.reference.get("sched_timesteps");
    let (sg, _) = fx.reference.get("sched_sigmas");
    assert_eq!(sched.timesteps.len(), ts.len());
    for (i, (a, b)) in sched.timesteps.iter().zip(ts.iter()).enumerate() {
        assert_eq!(*a, *b, "timestep {i}");
    }
    let mine: Vec<f32> = sched.sigmas.iter().map(|v| *v as f32).collect();
    report("sigmas", &mine, &sg, 0.999_999_9);
}

#[test]
fn acoustic_encode_matches_reference() {
    let Some(fx) = setup() else { return };
    let audio = fx.reference.tensor("audio_in", fx.device);
    let dims = audio.dims().to_vec();
    let audio = audio.reshape(vec![dims[0], 1usize, dims[1]]).unwrap();
    let got = fx.model.acoustic.encode(&audio, None).expect("encode");
    let (want, _) = fx.reference.get("acoustic_encode_mean");
    report("acoustic_encode", &host(&got), &want, 0.9999);
}

#[test]
fn semantic_encode_matches_reference() {
    let Some(fx) = setup() else { return };
    let audio = fx.reference.tensor("audio_in", fx.device);
    let dims = audio.dims().to_vec();
    let audio = audio.reshape(vec![dims[0], 1usize, dims[1]]).unwrap();
    let got = fx.model.semantic.encode(&audio, None).expect("encode");
    let (want, _) = fx.reference.get("semantic_encode_mean");
    report("semantic_encode", &host(&got), &want, 0.9999);
}

#[test]
fn acoustic_decode_full_matches_reference() {
    let Some(fx) = setup() else { return };
    let lat = fx.reference.tensor("decode_latents", fx.device);
    let got = fx.model.acoustic.decode(&lat, None).expect("decode");
    let (want, _) = fx.reference.get("acoustic_decode_full");
    report("acoustic_decode_full", &host(&got), &want, 0.9999);
}

#[test]
fn streaming_decode_and_semantic_match_reference() {
    let Some(fx) = setup() else { return };
    let lat = fx.reference.tensor("decode_latents", fx.device);
    let frames = lat.dims()[1];
    let mut acache = fx.model.acoustic.new_cache();
    let mut scache = fx.model.semantic.new_cache();
    let mut audio: Vec<f32> = Vec::new();
    let mut sem: Vec<f32> = Vec::new();
    for i in 0..frames {
        let step = lat.narrow(1, i, 1).unwrap().contiguous().unwrap();
        let chunk = fx
            .model
            .acoustic
            .decode(&step, Some(&mut acache))
            .expect("stream decode");
        audio.extend_from_slice(&host(&chunk));
        let s = fx
            .model
            .semantic
            .encode(&chunk, Some(&mut scache))
            .expect("stream semantic");
        sem.extend_from_slice(&host(&s));
    }
    let (want_audio, _) = fx.reference.get("acoustic_decode_stream");
    let (want_sem, _) = fx.reference.get("semantic_encode_stream");
    report("acoustic_decode_stream", &audio, &want_audio, 0.9999);
    report("semantic_encode_stream", &sem, &want_sem, 0.9999);
}

#[test]
fn diffusion_head_matches_reference() {
    let Some(fx) = setup() else { return };
    let noisy = fx.reference.tensor("head_noisy", fx.device);
    let cond = fx.reference.tensor("head_cond", fx.device);
    let (ts, _) = fx.reference.get("head_t");
    let got = fx.model.head.forward(&noisy, &ts, &cond).expect("head");
    let (want, _) = fx.reference.get("head_out");
    report("diffusion_head", &host(&got), &want, 0.9999);
}

#[test]
fn connectors_match_reference() {
    let Some(fx) = setup() else { return };
    let ain = fx.reference.tensor("conn_acoustic_in", fx.device);
    let sin = fx.reference.tensor("conn_semantic_in", fx.device);
    let aout = fx.model.acoustic_connector.forward(&ain).expect("acoustic conn");
    let sout = fx.model.semantic_connector.forward(&sin).expect("semantic conn");
    let (want_a, _) = fx.reference.get("conn_acoustic_out");
    let (want_s, _) = fx.reference.get("conn_semantic_out");
    report("acoustic_connector", &host(&aout), &want_a, 0.9999);
    report("semantic_connector", &host(&sout), &want_s, 0.9999);
}

#[test]
fn language_model_matches_reference() {
    let Some(fx) = setup() else { return };
    let ids = fx.reference.ids("lm_ids");
    let embeds = fx.model.lm.embed_tokens(&ids).expect("embed");
    let mut cache = fx.model.lm.new_cache(64).expect("cache");
    let hidden = fx.model.lm.forward(&embeds, &mut cache).expect("prefill");
    let (want_hidden, _) = fx.reference.get("lm_hidden");
    report("lm_hidden", &host(&hidden), &want_hidden, 0.9995);

    let logits = fx.model.lm.lm_logits(&hidden).expect("logits");
    let (want_logits, _) = fx.reference.get("lm_logits");
    report("lm_logits", &host(&logits), &want_logits, 0.9995);

    let next = fx.reference.ids("lm_next_ids");
    let nembed = fx.model.lm.embed_tokens(&next).expect("embed next");
    let nhidden = fx.model.lm.forward(&nembed, &mut cache).expect("decode");
    let (want_next, _) = fx.reference.get("lm_next_hidden");
    report("lm_next_hidden", &host(&nhidden), &want_next, 0.9995);
}

#[test]
fn cfg_sampling_matches_reference() {
    let Some(fx) = setup() else { return };
    let pos = fx.reference.tensor("cfg_pos", fx.device);
    let neg = fx.reference.tensor("cfg_neg", fx.device);
    let init_full = fx.reference.tensor("cfg_init_noise", fx.device);
    let batch = pos.dims()[0];
    let init = init_full.narrow(0, 0, batch).unwrap().contiguous().unwrap();
    let mut gen = SpeechGenerator::new(&fx.model, &fx.processor, 0).expect("generator");
    let got = gen
        .sample_latent(&pos, &neg, 1.3, 20, &init)
        .expect("sample");
    let (want, _) = fx.reference.get("cfg_sampled");
    report("cfg_sampled", &host(&got), &want, 0.999);
}

#[test]
fn prompt_tokens_match_reference() {
    let Some(fx) = setup() else { return };
    let want = fx.reference.ids("prompt_input_ids");
    let (v0, _) = fx.reference.get("prompt_voice_raw_0");
    let (v1, _) = fx.reference.get("prompt_voice_raw_1");
    let voices = vec![v0, v1];
    let script =
        "Speaker 1: Hello there, this is a parity probe.\nSpeaker 2: And this is the second speaker line.";
    let prompt = fx
        .processor
        .build_prompt(script, &voices)
        .expect("build prompt");
    assert_eq!(
        prompt.input_ids.len(),
        want.len(),
        "prompt length {} != {}",
        prompt.input_ids.len(),
        want.len()
    );
    for (i, (a, b)) in prompt.input_ids.iter().zip(want.iter()).enumerate() {
        assert_eq!(a, b, "token {i}");
    }
    let (mask_ref, _) = fx.reference.get("prompt_speech_input_mask");
    for (i, (a, b)) in prompt
        .speech_input_mask
        .iter()
        .zip(mask_ref.iter())
        .enumerate()
    {
        assert_eq!(*a, *b > 0.5, "speech mask {i}");
    }
    let (want_speech, speech_shape) = fx.reference.get("prompt_speech_tensors");
    let l = speech_shape[1];
    let mut flat: Vec<f32> = Vec::new();
    for wav in &prompt.speech_tensors {
        assert_eq!(wav.len(), l, "padded voice length");
        flat.extend_from_slice(wav);
    }
    report("prompt_speech_tensors", &flat, &want_speech, 0.999_999);
    let (want_masks, _) = fx.reference.get("prompt_speech_masks");
    let mut mine: Vec<f32> = Vec::new();
    for m in &prompt.speech_masks {
        mine.extend(m.iter().map(|v| if *v { 1.0 } else { 0.0 }));
    }
    assert_eq!(mine.len(), want_masks.len(), "speech mask shape");
    for (i, (a, b)) in mine.iter().zip(want_masks.iter()).enumerate() {
        assert_eq!(a, b, "speech mask row {i}");
    }
    let _ = GenerationConfig::default();
}
