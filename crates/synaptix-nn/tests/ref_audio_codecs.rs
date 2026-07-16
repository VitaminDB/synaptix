use synaptix_core::dtype::DType;
use synaptix_kernels_cpu::ensure_registered;
use synaptix_nn::audio::{
    fsq::FiniteScalarQuantizer, lfq::Lfq, rvq::ResidualVQ,
};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t23_1_fsq_3_3_3_quantize() {
    setup();
    let t = load_case("nn_audio_codecs", "fsq_3_3_3");
    let q = FiniteScalarQuantizer::new(vec![3, 3, 3]);
    let (codes, indices) = q.quantize(&t["z"]).unwrap();
    assert_allclose(&codes, &t["codes"], 1e-5, 1e-5);
    let py_idx = t["indices"].to_vec2::<i64>().unwrap().into_iter().flatten().collect::<Vec<_>>();
    let our_idx = indices.to_vec2::<i64>().unwrap().into_iter().flatten().collect::<Vec<_>>();
    assert_eq!(our_idx, py_idx);
}

#[test]
fn t23_2_fsq_3_3_3_dequantize() {
    setup();
    let t = load_case("nn_audio_codecs", "fsq_3_3_3");
    let q = FiniteScalarQuantizer::new(vec![3, 3, 3]);
    let dq = q.dequantize(&t["indices"], DType::F32).unwrap();
    assert_allclose(&dq, &t["dequantized"], 1e-5, 1e-5);
}

#[test]
fn t23_3_fsq_4_4_4_4() {
    setup();
    let t = load_case("nn_audio_codecs", "fsq_4_4_4_4");
    let q = FiniteScalarQuantizer::new(vec![4, 4, 4, 4]);
    let (codes, indices) = q.quantize(&t["z"]).unwrap();
    assert_allclose(&codes, &t["codes"], 1e-5, 1e-5);
    let py_idx = t["indices"].to_vec2::<i64>().unwrap().into_iter().flatten().collect::<Vec<_>>();
    let our_idx = indices.to_vec2::<i64>().unwrap().into_iter().flatten().collect::<Vec<_>>();
    assert_eq!(our_idx, py_idx);
    let dq = q.dequantize(&indices, DType::F32).unwrap();
    assert_allclose(&dq, &t["dequantized"], 1e-5, 1e-5);
}

#[test]
fn t23_4_lfq_dim4() {
    setup();
    let t = load_case("nn_audio_codecs", "lfq_dim4");
    let q = Lfq::new(16, 4);
    let (codes, indices) = q.quantize(&t["z"]).unwrap();
    assert_allclose(&codes, &t["codes"], 1e-5, 1e-5);
    let py_idx = t["indices"].to_vec2::<i64>().unwrap().into_iter().flatten().collect::<Vec<_>>();
    let our_idx = indices.to_vec2::<i64>().unwrap().into_iter().flatten().collect::<Vec<_>>();
    assert_eq!(our_idx, py_idx);
    let dq = q.dequantize(&indices, DType::F32).unwrap();
    assert_allclose(&dq, &t["dequantized"], 1e-5, 1e-5);
}

#[test]
fn t23_5_rvq_3cb_8sz_4dim() {
    setup();
    let t = load_case("nn_audio_codecs", "rvq_3cb_8sz_4dim");
    let rvq = ResidualVQ::from_codebooks(vec![
        t["cb0"].clone(), t["cb1"].clone(), t["cb2"].clone(),
    ]).unwrap();
    let indices = rvq.encode(&t["x"]).unwrap();
    let py_idx = t["indices"].to_vec3::<i64>().unwrap()
        .into_iter().flatten().flatten().collect::<Vec<_>>();
    let our_idx = indices.to_vec3::<i64>().unwrap()
        .into_iter().flatten().flatten().collect::<Vec<_>>();
    assert_eq!(our_idx, py_idx);
    let recon = rvq.decode(&indices, DType::F32).unwrap();
    assert_allclose(&recon, &t["recon"], 1e-5, 1e-5);
}
