//! synaptix-recsys — feedforward модели с контролируемыми весами (аналитически).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::linear::Linear;
use synaptix_recsys::din::Din;
use synaptix_recsys::dlrm::Dlrm;
use synaptix_recsys::embedding_table::EmbeddingTable;
use synaptix_recsys::sasrec::Sasrec;
use synaptix_recsys::two_tower::TwoTower;
use synaptix_recsys::wide_deep::WideDeep;
use synaptix_kernels_cpu::ensure_registered;

fn setup() { ensure_registered(); }

/// Linear с известными весами; `rows` = `[out][in]`.
fn lin(rows: &[&[f32]]) -> Linear {
    let out = rows.len();
    let inp = rows[0].len();
    let mut data = Vec::new();
    for r in rows { data.extend_from_slice(r); }
    let w = Tensor::from_vec::<_, f32>(data, vec![out, inp], Device::Cpu).unwrap();
    Linear::new(w, None).unwrap()
}

fn mat(rows: &[&[f32]]) -> Tensor {
    let cols = rows[0].len();
    let mut data = Vec::new();
    for r in rows { data.extend_from_slice(r); }
    Tensor::from_vec::<_, f32>(data, vec![rows.len(), cols], Device::Cpu).unwrap()
}

#[test]
fn t70_1_embedding_table_lookup() {
    setup();
    let mut t = EmbeddingTable::new(4, 2, Device::Cpu, DType::F32).unwrap();
    t.weight = mat(&[&[0.0, 1.0], &[2.0, 3.0], &[4.0, 5.0], &[6.0, 7.0]]);
    let idx = Tensor::from_vec::<_, i64>(vec![2, 0, 3], vec![3], Device::Cpu).unwrap();
    let out = t.forward(&idx).unwrap();
    assert_eq!(out.dims(), &[3, 2]);
    assert_eq!(out.to_vec2::<f32>().unwrap(), vec![vec![4.0, 5.0], vec![0.0, 1.0], vec![6.0, 7.0]]);
}

#[test]
fn t70_2_two_tower_dot_score() {
    setup();
    // Башни-тождества (identity), dim 2.
    let tt = TwoTower::from_towers(lin(&[&[1.0, 0.0], &[0.0, 1.0]]), lin(&[&[1.0, 0.0], &[0.0, 1.0]]));
    let q = mat(&[&[1.0, 2.0]]);
    let i = mat(&[&[3.0, 4.0]]);
    // dot([1,2],[3,4]) = 11.
    assert!((tt.score(&q, &i).unwrap() - 11.0).abs() < 1e-5);
}

#[test]
fn t70_3_two_tower_batch_scores() {
    setup();
    let tt = TwoTower::from_towers(lin(&[&[1.0, 0.0], &[0.0, 1.0]]), lin(&[&[1.0, 0.0], &[0.0, 1.0]]));
    let q = mat(&[&[1.0, 0.0], &[0.0, 2.0]]);
    let i = mat(&[&[5.0, 9.0], &[1.0, 3.0]]);
    // row0: 1*5+0*9=5; row1: 0*1+2*3=6.
    let s = tt.scores(&q, &i).unwrap();
    assert_eq!(s.dims(), &[2]);
    assert_eq!(s.to_vec1::<f32>().unwrap(), vec![5.0, 6.0]);
}

#[test]
fn t70_4_wide_deep_sum() {
    setup();
    // wide: [1,1] (сумма входа); deep: один слой [2,0] (= 2*x0).
    let wd = WideDeep::from_layers(lin(&[&[1.0, 1.0]]), vec![lin(&[&[2.0, 0.0]])]);
    let wide_x = mat(&[&[1.0, 2.0]]); // wide = 3
    let deep_x = mat(&[&[3.0, 4.0]]); // deep = 6
    let out = wd.forward(&wide_x, &deep_x).unwrap();
    assert_eq!(out.dims(), &[1, 1]);
    assert!((out.to_vec2::<f32>().unwrap()[0][0] - 9.0).abs() < 1e-5);
}

#[test]
fn t70_5_dlrm_interaction() {
    setup();
    // bottom = identity [2,2] → dense_emb = dense.
    let bottom = vec![lin(&[&[1.0, 0.0], &[0.0, 1.0]])];
    // emb table: row0 = [2,3].
    let mut emb = EmbeddingTable::new(2, 2, Device::Cpu, DType::F32).unwrap();
    emb.weight = mat(&[&[2.0, 3.0], &[0.0, 0.0]]);
    // top: [3->1] = сумма (dense_emb(2) + 1 пара).
    let top = vec![lin(&[&[1.0, 1.0, 1.0]])];
    let dlrm = Dlrm::from_layers(bottom, emb, top);

    let dense = mat(&[&[1.0, 1.0]]); // dense_emb = [1,1]
    let sparse = vec![Tensor::from_vec::<_, i64>(vec![0], vec![1], Device::Cpu).unwrap()]; // → [2,3]
    // interaction dot([1,1],[2,3]) = 5; top_in = [1,1,5]; сумма = 7.
    let out = dlrm.forward(&dense, &sparse).unwrap();
    assert_eq!(out.dims(), &[1, 1]);
    assert!((out.to_vec2::<f32>().unwrap()[0][0] - 7.0).abs() < 1e-5);
}

fn emb_table(rows: &[&[f32]]) -> EmbeddingTable {
    let mut t = EmbeddingTable::new(rows.len(), rows[0].len(), Device::Cpu, DType::F32).unwrap();
    t.weight = mat(rows);
    t
}

#[test]
fn t70_6_din_attention_pool() {
    setup();
    let din = Din::from_layers(emb_table(&[&[0.0, 0.0]]), vec![lin(&[&[1.0, 1.0, 1.0, 1.0]])]);
    let target = mat(&[&[1.0, 0.0]]);
    let behaviors = mat(&[&[1.0, 0.0], &[0.0, 1.0]]);
    // scale=1/√2: scores=[0.7071,0]; softmax≈[0.66978,0.33022]; pooled≈[0.66978,0.33022].
    let pooled = din.attention_pool(&target, &behaviors).unwrap();
    let p = pooled.to_vec2::<f32>().unwrap()[0].clone();
    assert!((p[0] - 0.66978).abs() < 1e-3, "p0={}", p[0]);
    assert!((p[1] - 0.33022).abs() < 1e-3, "p1={}", p[1]);
}

#[test]
fn t70_7_sasrec_causal_self_attention() {
    setup();
    let sas = Sasrec::from_emb(emb_table(&[&[0.0, 0.0]]));
    let embs = mat(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let att = sas.self_attention(&embs).unwrap();
    let a = att.to_vec2::<f32>().unwrap();
    // Позиция 0 видит только себя → att[0] == embs[0].
    assert!((a[0][0] - 1.0).abs() < 1e-5 && a[0][1].abs() < 1e-5, "att0={:?}", a[0]);
    // Позиция 1: softmax([0, 0.7071]) ≈ [0.33022, 0.66978].
    assert!((a[1][0] - 0.33022).abs() < 1e-3 && (a[1][1] - 0.66978).abs() < 1e-3, "att1={:?}", a[1]);
}

#[test]
fn t70_8_sasrec_next_item_logits() {
    setup();
    let sas = Sasrec::from_emb(emb_table(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]));
    let ids = Tensor::from_vec::<_, i64>(vec![0, 1], vec![2], Device::Cpu).unwrap();
    // last hidden ≈ [0.33022, 0.66978]; логиты = dot с каждой строкой эмбеддинга.
    let logits = sas.forward(&ids).unwrap();
    assert_eq!(logits.dims(), &[3]);
    let v = logits.to_vec1::<f32>().unwrap();
    assert!((v[0] - 0.33022).abs() < 1e-3, "v0={}", v[0]);
    assert!((v[1] - 0.66978).abs() < 1e-3, "v1={}", v[1]);
    assert!((v[2] - 1.0).abs() < 1e-3, "v2={}", v[2]);
}
