use synaptix_autograd::init as autograd_init;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::ensure_registered;

fn setup() {
    ensure_registered();
    autograd_init().unwrap();
}

fn leaf(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape.to_vec(), Device::Cpu)
        .unwrap()
        .requires_grad_(true)
}

fn const_t(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape.to_vec(), Device::Cpu).unwrap()
}

fn flat(t: &Tensor) -> Vec<f32> {
    let numel: usize = t.dims().iter().product();
    t.contiguous().unwrap().reshape((numel,)).unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn index_select_backward_scatters_grad() {
    setup();
    let weight = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
    let indices = Tensor::from_vec(vec![0u32, 2, 0, 3], (4usize,), Device::Cpu).unwrap();
    let selected = weight.index_select(0, &indices).unwrap();
    assert_eq!(selected.dims(), &[4, 2]);
    selected.sum_all().unwrap().backward().unwrap();
    let g = flat(&weight.grad().unwrap());
    assert_eq!(g, vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn gather_backward_scatters_grad() {
    setup();
    let weight = leaf(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let indices = Tensor::from_vec(vec![0u32, 2, 3, 1], (2usize, 2), Device::Cpu).unwrap();
    let gathered = weight.gather(&indices, 1).unwrap();
    assert_eq!(gathered.dims(), &[2, 2]);
    gathered.sum_all().unwrap().backward().unwrap();
    let g = flat(&weight.grad().unwrap());
    assert_eq!(g, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn masked_fill_backward_blocks_grad_at_mask() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mask = Tensor::from_vec(vec![0u8, 1, 1, 0], (2usize, 2), Device::Cpu).unwrap();
    let filled = a.masked_fill(&mask, -1e9).unwrap();
    filled.sum_all().unwrap().backward().unwrap();
    let g = flat(&a.grad().unwrap());
    assert_eq!(g, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn where_cond_backward_routes_grad_by_mask() {
    setup();
    let a = leaf(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let b = leaf(vec![100.0, 200.0, 300.0, 400.0], &[2, 2]);
    let cond = Tensor::from_vec(vec![1u8, 0, 0, 1], (2usize, 2), Device::Cpu).unwrap();
    let out = Tensor::where_cond(&cond, &a, &b).unwrap();
    out.sum_all().unwrap().backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());
    assert_eq!(ga, vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(gb, vec![0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn cat_backward_splits_grad() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let b = leaf(vec![10.0, 20.0], &[2]);
    let c = Tensor::cat(&[&a, &b], 0).unwrap();
    assert_eq!(c.dims(), &[5]);
    let weights = const_t(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    c.mul(&weights).unwrap().sum_all().unwrap().backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());
    assert_eq!(ga, vec![1.0, 2.0, 3.0]);
    assert_eq!(gb, vec![4.0, 5.0]);
}

#[test]
fn cat_2d_backward_splits_along_axis_1() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = leaf(vec![5.0, 6.0], &[2, 1]);
    let c = Tensor::cat(&[&a, &b], 1).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    c.sum_all().unwrap().backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());
    assert_eq!(ga, vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(gb, vec![1.0, 1.0]);
}

#[test]
fn stack_backward_via_composition() {
    setup();
    let a = leaf(vec![1.0, 2.0, 3.0], &[3]);
    let b = leaf(vec![10.0, 20.0, 30.0], &[3]);
    let s = Tensor::stack(&[&a, &b], 0).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    s.sum_all().unwrap().backward().unwrap();
    let ga = flat(&a.grad().unwrap());
    let gb = flat(&b.grad().unwrap());
    assert_eq!(ga, vec![1.0, 1.0, 1.0]);
    assert_eq!(gb, vec![1.0, 1.0, 1.0]);
}

#[test]
fn token_embedding_lookup_with_grad() {
    setup();
    let vocab = 5usize;
    let dim = 4usize;
    let embedding = leaf(
        (1..=(vocab * dim) as i32).map(|x| x as f32).collect(),
        &[vocab, dim],
    );
    let token_ids = Tensor::from_vec(vec![0u32, 2, 4, 2], (4usize,), Device::Cpu).unwrap();
    let _ = DType::F32;
    let looked_up = embedding.index_select(0, &token_ids).unwrap();
    assert_eq!(looked_up.dims(), &[4, dim]);
    looked_up.sum_all().unwrap().backward().unwrap();
    let g = flat(&embedding.grad().unwrap());
    let row = |i: usize| -> Vec<f32> { g[i * dim..(i + 1) * dim].to_vec() };
    assert_eq!(row(0), vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(row(1), vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(row(2), vec![2.0, 2.0, 2.0, 2.0]);
    assert_eq!(row(3), vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(row(4), vec![1.0, 1.0, 1.0, 1.0]);
}
