//! Тесты `synaptix_autograd::graph::ComputeGraph::backward` (делегация в core::grad) +
//! topological_order на основе edges.

use synaptix_autograd::graph::ComputeGraph;
use synaptix_autograd::init as autograd_init;
use synaptix_core::device::Device;
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

fn flat(t: &Tensor) -> Vec<f32> {
    let n: usize = t.dims().iter().product();
    t.contiguous().unwrap().reshape((n,)).unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn backward_propagates_via_core_grad() {
    setup();
    // y = (a @ b) ; loss = y.sum()
    // dL/da = 1 @ b.T (broadcasted); dL/db = a.T @ 1
    let a_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let b_data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.5).collect();
    let a = leaf(a_data, &[2, 3]);
    let b = leaf(b_data, &[3, 4]);
    let y = a.matmul(&b).unwrap();
    let loss = y.sum_all().unwrap();

    let mut g = ComputeGraph::new();
    let id_a = g.add_node(a.clone());
    let id_b = g.add_node(b.clone());
    let id_y = g.add_node(y.clone());
    let id_l = g.add_node(loss.clone());
    g.add_edge(id_a, id_y);
    g.add_edge(id_b, id_y);
    g.add_edge(id_y, id_l);

    g.backward(id_l).unwrap();

    let ga = flat(&a.grad().expect("a.grad"));
    let gb = flat(&b.grad().expect("b.grad"));
    assert_eq!(ga.len(), 6);
    assert_eq!(gb.len(), 12);

    // Через ComputeGraph::grad тоже доступны те же градиенты.
    let ga2 = flat(g.grad(id_a).expect("graph.grad(a)"));
    let gb2 = flat(g.grad(id_b).expect("graph.grad(b)"));
    assert_eq!(ga2, ga);
    assert_eq!(gb2, gb);
}

#[test]
fn backward_node_id_out_of_range() {
    setup();
    let mut g = ComputeGraph::new();
    let err = g.backward(0);
    assert!(err.is_err(), "пустой граф не должен иметь node 0");
}

#[test]
fn topological_order_linear_chain() {
    let mut g = ComputeGraph::new();
    let dummy = Tensor::from_vec(vec![0.0f32], vec![1], Device::Cpu).unwrap();
    for _ in 0..4 {
        g.add_node(dummy.clone());
    }
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    let order = g.topological_order().unwrap();
    assert_eq!(order, vec![0, 1, 2, 3]);
}

#[test]
fn topological_order_diamond() {
    let mut g = ComputeGraph::new();
    let dummy = Tensor::from_vec(vec![0.0f32], vec![1], Device::Cpu).unwrap();
    for _ in 0..4 {
        g.add_node(dummy.clone());
    }
    //   0
    //  / \
    // 1   2
    //  \ /
    //   3
    g.add_edge(0, 1);
    g.add_edge(0, 2);
    g.add_edge(1, 3);
    g.add_edge(2, 3);
    let order = g.topological_order().unwrap();
    // Возможны два валидных порядка: [0,1,2,3] или [0,2,1,3]. Главное — 0 перед {1,2}, и 3 после.
    let idx = |n: usize| order.iter().position(|&x| x == n).unwrap();
    assert!(idx(0) < idx(1));
    assert!(idx(0) < idx(2));
    assert!(idx(1) < idx(3));
    assert!(idx(2) < idx(3));
}

#[test]
fn topological_order_detects_cycle() {
    let mut g = ComputeGraph::new();
    let dummy = Tensor::from_vec(vec![0.0f32], vec![1], Device::Cpu).unwrap();
    for _ in 0..3 {
        g.add_node(dummy.clone());
    }
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 0); // cycle
    let err = g.topological_order();
    assert!(err.is_err());
    let msg = format!("{}", err.unwrap_err());
    assert!(msg.contains("cycle"), "ожидали 'cycle', получили: {msg}");
}

#[test]
fn topological_order_out_of_range_edge_errors() {
    let mut g = ComputeGraph::new();
    let dummy = Tensor::from_vec(vec![0.0f32], vec![1], Device::Cpu).unwrap();
    g.add_node(dummy);
    g.add_edge(0, 5); // 5 нет в графе
    let err = g.topological_order();
    assert!(err.is_err());
}
