//! Дамп графа ONNX и выгрузка весов: `onnx_dump model.onnx [--weights out.safetensors] [--small N]`.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = synaptix_io::onnx::Model::open(&a[1]).expect("onnx");
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let small: usize = get("--small").map(|s| s.parse().unwrap()).unwrap_or(8);
    if let Some(out) = get("--weights") {
        let w = m.named_weights(16);
        synaptix_io::onnx::write_safetensors(&w, std::path::Path::new(&out)).expect("write");
        eprintln!("{} тензоров → {out}", w.len());
        return;
    }
    print!("{}", m.dump_filtered(small, get("--filter").as_deref()));
}
