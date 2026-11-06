//! Дамп графа ONNX и выгрузка весов: `onnx_dump model.onnx [--weights out.safetensors [--prefix P]] [--small N] [--filter УЗЕЛ]`.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = synaptix_io::onnx::Model::open(&a[1]).expect("onnx");
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let small: usize = get("--small").map(|s| s.parse().unwrap()).unwrap_or(8);
    if let Some(out) = get("--weights") {
        let mut w = m.named_weights(16);
        // --prefix P: имена весов `P.<имя>` (несколько графов в одном компоненте бандла)
        if let Some(p) = get("--prefix") {
            for t in &mut w {
                t.name = format!("{p}.{}", t.name);
            }
        }
        synaptix_io::onnx::write_safetensors(&w, std::path::Path::new(&out)).expect("write");
        eprintln!("{} тензоров → {out}", w.len());
        return;
    }
    print!("{}", m.dump_filtered(small, get("--filter").as_deref()));
}
