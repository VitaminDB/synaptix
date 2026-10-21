//! Извлечь файлы бандла (не тензоры) в каталог: `syn_extract <file.syn> <out_dir>`;
//! `syn_extract <file.syn> --tensors` — список тензоров.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let b = synaptix_bundle::Bundle::open(&a[1]).expect("open");
    if a[2] == "--tensors" {
        let t = synaptix_bundle::inspect::read_header_slice(b.tensors_slice().expect("tensors")).expect("header");
        for x in t { println!("{} {} {:?}", x.name, x.dtype, x.shape); }
        return;
    }
    let out = std::path::Path::new(&a[2]);
    let names: Vec<String> = b.list_files().map(|e| e.name.clone()).collect();
    for n in names {
        let rel = n.trim_start_matches("files/").trim_start_matches('/');
        let p = out.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        match b.read_file(&n) {
            Ok(bytes) => { std::fs::write(&p, &bytes).unwrap(); println!("{n} -> {}", p.display()); }
            Err(e) => eprintln!("{n}: {e}"),
        }
    }
}
