//! Выгрузить сырые байты тензоров бандла: `syn_tensor_dump <file.syn> <out_dir> <имя>...`
//! (файл `<out_dir>/<имя>.bin`, рядом `<имя>.json` — dtype и форма). Для сверки весов
//! с внешним чекпойнтом.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let b = synaptix_bundle::Bundle::open(&a[1]).expect("open");
    let st = b.tensors_slice().expect("tensors");
    let hlen = u64::from_le_bytes(st[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&st[8..8 + hlen]).expect("header");
    let data = &st[8 + hlen..];
    let out = std::path::Path::new(&a[2]);
    std::fs::create_dir_all(out).unwrap();
    for name in &a[3..] {
        let Some(spec) = header.get(name) else { eprintln!("{name}: нет в бандле"); continue };
        let off = spec["data_offsets"].as_array().unwrap();
        let (s, e) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
        std::fs::write(out.join(format!("{name}.bin")), &data[s..e]).unwrap();
        std::fs::write(out.join(format!("{name}.json")), spec.to_string()).unwrap();
        println!("{name} {} {}", spec["dtype"], spec["shape"]);
    }
}
