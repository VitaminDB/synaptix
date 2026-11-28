//! Сравнить тензоры двух бандлов: `syn_tensor_diff <a.syn> <b.syn>` — какие тензоры есть только
//! в одном, у каких другие dtype/форма, у каких отличаются байты (с долей отличающихся байтов).
//! Для проверки дообученных/правленых чекпойнтов: что именно поменялось относительно исходного.
fn tensors(b: &synaptix_bundle::Bundle) -> (serde_json::Map<String, serde_json::Value>, &[u8]) {
    let st = b.tensors_slice().expect("tensors");
    let hlen = u64::from_le_bytes(st[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&st[8..8 + hlen]).expect("header");
    let mut map = header.as_object().unwrap().clone();
    map.remove("__metadata__");
    (map, &st[8 + hlen..])
}

fn bytes<'a>(spec: &serde_json::Value, data: &'a [u8]) -> &'a [u8] {
    let off = spec["data_offsets"].as_array().unwrap();
    &data[off[0].as_u64().unwrap() as usize..off[1].as_u64().unwrap() as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (ba, bb) = (synaptix_bundle::Bundle::open(&a[1]).expect("open a"), synaptix_bundle::Bundle::open(&a[2]).expect("open b"));
    let ((ha, da), (hb, db)) = (tensors(&ba), tensors(&bb));
    let mut names: Vec<&String> = ha.keys().chain(hb.keys().filter(|k| !ha.contains_key(*k))).collect();
    names.sort();
    let (mut same, mut diff, mut only) = (0usize, 0usize, 0usize);
    for n in names {
        match (ha.get(n), hb.get(n)) {
            (Some(x), Some(y)) => {
                if x["dtype"] != y["dtype"] || x["shape"] != y["shape"] {
                    println!("{n}: {} {} → {} {}", x["dtype"], x["shape"], y["dtype"], y["shape"]);
                    diff += 1;
                    continue;
                }
                let (p, q) = (bytes(x, da), bytes(y, db));
                if p == q {
                    same += 1;
                } else {
                    let d = p.iter().zip(q).filter(|(u, v)| u != v).count();
                    println!("{n}: {} {} байты отличаются: {:.2} %", x["dtype"], x["shape"], 100.0 * d as f64 / p.len() as f64);
                    diff += 1;
                }
            }
            (Some(_), None) => { println!("{n}: только в первом"); only += 1 }
            _ => { println!("{n}: только во втором"); only += 1 }
        }
    }
    println!("одинаковых {same}, отличаются {diff}, только в одном {only}");
}
