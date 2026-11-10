//! Проба низкобитного кванта на настоящем весе: матрица NVFP4 из чекпойнта
//! compressed-tensors (`<имя>.weight_packed`, `.weight_scale`,
//! `.weight_global_scale`) деквантуется и кодируется в IQ3_S-128, IQ2_XXS и
//! IQ1_S; печатается относительная ошибка весов каждого формата.
//!
//!   lowbit_probe <каталог HF> <имя матрицы без суффикса> [строк, 256]
//!
//! NVFP4 по эталону sglang: младший полубайт — чётный столбец, вес =
//! e2m1 · weight_scale(e4m3, блок 16) / weight_global_scale.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use synaptix_core::quant::ggml::{quantize_iq3_s128, GgmlType};
use synaptix_core::quant::ggml_dequant::dequantize;
use synaptix_core::quant::ggml_lowbit::quantize_lowbit;



/// Байты тензора `name` и его форма.
fn tensor(dir: &Path, index: &serde_json::Value, name: &str) -> Result<(Vec<u8>, Vec<usize>), Box<dyn std::error::Error>> {
    let file = index["weight_map"][name].as_str().ok_or_else(|| format!("{name}: нет в индексе"))?;
    let mut f = std::fs::File::open(dir.join(file))?;
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8)?;
    let hl = u64::from_le_bytes(b8);
    let mut hb = vec![0u8; hl as usize];
    f.read_exact(&mut hb)?;
    let header: serde_json::Value = serde_json::from_slice(&hb)?;
    let e = &header[name];
    let off = e["data_offsets"].as_array().ok_or("data_offsets")?;
    let (a, b) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
    let shape = e["shape"].as_array().ok_or("shape")?.iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let mut data = vec![0u8; (b - a) as usize];
    f.seek(SeekFrom::Start(8 + hl + a))?;
    f.read_exact(&mut data)?;
    Ok((data, shape))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        return Err("usage: lowbit_probe <каталог HF> <имя матрицы> [строк, 256]".into());
    }
    let dir = Path::new(&a[1]);
    let name = &a[2];
    let rows: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
    let (packed, shape) = tensor(dir, &index, &format!("{name}.weight_packed"))?;
    let (scale, _) = tensor(dir, &index, &format!("{name}.weight_scale"))?;
    let (gs, _) = tensor(dir, &index, &format!("{name}.weight_global_scale"))?;
    let g = f32::from_le_bytes([gs[0], gs[1], gs[2], gs[3]]);
    let (n_all, k) = (shape[0], shape[1] * 2);
    let n = rows.min(n_all);
    // Строки равномерно по матрице.
    let mut x = vec![0f32; n * k];
    synaptix_core::quant::nvfp4_ct::dequant_rows(&packed, &scale, g, k, (0..n).map(|r| r * n_all / n), &mut x);
    let rms = (x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len() as f64).sqrt();
    let zeros = x.iter().filter(|v| **v == 0.0).count() as f64 / x.len() as f64;
    println!("{name}: [{n_all}, {k}], строк в пробе {n}, СКО {rms:.5}, нулей {:.1} %, глобальный масштаб {g:.3}", zeros * 100.0);
    let report = |label: &str, ty: GgmlType, q: Vec<u8>| {
        let mut y = vec![0f32; n * k];
        dequantize(ty, &q, n * k, &mut y).unwrap();
        let num: f64 = x.iter().zip(&y).map(|(a, b)| ((a - b) * (a - b)) as f64).sum();
        let den: f64 = x.iter().map(|a| (a * a) as f64).sum();
        println!("  {label:<10} {:.2} бита/вес: отн. ошибка {:.4}", q.len() as f64 * 8.0 / (n * k) as f64, (num / den).sqrt());
    };
    report("IQ3_S-128", GgmlType::Iq3S128, quantize_iq3_s128(&x, n, k, None));
    report("IQ2_XXS", GgmlType::Iq2Xxs, quantize_lowbit(GgmlType::Iq2Xxs, &x, n, k, None));
    report("IQ1_S", GgmlType::Iq1S, quantize_lowbit(GgmlType::Iq1S, &x, n, k, None));
    Ok(())
}
