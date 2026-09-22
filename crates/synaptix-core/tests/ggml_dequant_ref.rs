//! CPU-эталон деквантования ggml против gguf-py (`gguf.quants.dequantize`):
//! случайные валидные блоки 23 типов, ожидаемые f32 из python.
//! Фикстура: `reference_data/ggml_dequant_ref.bin`, записи
//! `[type u32][n u32][bytes u32][bytes…][n × f32 LE]`.

use synaptix_core::quant::ggml::GgmlType;
use synaptix_core::quant::ggml_dequant::dequantize;

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[test]
fn matches_gguf_py_reference() {
    let data = include_bytes!("reference_data/ggml_dequant_ref.bin");
    let mut o = 0usize;
    let mut checked = Vec::new();
    while o < data.len() {
        let ty_id = rd_u32(data, o);
        let n = rd_u32(data, o + 4) as usize;
        let nbytes = rd_u32(data, o + 8) as usize;
        o += 12;
        let src = &data[o..o + nbytes];
        o += nbytes;
        let want: Vec<f32> = (0..n).map(|i| f32::from_le_bytes(data[o + i * 4..o + i * 4 + 4].try_into().unwrap())).collect();
        o += n * 4;
        let ty = GgmlType::from_u32(ty_id).unwrap_or_else(|| panic!("тип {ty_id}"));
        assert_eq!(nbytes, ty.bytes_for(n), "{}: размер блоков", ty.name());
        let mut got = vec![0f32; n];
        dequantize(ty, src, n, &mut got).unwrap_or_else(|e| panic!("{}: {e}", ty.name()));
        let mut worst = 0f32;
        for i in 0..n {
            let (g, w) = (got[i], want[i]);
            let tol = 1e-6 * w.abs().max(1e-6);
            let diff = (g - w).abs();
            if diff > worst {
                worst = diff;
            }
            assert!(
                diff <= tol || g.to_bits() == w.to_bits(),
                "{} [{i}]: наш {g} против gguf-py {w} (diff {diff})",
                ty.name()
            );
        }
        checked.push((ty.name(), worst));
    }
    assert_eq!(checked.len(), 23, "типов в фикстуре: {checked:?}");
}
