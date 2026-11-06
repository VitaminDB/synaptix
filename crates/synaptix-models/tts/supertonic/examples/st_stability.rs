//! Повторяемость на устройстве: N прогонов, расхождения промежуточных с первым.
use std::path::Path;
use synaptix_core::device::Device;
use synaptix_tts_supertonic::{Options, SupertonicPipeline};
fn rel(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) { d += ((x - y) as f64).powi(2); n += (*y as f64).powi(2); }
    (d / n.max(1e-30)).sqrt() as f32
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let p = SupertonicPipeline::from_syn(Path::new(&a[1]), Device::Cuda(0)).unwrap();
    let v = p.voice("M1").unwrap().clone();
    let o = Options::default();
    let first = p.debug_parts(&a[2], &a[3], &v, &o).unwrap();
    let h = |x: &[f32]| x.iter().map(|v| v.abs() as f64).sum::<f64>();
    eprintln!("ПЕРВЫЙ: dur {:.5} text {:.4} шаги [{}] pcm {:.4}", first.0, h(&first.1), first.2.iter().map(|s| format!("{:.2}", h(s))).collect::<Vec<_>>().join(" "), h(&first.3));
    if std::env::var_os("ONCE").is_some() { return; }
    for _ in 0..6 {
        let r = p.debug_parts(&a[2], &a[3], &v, &o).unwrap();
        let steps: Vec<String> = r.2.iter().zip(&first.2).map(|(x, y)| format!("{:.0e}", rel(x, y))).collect();
        eprintln!("dur {:.5}/{:.5} text {:.1e} шаги [{}] pcm {:.1e}", r.0, first.0, rel(&r.1, &first.1), steps.join(" "), rel(&r.3, &first.3));
    }
}
