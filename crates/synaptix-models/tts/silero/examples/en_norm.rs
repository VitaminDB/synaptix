fn main() {
    let a: Vec<String> = std::env::args().collect();
    synaptix_kernels_cpu::ensure_registered();
    let g = synaptix_tts_leokani::g2p::LeokaniG2P::from_syn(std::path::Path::new(&a[1])).expect("g2p");
    let n = g.normalize(&a[2], true);
    println!("{n:?}");
    println!("{}", g.phonemize(&a[2]));
}
