fn main() {
    let p = std::env::args().nth(1).unwrap();
    let comp = std::env::args().nth(2);
    use synaptix_io::weights::WeightLoader;
    let mut l = synaptix_io::weights::syn_bundle::SynBundleLoader::open(&p).unwrap();
    if let Some(c) = &comp {
        l = l.with_component(c);
    }
    let names = l.names();
    let s: Vec<String> = names.iter().take(5).map(|s| s.to_string()).collect();
    println!("component={comp:?} total={} sample={s:?}", names.len());
}
