//! Стоимость пути Kaldi (transition-id по кадрам) в нашей модели:
//! акустика·0.1 + веса переходов. `tid_path <model.mfa> <feats.txt> <tids.txt>`
fn nums(path: &str) -> Vec<f32> {
    std::fs::read_to_string(path).unwrap().split_whitespace().filter_map(|t| t.parse().ok()).collect()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let m = synaptix_align_mfa::MfaModel::load(std::path::Path::new(&a[1])).unwrap();
    let feats = nums(&a[2]);
    let tids: Vec<usize> = nums(&a[3]).into_iter().map(|x| x as usize).collect();
    let (mut ac, mut tr) = (0f32, 0f32);
    for (t, &tid) in tids.iter().enumerate() {
        let (p, hs, pdf, j) = m.transitions.decode_tid(tid).unwrap();
        ac += 0.1 * m.gmms.loglike(pdf, &feats[t * 40..(t + 1) * 40]);
        let topo = m.transitions.topology(p).unwrap();
        let st = &topo.states[hs];
        let lp: Vec<f32> = (0..st.transitions.len()).map(|k| m.transitions.log_prob(p, hs, pdf, k).unwrap()).collect();
        let p_self: f32 = st.transitions.iter().zip(&lp).filter(|((d, _), _)| *d == hs).map(|(_, l)| l.exp()).sum();
        let l = lp[j];
        let w = if st.transitions[j].0 == hs { 0.1 * l } else { let r = (1.0 - p_self).max(1e-10).ln(); l - r + 0.1 * r };
        tr += w;
    }
    println!("путь Kaldi: акустика·0.1 {ac:.2}, переходы {tr:.2}, сумма {:.2}", ac + tr);
}
