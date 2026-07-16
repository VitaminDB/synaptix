use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

fn main() {
    let t = HfTokenizer::from_file("models/gemma-3-12b-qat/tokenizer.json")
        .expect("tokenizer");
    let p = "3D cartoon boy says in Russian with a clear warm child voice: 'Привет, белочка! Какой чудесный день!' The squirrel replies in Russian: 'Привет! Пойдём гулять!' Clear Russian children voices, quiet background.";
    let enc = t.encode(p, false).expect("encode");
    println!("n={}", enc.ids.len());
    for (i, id) in enc.ids.iter().enumerate().take(16) {
        println!("{i} {id}");
    }
}
