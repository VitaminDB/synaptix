//! Сверка текстового BPE и грамматики промпта с эталоном релиза.
//!
//! Ожидаемые ID сняты питоновским `YuE2TextTokenizer` (tiktoken) на тех же
//! строках — см. `scratchpad/ref/tokenizer_ref.json`.

use std::path::PathBuf;

use synaptix_music_yue2::loader::read_bundle_file;
use synaptix_music_yue2::protocol::{
    negative_prefix, token_prefixes, Cot, SongRequest, ABC_END, ABC_START, EOD, MUSIC_START,
};
use synaptix_music_yue2::tokenizer::Yue2Tokenizer;

fn bundle() -> Option<PathBuf> {
    let p = std::env::var("SYN_YUE2_BUNDLE").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("Storage/syn_models/yue2-3b.syn")
    });
    p.exists().then_some(p)
}

fn tokenizer() -> Option<Yue2Tokenizer> {
    let path = bundle()?;
    let raw = read_bundle_file(&path, "qwen.tiktoken").expect("qwen.tiktoken в бандле");
    Some(Yue2Tokenizer::from_tiktoken_bytes(&raw).expect("токенизатор"))
}

fn request() -> SongRequest {
    SongRequest {
        style: "english, pop, female vocal".into(),
        lyrics: "[verse]\nTonight I'm awake\n".into(),
        cot: Cot::Full,
        ..SongRequest::default()
    }
}

const ABC: &str = "X:1\nK:C\n|\"C\"c2 e2|\n";
const ABC_IDS: [u32; 14] =
    [55, 25, 16, 198, 42, 55992, 198, 47576, 34, 96946, 17, 384, 17, 7360];

#[test]
fn encodes_like_reference() {
    let Some(tok) = tokenizer() else {
        eprintln!("[yue2-tok] бандла нет — тест пропущен");
        return;
    };
    assert_eq!(tok.encode("Hello world"), vec![9707, 1879]);
    // Цифры у Qwen идут по одной — на ByteLevel-регэкспе GPT-2 было бы иначе.
    assert_eq!(tok.encode("X:1\nL:1/8\nQ:1/4=120")[..14], [55, 25, 16, 198, 43, 25, 16, 14, 23, 198, 48, 25, 16, 14]);
    assert_eq!(tok.encode(ABC), ABC_IDS.to_vec());
    // Сокращения с апострофом — отдельная ветка паттерна.
    let text = request().text();
    assert_eq!(
        tok.encode(&text)[..14],
        [31115, 264, 43221, 12, 3401, 657, 19360, 45840, 11, 1221, 6923, 4627, 448, 34647]
    );
}

#[test]
fn decodes_back() {
    let Some(tok) = tokenizer() else { return };
    assert_eq!(tok.decode(&ABC_IDS), ABC);
    // Специальные токены печатаются своими именами, кодек-токены пропускаются.
    assert_eq!(tok.decode(&[ABC_START, ABC_END]), "<abc></abc>");
    assert_eq!(tok.decode(&[9707, 200_000]), "Hello");
}

#[test]
fn prefixes_match_reference() {
    let Some(tok) = tokenizer() else { return };
    let request = request();

    // Полный режим без партитуры — префикс кончается на `<abc>`.
    let head = token_prefixes(&request, &tok, None).expect("префикс");
    assert_eq!(head[0], EOD);
    assert_eq!(*head.last().unwrap(), ABC_START);
    assert_eq!(head.len(), 44);

    // С партитурой — она вклеивается между `<abc>` и `</abc><music>`.
    let full = token_prefixes(&request, &tok, Some(&ABC_IDS)).expect("префикс");
    assert_eq!(full.len(), 60);
    assert_eq!(&full[..head.len()], &head[..]);
    assert_eq!(&full[head.len()..head.len() + ABC_IDS.len()], &ABC_IDS[..]);
    assert_eq!(&full[full.len() - 2..], &[ABC_END, MUSIC_START]);

    // Отрицательная ветка CFG: та же инструкция и та же партитура, без тегов.
    let negative = negative_prefix(&request, &tok, Some(&ABC_IDS)).expect("отрицательный префикс");
    assert_eq!(negative.len(), 38);
    assert_eq!(negative[0], EOD);
    assert_eq!(&negative[negative.len() - 2..], &[ABC_END, MUSIC_START]);

    // Без партитуры символьный CFG невозможен — это ошибка, а не «как-нибудь».
    assert!(negative_prefix(&request, &tok, None).is_err());

    // Режим off: партитуры нет вообще, музыка начинается сразу. Длина сверена
    // с эталоном на его же запросе (короче инструкция — короче префикс).
    let off = SongRequest {
        style: "english, pop".into(),
        lyrics: "la la la".into(),
        cot: Cot::Off,
        ..SongRequest::default()
    };
    let ids = token_prefixes(&off, &tok, None).expect("префикс off");
    assert_eq!(ids.len(), 29);
    assert_eq!(&ids[ids.len() - 3..], &[ABC_START, ABC_END, MUSIC_START]);
    let negative_off = negative_prefix(&off, &tok, None).expect("отрицательный off");
    assert_eq!(negative_off.len(), 12);
    assert_eq!(*negative_off.last().unwrap(), MUSIC_START);
}

#[test]
fn abc_ids_must_stay_in_text_vocabulary() {
    let Some(tok) = tokenizer() else { return };
    let request = request();
    // Кодек-токен в партитуре — сломанный префикс; ловим на входе.
    assert!(token_prefixes(&request, &tok, Some(&[55, 151_900])).is_err());
}
