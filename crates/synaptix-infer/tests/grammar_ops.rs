//! D6: regex → байтовый NFA (OutlinesConstraint) + Grammar::allowed_tokens.

use synaptix_infer::structured::grammar::Grammar;
use synaptix_infer::structured::outlines::OutlinesConstraint;

#[test]
fn t43_1_literal_sequence() {
    let c = OutlinesConstraint::new("abc");
    assert_eq!(c.allowed_bytes(b""), vec![b'a']);
    assert_eq!(c.allowed_bytes(b"ab"), vec![b'c']);
    assert!(c.is_match(b"abc"));
    assert!(!c.is_match(b"ab"));
    assert!(!c.is_match(b"abcd"));
    assert!(c.allowed_bytes(b"abc").is_empty(), "ничего после полного совпадения");
}

#[test]
fn t43_2_char_class_plus() {
    let c = OutlinesConstraint::new("[a-c]+");
    assert_eq!(c.allowed_bytes(b""), vec![b'a', b'b', b'c']);
    assert!(c.is_match(b"a"));
    assert!(c.is_match(b"abccba"));
    assert!(!c.is_match(b""));
    assert!(!c.is_match(b"ax"));
    // После одного символа можно продолжать классом.
    assert_eq!(c.allowed_bytes(b"ab"), vec![b'a', b'b', b'c']);
}

#[test]
fn t43_3_alternation() {
    let c = OutlinesConstraint::new("cat|dog");
    assert_eq!(c.allowed_bytes(b""), vec![b'c', b'd']);
    assert!(c.accepts("cat"));
    assert!(c.accepts("dog"));
    assert!(!c.accepts("cot"));
    assert_eq!(c.allowed_bytes(b"c"), vec![b'a']);
}

#[test]
fn t43_4_optional_and_star() {
    let c = OutlinesConstraint::new("https?");
    assert!(c.accepts("http"));
    assert!(c.accepts("https"));
    assert!(!c.accepts("htt"));

    let star = OutlinesConstraint::new("ab*c");
    assert!(star.accepts("ac"));
    assert!(star.accepts("abc"));
    assert!(star.accepts("abbbbc"));
    assert!(!star.accepts("abb"));
}

#[test]
fn t43_5_dot_and_negation() {
    let dot = OutlinesConstraint::new("a.c");
    assert!(dot.accepts("axc"));
    assert!(dot.accepts("a c"));
    assert!(!dot.accepts("ac"));

    let neg = OutlinesConstraint::new("[^0-9]");
    let allowed = neg.allowed_bytes(b"");
    assert!(allowed.contains(&b'a'));
    assert!(!allowed.contains(&b'5'));
}

#[test]
fn t43_6_invalid_regex_matches_nothing() {
    let bad = OutlinesConstraint::new("a(bc");
    assert!(bad.allowed_bytes(b"").is_empty());
    assert!(!bad.is_match(b"abc"));
    assert!(OutlinesConstraint::compile("a(bc").is_err());
    assert!(OutlinesConstraint::compile("a(bc)").is_ok());
}

#[test]
fn t43_7_grammar_regex_allowed_tokens() {
    let g = Grammar::regex("[0-9]+");
    let mut allowed = g.allowed_tokens(&[], 256);
    allowed.sort_unstable();
    assert_eq!(allowed, (b'0'..=b'9').map(|b| b as u32).collect::<Vec<_>>());
    // После цифры — снова цифры.
    let mut after = g.allowed_tokens(&[b'7' as u32], 256);
    after.sort_unstable();
    assert_eq!(after, (b'0'..=b'9').map(|b| b as u32).collect::<Vec<_>>());
    // vocab_size фильтрует байты ≥ предела.
    assert!(g.allowed_tokens(&[], 48).is_empty(), "цифры (48..58) вне vocab=48");
}

#[test]
fn t43_8_grammar_rules_alternation() {
    let mut g = Grammar::new("unused");
    g.add_rule("cat").add_rule("dog");
    let mut start = g.allowed_tokens(&[], 256);
    start.sort_unstable();
    assert_eq!(start, vec![b'c' as u32, b'd' as u32]);
}
