use synaptix_infer::sampling::grammar_mask::GrammarMaskProcessor;
use synaptix_infer::sampling::{LogitProcessor, ProcessorContext};
use synaptix_infer::structured::{JsonSchemaConstraint, JsonState, LinearGrammar};

#[test]
fn t22_1_linear_grammar() {
    let mut g = LinearGrammar::new(vec![
        vec![1, 2, 3],
        vec![10, 20],
        vec![100],
    ]);
    assert_eq!(g.allowed_tokens(), vec![1, 2, 3]);
    assert!(g.advance(2));
    assert_eq!(g.allowed_tokens(), vec![10, 20]);
    assert!(!g.advance(11));
    assert!(g.advance(20));
    assert_eq!(g.allowed_tokens(), vec![100]);
    assert!(g.advance(100));
    assert!(g.is_finished());
    assert_eq!(g.allowed_tokens(), Vec::<u32>::new());
}

#[test]
fn t22_2_grammar_mask_processor() {
    let mut processor = GrammarMaskProcessor::new(vec![1, 5, 9]);
    let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let ctx = ProcessorContext { input_ids: Vec::new(), step: 0, batch_idx: 0 };
    processor.process(&mut logits, &ctx).unwrap();
    for (i, &l) in logits.iter().enumerate() {
        if [1, 5, 9].contains(&(i as u32)) {
            assert!(l.is_finite(), "i={} should be finite, got {}", i, l);
        } else {
            assert_eq!(l, f32::NEG_INFINITY, "i={} should be -inf, got {}", i, l);
        }
    }
}

#[test]
fn t22_3_json_schema_simple_object() {
    let schema_str = r#"{"type": "object", "required": ["name"]}"#;
    let mut con = JsonSchemaConstraint::from_str(schema_str).unwrap();
    assert_eq!(con.state, JsonState::ExpectStart);

    for ch in r#"{"name":"Alice"}"#.chars() {
        assert!(con.advance(ch), "failed on char '{}', state={:?}", ch, con.state);
    }
    assert!(con.is_done());
    assert!(con.used_keys.contains(&"name".to_string()));
}

#[test]
fn t22_4_json_schema_rejects_invalid() {
    let mut con = JsonSchemaConstraint::from_str("{}").unwrap();
    assert!(con.advance('{'));
    assert!(!con.advance('!'));
}

#[test]
fn t22_5_json_schema_required_missing() {
    let schema_str = r#"{"type": "object", "required": ["age"]}"#;
    let mut con = JsonSchemaConstraint::from_str(schema_str).unwrap();
    for ch in r#"{"name":"Alice""#.chars() {
        assert!(con.advance(ch));
    }
    assert!(!con.advance('}'));
}
