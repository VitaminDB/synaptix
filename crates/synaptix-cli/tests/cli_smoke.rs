use std::path::PathBuf;

use synaptix_cli::commands::{convert, inspect};

fn ref_data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data")
        .join(name)
}

#[test]
fn t32_1_inspect_safetensors() {
    let path = ref_data("nn_heads/lm_head.safetensors");
    inspect::run(inspect::InspectArgs {
        file: path,
        verbose: false,
        filter: None,
    }).unwrap();
}

#[test]
fn t32_2_inspect_with_filter() {
    let path = ref_data("nn_heads/lm_head.safetensors");
    inspect::run(inspect::InspectArgs {
        file: path,
        verbose: true,
        filter: Some("weight".into()),
    }).unwrap();
}

#[test]
fn t32_3_convert_safetensors_to_syn_then_inspect() {
    let input = ref_data("nn_heads/lm_head.safetensors");
    let tmp = std::env::temp_dir().join("synaptix_cli_test.syn");
    if tmp.exists() {
        let _ = std::fs::remove_file(&tmp);
    }
    convert::run(convert::ConvertArgs {
        input: input.clone(),
        output: tmp.clone(),
        format: "syn".into(),
        arch: Some("test-arch".into()),
        component: Some("main".into()),
    }).unwrap();
    assert!(tmp.exists(), "bundle file not created");
    inspect::run(inspect::InspectArgs {
        file: tmp.clone(),
        verbose: false,
        filter: None,
    }).unwrap();
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn t32_4_inspect_unknown_format_errors() {
    let bad = std::env::temp_dir().join("foo.unknown");
    let r = inspect::run(inspect::InspectArgs {
        file: bad,
        verbose: false,
        filter: None,
    });
    assert!(r.is_err());
}

#[test]
fn t32_5_convert_unsupported_extension_errors() {
    let r = convert::run(convert::ConvertArgs {
        input: PathBuf::from("/tmp/foo.bin"),
        output: PathBuf::from("/tmp/foo.gguf"),
        format: "gguf".into(),
        arch: None,
        component: None,
    });
    assert!(r.is_err());
}
