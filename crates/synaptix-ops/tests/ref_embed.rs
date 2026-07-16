use synaptix_kernels_cpu::ensure_registered;
use synaptix_ops::embed::{patch_embed_2d, timestep_embedding, token_embedding};
use synaptix_test_utils::{assert_allclose, load_case};

fn setup() { ensure_registered(); }

#[test]
fn t06_1_token_embedding() {
    setup();
    let t = load_case("embed", "token_embedding");
    let result = token_embedding(&t["input_ids"], &t["weight"]).unwrap();
    assert_allclose(&result, &t["output"], 1e-6, 1e-6);
}

#[test]
fn t06_2_patch_embed_2d() {
    setup();
    let t = load_case("embed", "patch_embed_2d");
    let input = &t["input"];
    let weight = &t["weight"];
    let bias = &t["bias"];
    let expected = &t["output"];
    let patch_size = weight.dims()[2];
    let result = patch_embed_2d(input, weight, Some(bias), patch_size, None).unwrap();
    let batch = expected.dims()[0];
    let num_patches = expected.dims()[1];
    let embed_dim = expected.dims()[2];
    let result_flat = result.reshape((batch, num_patches, embed_dim)).unwrap();
    assert_allclose(&result_flat, expected, 1e-5, 1e-5);
}

#[test]
fn t06_4_timestep_embedding() {
    setup();
    let t = load_case("embed", "timestep_embedding");
    let timesteps = &t["timesteps"];
    let expected = &t["output"];
    let dim = expected.dims()[1];
    let result = timestep_embedding(timesteps, dim, 10000.0).unwrap();
    assert_allclose(&result, expected, 1e-4, 1e-4);
}
