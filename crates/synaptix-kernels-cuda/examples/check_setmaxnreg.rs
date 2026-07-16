#![cfg(feature = "cuda")]
const SRC: &str = r#"
extern "C" __global__ void smaxnreg_probe(int *out) {
    if (threadIdx.x < 128)
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    else
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
    if (threadIdx.x == 0) out[0] = 1;
}
"#;
fn main() {
    let ctx = synaptix_core::device::cuda::get(0).expect("ctx");
    match synaptix_kernels_cuda::kernels::compile::compile_module_with_opts(&ctx, SRC, "smaxnreg.cu", &[], Some("sm_120a")) {
        Ok(m) => match m.load_function("smaxnreg_probe") {
            Ok(_) => println!("OK: setmaxnreg компилится под sm_120a — register reallocation доступна → warp-spec может пробить cliff"),
            Err(e) => println!("LOAD FAIL: {e:?}"),
        },
        Err(e) => println!("COMPILE FAIL (setmaxnreg НЕ поддержан sm_120 → warp-spec НЕ пробьёт RF-cliff): {e}"),
    }
}
