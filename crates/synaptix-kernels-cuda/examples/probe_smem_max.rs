#![cfg(feature = "cuda")]

use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute_enum};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

const SRC: &str = r#"
extern "C" __global__ void touch_smem(int *out, int nbytes) {
    extern __shared__ char s[];
    if (threadIdx.x == 0) {
        s[nbytes - 1] = (char)nbytes;
        out[0] = (int)s[nbytes - 1];
    }
}
"#;

fn main() {
    let ctx = synaptix_core::device::cuda::get(0).expect("ctx");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");

    let optin = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        .unwrap();
    let per_sm = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
        .unwrap();
    let reserved = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK)
        .unwrap();
    let default_max = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        .unwrap();
    println!(
        "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN   = {optin} bytes ({:.1} KB)",
        optin as f64 / 1024.0
    );
    println!(
        "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR= {per_sm} bytes ({:.1} KB)",
        per_sm as f64 / 1024.0
    );
    println!("RESERVED_SHARED_MEMORY_PER_BLOCK    = {reserved} bytes",);
    println!(
        "MAX_SHARED_MEMORY_PER_BLOCK(default)= {default_max} bytes ({:.1} KB)",
        default_max as f64 / 1024.0
    );

    let module = synaptix_kernels_cuda::kernels::compile::compile_module_with_opts(
        &ctx,
        SRC,
        "touch_smem.cu",
        &[],
        Some("sm_120a"),
    )
    .expect("compile");
    let f = module.load_function("touch_smem").expect("load");

    let mut out: CudaSlice<i32> = stream.alloc_zeros(1).unwrap();
    // Кандидаты: 99KB, 100KB, 101376 (ровно 256x128 s2 данные), 101424 (+mbar), 102KB, 116KB, 164KB, optin.
    let candidates: Vec<i32> = vec![
        99 * 1024,
        100 * 1024,
        101376,
        101376 + 48,
        102 * 1024,
        116 * 1024,
        164 * 1024,
        optin,
    ];
    println!("\n--- set_attribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) + launch test ---");
    for &cand in &candidates {
        let set_ok = f
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                cand,
            )
            .is_ok();
        if !set_ok {
            println!(
                "  {cand:>7} B ({:>5.1} KB): set_attribute FAIL",
                cand as f64 / 1024.0
            );
            continue;
        }
        let launch = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: cand as u32,
        };
        let nb = cand;
        let mut bld = stream.launch_builder(&f);
        bld.arg(&mut out).arg(&nb);
        let launch_res = unsafe { bld.launch(launch) };
        let sync_res = stream.synchronize();
        match (launch_res, sync_res) {
            (Ok(_), Ok(_)) => println!(
                "  {cand:>7} B ({:>5.1} KB): set OK + launch OK",
                cand as f64 / 1024.0
            ),
            (Ok(_), Err(e)) => println!(
                "  {cand:>7} B ({:>5.1} KB): set OK, launch issued, SYNC FAIL: {e:?}",
                cand as f64 / 1024.0
            ),
            (Err(e), _) => println!(
                "  {cand:>7} B ({:>5.1} KB): set OK, LAUNCH FAIL: {e:?}",
                cand as f64 / 1024.0
            ),
        }
    }
}
