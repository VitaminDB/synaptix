//! Probe: подтвердить TMA-механику (cp.async.bulk.tensor + mbarrier) в нашем стеке.
//! Host создаёт CUtensorMap через cudarc::driver::sys::cuTensorMapEncodeTiled (2D uint8,
//! swizzle NONE), кладёт его в gmem; kernel грузит box тайла в smem через TMA + mbarrier,
//! выгружает в gmem; host сверяет байты с эталоном (matrix[y+i][x+j]).
//!
//! Де-рискует: создание дескриптора из Rust, TMA-PTX, mbarrier-протокол (init/arrive.expect_tx/
//! try_wait.parity). Это фундамент для TMA-GEMM (2dt). Аналог probe_ldmatrix.
//!
//! cargo run -p synaptix-kernels-cuda --features cuda --release --example probe_tma
#![cfg(feature = "cuda")]

use std::ffi::c_void;

use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use synaptix_kernels_cuda::kernels::compile::{compile_module_with_opts, load_fn};

const ROWS: u32 = 64;
const COLS: u32 = 128;
const BOX_ROWS: u32 = 32;
const BOX_COLS: u32 = 64;
const BOX_BYTES: u32 = BOX_ROWS * BOX_COLS;

const SRC: &str = r#"
struct __align__(64) TmaDesc { unsigned long long opaque[16]; };

extern "C" __global__ void probe_tma(
    const unsigned char* __restrict__ tensor_map,   // дескриптор в gmem (128 байт)
    unsigned char* __restrict__ out,                // BOX_BYTES
    unsigned int x, unsigned int y)
{
    constexpr unsigned int BOX_BYTES = 32u * 64u;
    __shared__ __align__(128) unsigned char smem[BOX_BYTES];
    __shared__ __align__(8) unsigned long long mbar;

    unsigned int tid = threadIdx.x;
    unsigned int smem_addr = (unsigned int)__cvta_generic_to_shared(smem);
    unsigned int bar_addr  = (unsigned int)__cvta_generic_to_shared(&mbar);

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bar_addr));
    }
    __syncthreads();

    if (tid == 0) {
        unsigned long long state;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
                     : "=l"(state) : "r"(bar_addr), "r"(BOX_BYTES));
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3}], [%4];\n"
          :: "r"(smem_addr), "l"((unsigned long long)tensor_map),
             "r"(x), "r"(y), "r"(bar_addr) : "memory");
    }
    asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" :: "r"(bar_addr) : "memory");

    for (unsigned int i = tid * 4u; i < BOX_BYTES; i += blockDim.x * 4u) {
        *(unsigned int*)(out + i) = *(unsigned int*)(smem + i);
    }
}
"#;

fn main() {
    let ctx = synaptix_core::device::cuda::get(0).expect("cuda ctx");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    let module = compile_module_with_opts(&ctx, SRC, "probe_tma", &[], Some("sm_120a"))
        .expect("compile probe_tma");
    let kfn = load_fn(&module, "probe_tma").expect("load probe_tma");

    // Матрица ROWS×COLS, байт[r][c] = детерминированный хеш позиции.
    let mat: Vec<u8> = (0..(ROWS * COLS))
        .map(|idx| {
            let r = idx / COLS;
            let c = idx % COLS;
            ((r.wrapping_mul(131).wrapping_add(c.wrapping_mul(7))) & 0xFF) as u8
        })
        .collect();
    let mat_dev: CudaSlice<u8> = stream.clone_htod(&mat).unwrap();

    // Дескриптор: 2D uint8, dim0=cols (fastest), dim1=rows; box [BOX_COLS, BOX_ROWS].
    let (gptr, _rec) = mat_dev.device_ptr(&stream);
    let mut map = sys::CUtensorMap { opaque: [0u64; 16] };
    let global_dim: [u64; 2] = [COLS as u64, ROWS as u64];
    let global_strides: [u64; 1] = [COLS as u64]; // байтовый шаг dim1 (строки), кратен 16
    let box_dim: [u32; 2] = [BOX_COLS, BOX_ROWS];
    let elem_strides: [u32; 2] = [1, 1];
    let res = unsafe {
        sys::cuTensorMapEncodeTiled(
            &mut map as *mut _,
            sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            gptr as *mut c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            elem_strides.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    assert_eq!(
        res,
        sys::CUresult::CUDA_SUCCESS,
        "cuTensorMapEncodeTiled failed: {res:?}"
    );

    // Дескриптор → gmem (128 байт).
    let desc_bytes: Vec<u8> = map.opaque.iter().flat_map(|w| w.to_le_bytes()).collect();
    let desc_dev: CudaSlice<u8> = stream.clone_htod(&desc_bytes).unwrap();

    let mut out: CudaSlice<u8> = stream.alloc_zeros(BOX_BYTES as usize).unwrap();

    // Грузим box по (x=col, y=row).
    let (x, y) = (32u32, 8u32);
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&kfn);
    b.arg(&desc_dev).arg(&mut out).arg(&x).arg(&y);
    unsafe { b.launch(cfg).expect("launch probe_tma") };
    stream.synchronize().unwrap();

    let got = stream.clone_dtoh(&out).unwrap();

    let mut ok = true;
    let mut shown = 0;
    for i in 0..BOX_ROWS {
        for j in 0..BOX_COLS {
            let exp = mat[((y + i) * COLS + (x + j)) as usize];
            let g = got[(i * BOX_COLS + j) as usize];
            if g != exp {
                ok = false;
                if shown < 8 {
                    println!("mismatch [{i},{j}]: got {g} exp {exp}");
                    shown += 1;
                }
            }
        }
    }
    if ok {
        println!(
            "PROBE TMA PASS — cp.async.bulk.tensor + mbarrier работает, дескриптор корректен."
        );
    } else {
        println!("PROBE TMA FAIL");
        std::process::exit(1);
    }
}
