//! Probe: подтвердить, что `ldmatrix.x4.b16` (A) и `ldmatrix.x2.b16` (B) воспроизводят
//! РОВНО тот fragment, который ожидает FP4-MMA `m16n8k64` (= наш bit-exact manual-read из
//! `mma_gemm_shuf_2dr_impl`). Layout выведен из CUTLASS `mma_traits_sm120.hpp` +
//! `copy_traits_sm75.hpp` (см. memory `synaptix-ldmatrix-fp4-fragment-decoded-2026`).
//!
//! W-тайл = 16 строк × 32 байта (= наш W-repack `(N/16,K/64,16,32)`, один chunk = 512 байт).
//! X-тайл = 8 строк × 32 байта (256 байт).
//!
//! Заполняем тайлы так, что каждый u32 = его собственный byte-offset/4 → проверка вскрывает
//! точный offset, который вернул ldmatrix.
//!
//! cargo run -p synaptix-kernels-cuda --features cuda --release --example probe_ldmatrix
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use synaptix_kernels_cuda::kernels::compile::{compile_module_with_opts, load_fn};

const SRC: &str = r#"
extern "C" __global__ void probe_ldmatrix(
    const unsigned char* __restrict__ w_tile,   // 512 байт = 16 строк × 32
    const unsigned char* __restrict__ x_tile,   // 256 байт = 8 строк × 32
    unsigned int* __restrict__ out_a,            // 32 lane × 4
    unsigned int* __restrict__ out_b)            // 32 lane × 2
{
    __shared__ unsigned char sw[512];
    __shared__ unsigned char sx[256];
    unsigned int lane = threadIdx.x & 31u;
    for (unsigned int i = lane * 4u; i < 512u; i += 128u)
        *(unsigned int*)(sw + i) = *(const unsigned int*)(w_tile + i);
    for (unsigned int i = lane * 4u; i < 256u; i += 128u)
        *(unsigned int*)(sx + i) = *(const unsigned int*)(x_tile + i);
    __syncwarp();

    // A: матрицы 0..3 = TL/BL/TR/BR; lane t → row (t&7) матрицы (t>>3).
    // addr = (lane&15)*32 + (lane>>4)*16.
    unsigned int a_off = (lane & 15u) * 32u + (lane >> 4) * 16u;
    unsigned int sa = (unsigned int)__cvta_generic_to_shared(sw + a_off);
    unsigned int a0, a1, a2, a3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(sa));
    out_a[lane * 4u + 0u] = a0;
    out_a[lane * 4u + 1u] = a1;
    out_a[lane * 4u + 2u] = a2;
    out_a[lane * 4u + 3u] = a3;

    // B: матрицы 0..1 = cols 0-7 / 8-15 b16; lane t → row (t&7) матрицы ((t&8)?1:0).
    unsigned int b_off = (lane & 7u) * 32u + ((lane & 8u) ? 16u : 0u);
    unsigned int sb = (unsigned int)__cvta_generic_to_shared(sx + b_off);
    unsigned int b0, b1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];\n"
        : "=r"(b0), "=r"(b1) : "r"(sb));
    out_b[lane * 2u + 0u] = b0;
    out_b[lane * 2u + 1u] = b1;
}
"#;

fn main() {
    let ctx = synaptix_core::device::cuda::get(0).expect("cuda ctx");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    let module = compile_module_with_opts(&ctx, SRC, "probe_ldmatrix", &[], Some("sm_120a"))
        .expect("compile probe");
    let kfn = load_fn(&module, "probe_ldmatrix").expect("load probe");

    // Каждый u32 = его byte-offset/4 → значение прямо кодирует, откуда взялись данные.
    let w_host: Vec<u32> = (0..128u32).collect(); // 512 байт
    let x_host: Vec<u32> = (0..64u32).map(|j| 0x1000 + j).collect(); // 256 байт
    let w_dev: CudaSlice<u32> = stream.clone_htod(&w_host).unwrap();
    let x_dev: CudaSlice<u32> = stream.clone_htod(&x_host).unwrap();
    let mut out_a: CudaSlice<u32> = stream.alloc_zeros(128).unwrap();
    let mut out_b: CudaSlice<u32> = stream.alloc_zeros(64).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(&kfn);
    b.arg(&w_dev).arg(&x_dev).arg(&mut out_a).arg(&mut out_b);
    unsafe { b.launch(cfg).expect("launch probe") };
    stream.synchronize().unwrap();

    let got_a = stream.clone_dtoh(&out_a).unwrap();
    let got_b = stream.clone_dtoh(&out_b).unwrap();

    // Эталон = manual-read из mma_gemm_shuf_2dr_impl (значение u32 = его offset/4).
    // A: top = (lane>>2)*32 + (lane&3)*4 ; bot = ((lane>>2)+8)*32 + (lane&3)*4.
    //   a0=@top a1=@bot a2=@top+16 a3=@bot+16. Значение = offset/4.
    let exp_a = |lane: u32| -> [u32; 4] {
        let kt = lane >> 2;
        let mt = lane & 3;
        let top = kt * 32 + mt * 4;
        let bot = (kt + 8) * 32 + mt * 4;
        [top / 4, bot / 4, (top + 16) / 4, (bot + 16) / 4]
    };
    // B: b0 = (lane>>2)*32 + (lane&3)*4 ; b1 = +16. Значение = 0x1000 + offset/4.
    let exp_b = |lane: u32| -> [u32; 2] {
        let nt = lane >> 2;
        let mt = lane & 3;
        let o = nt * 32 + mt * 4;
        [0x1000 + o / 4, 0x1000 + (o + 16) / 4]
    };

    let mut a_ok = true;
    let mut b_ok = true;
    println!("=== A (ldmatrix.x4.b16 vs manual 2dr A-fragment) ===");
    for lane in 0..32u32 {
        let e = exp_a(lane);
        let g = [
            got_a[(lane * 4) as usize],
            got_a[(lane * 4 + 1) as usize],
            got_a[(lane * 4 + 2) as usize],
            got_a[(lane * 4 + 3) as usize],
        ];
        if g != e {
            a_ok = false;
            // детект перестановки регистров
            let perm: Vec<i32> = g
                .iter()
                .map(|v| {
                    e.iter()
                        .position(|x| x == v)
                        .map(|p| p as i32)
                        .unwrap_or(-1)
                })
                .collect();
            println!("lane {lane:2}: got {g:?} exp {e:?}  perm={perm:?}");
        }
    }
    println!("A: {}", if a_ok { "MATCH ✓" } else { "MISMATCH ✗" });

    println!("=== B (ldmatrix.x2.b16 vs manual 2dr B-fragment) ===");
    for lane in 0..32u32 {
        let e = exp_b(lane);
        let g = [got_b[(lane * 2) as usize], got_b[(lane * 2 + 1) as usize]];
        if g != e {
            b_ok = false;
            println!("lane {lane:2}: got {g:?} exp {e:?}");
        }
    }
    println!("B: {}", if b_ok { "MATCH ✓" } else { "MISMATCH ✗" });

    if a_ok && b_ok {
        println!("\nPROBE PASS — ldmatrix воспроизводит FP4-MMA фрагмент. 2drl можно строить.");
    } else {
        println!("\nPROBE FAIL — нужна коррекция addr/perm маппинга перед 2drl.");
        std::process::exit(1);
    }
}
