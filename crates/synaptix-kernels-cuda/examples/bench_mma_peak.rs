//! ISA-пик mma.sync.m16n8k16 bf16→f32 на sm_120: чистый регистровый цикл без
//! памяти. Истинный потолок для bf16-attention/GEMM через mma.sync (не wgmma).
#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::time::Instant;

const SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void mma_peak(float* out, int iters) {
  unsigned a0=threadIdx.x, a1=threadIdx.x+1, a2=threadIdx.x+2, a3=threadIdx.x+3;
  unsigned b0=threadIdx.x+5, b1=threadIdx.x+7;
  float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
  float e0=0.f,e1=0.f,e2=0.f,e3=0.f;
  float f0=0.f,f1=0.f,f2=0.f,f3=0.f;
  float g0=0.f,g1=0.f,g2=0.f,g3=0.f;
  for (int i = 0; i < iters; ++i) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(e0),"+f"(e1),"+f"(e2),"+f"(e3) : "r"(a1),"r"(a2),"r"(a3),"r"(a0),"r"(b1),"r"(b0));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(f0),"+f"(f1),"+f"(f2),"+f"(f3) : "r"(a2),"r"(a3),"r"(a0),"r"(a1),"r"(b0),"r"(b1));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(g0),"+f"(g1),"+f"(g2),"+f"(g3) : "r"(a3),"r"(a0),"r"(a1),"r"(a2),"r"(b1),"r"(b0));
  }
  if (d0+e0+f0+g0 == 12345.f) out[0] = d1+e1+f1+g1;
}
extern "C" __global__ void mma_peak_f16acc(float* out, int iters) {
  unsigned a0=threadIdx.x, a1=threadIdx.x+1, a2=threadIdx.x+2, a3=threadIdx.x+3;
  unsigned b0=threadIdx.x+5, b1=threadIdx.x+7;
  unsigned c0=0u, c1=0u;
  unsigned e0=0u, e1=0u;
  unsigned f0=0u, f1=0u;
  unsigned g0=0u, g1=0u;
  for (int i = 0; i < iters; ++i) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
      : "+r"(c0),"+r"(c1) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
      : "+r"(e0),"+r"(e1) : "r"(a1),"r"(a2),"r"(a3),"r"(a0),"r"(b1),"r"(b0));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
      : "+r"(f0),"+r"(f1) : "r"(a2),"r"(a3),"r"(a0),"r"(a1),"r"(b0),"r"(b1));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
      : "+r"(g0),"+r"(g1) : "r"(a3),"r"(a0),"r"(a1),"r"(a2),"r"(b1),"r"(b0));
  }
  if (c0+e0+f0+g0 == 12345u) out[0] = (float)(c1+e1+f1+g1);
}
"#;

fn main() {
    let ctx = synaptix_core::device::cuda::get(0).expect("ctx");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    let module = synaptix_kernels_cuda::kernels::compile::compile_module(&ctx, SRC, "mma_peak.cu").expect("compile");
    let out = stream.alloc_zeros::<f32>(4).unwrap();
    let iters: i32 = 20000;
    // grid: насытить все SM варпами: 82 SM × 12 warps = 4 блока по 256 на SM
    let cfg = LaunchConfig { grid_dim: (82 * 4, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    for name in ["mma_peak", "mma_peak_f16acc"] {
        let f = synaptix_kernels_cuda::kernels::compile::load_fn(&module, name).expect("fn");
        for _ in 0..2 {
            let mut b = stream.launch_builder(&f);
            b.arg(&out).arg(&iters);
            unsafe { b.launch(cfg).unwrap() };
        }
        stream.synchronize().unwrap();
        let t0 = Instant::now();
        let reps = 5;
        for _ in 0..reps {
            let mut b = stream.launch_builder(&f);
            b.arg(&out).arg(&iters);
            unsafe { b.launch(cfg).unwrap() };
        }
        stream.synchronize().unwrap();
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        // FLOP: grid_warps × iters × 4 mma × (16·8·16·2)
        let warps = (82 * 4 * 256 / 32) as f64;
        let flop = warps * iters as f64 * 4.0 * (16.0 * 8.0 * 16.0 * 2.0);
        println!("{name}: {:.2} ms  {:.1} TFLOP/s", dt * 1e3, flop / dt / 1e12);
    }
}
