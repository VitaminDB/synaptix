use half::bf16;
use cudarc::driver::CudaSlice;
use synaptix_kernels_cuda::best_cu::gemm::gemm_bf16::{best_gemm_bf16_cfg, Bf16Config, BestGemmBf16Kernels};
const SHAPES: &[(usize,usize,usize)] = &[
    (2048,1280,10240),(2048,5120,1280),(2048,1280,1280),(8192,640,5120),(8192,640,640),
];
fn run(cfg: Bf16Config, tag: &str) {
    let Ok(ctx) = synaptix_core::device::cuda::get(0) else { return };
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();
    let kernels = BestGemmBf16Kernels::for_context(&ctx).unwrap();
    for &(m,k,n) in SHAPES {
        let a: CudaSlice<bf16> = stream.alloc_zeros(m*k).unwrap();
        let b: CudaSlice<bf16> = stream.alloc_zeros(n*k).unwrap();
        let mut c: CudaSlice<bf16> = stream.alloc_zeros(m*n).unwrap();
        for _ in 0..30 { best_gemm_bf16_cfg(&kernels,&stream,&a,&b,&mut c,m as u32,n as u32,k as u32,cfg).unwrap(); }
        stream.synchronize().unwrap();
        let t=std::time::Instant::now(); let it=200;
        for _ in 0..it { best_gemm_bf16_cfg(&kernels,&stream,&a,&b,&mut c,m as u32,n as u32,k as u32,cfg).unwrap(); }
        stream.synchronize().unwrap();
        let dt=t.elapsed().as_secs_f64()/it as f64;
        eprintln!("[{tag} {m}x{k}x{n}] {:.3}ms {:.0} TF", dt*1e3, 2.0*m as f64*k as f64*n as f64/dt/1e12);
    }
}
#[test] #[ignore] fn bench_s3(){ run(Bf16Config::S3,"S3"); }
#[test] #[ignore] fn bench_s4(){ run(Bf16Config::S4,"S4"); }
#[test] #[ignore] fn bench_s5(){ run(Bf16Config::S5,"S5"); }
#[test] #[ignore] fn bench_s6(){ run(Bf16Config::S6,"S6"); }
