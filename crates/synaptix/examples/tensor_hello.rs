use synaptix::prelude::*;

fn main() -> Result<()> {
    synaptix::init()?;

    let a = Tensor::from_vec(
        (1..=6).map(|x| x as f32).collect::<Vec<f32>>(),
        (2, 3),
        Device::Cpu,
    )?;
    let b = Tensor::from_vec(
        (1..=12).map(|x| x as f32).collect::<Vec<f32>>(),
        (3, 4),
        Device::Cpu,
    )?;
    let c_cpu = a.matmul(&b)?;
    println!("CPU matmul result:");
    println!("{c_cpu}");

    #[cfg(feature = "cuda")]
    {
        if synaptix::device::cuda::get(0).is_ok() {
            let a_g = a.to_device(Device::Cuda(0))?;
            let b_g = b.to_device(Device::Cuda(0))?;
            let c_g = a_g.matmul(&b_g)?;
            let c_back = c_g.to_device(Device::Cpu)?;
            println!("CUDA matmul result (back on CPU):");
            println!("{c_back}");

            let cpu_v = c_cpu.to_vec2::<f32>()?;
            let cuda_v = c_back.to_vec2::<f32>()?;
            for i in 0..2 {
                for j in 0..4 {
                    assert!(
                        (cpu_v[i][j] - cuda_v[i][j]).abs() < 1e-3,
                        "mismatch at ({i},{j})"
                    );
                }
            }
            println!("OK: CPU and CUDA matmul agree within 1e-3.");
        } else {
            println!("(no CUDA device available; skipping GPU path)");
        }
    }

    Ok(())
}
