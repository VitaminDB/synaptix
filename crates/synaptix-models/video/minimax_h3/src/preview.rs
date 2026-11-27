//! Превью по ходу денойза MiniMax-H3: блочная линейная проекция среднего
//! латентного кадра (24 канала) в RGB; коэффициенты подобраны примером
//! `latent_rgb_fit` в synaptix-gen. Работает на обеих стадиях денойза
//! (черновая 960×544 и полное разрешение).

use std::sync::OnceLock;

use synaptix_core::gen_status;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

const C: usize = 24;

/// Коэффициенты: `SYN_H3_RGB_FACTORS` (файл f32 LE) или встроенные.
fn factors() -> Option<&'static gen_status::PreviewFactors> {
    static F: OnceLock<Option<gen_status::PreviewFactors>> = OnceLock::new();
    F.get_or_init(|| match std::env::var_os("SYN_H3_RGB_FACTORS") {
        Some(p) => gen_status::parse_block_factors(&std::fs::read(p).ok()?, C),
        None => gen_status::parse_block_factors(include_bytes!("../assets/latent_rgb.f32"), C),
    })
    .as_ref()
}

/// Превью среднего кадра по оценке `denoised_v` формы `[1, 24, F, H, W]`.
pub fn emit(denoised_v: &Tensor, step: usize, total: usize) {
    let (Some(f), n) = (factors(), gen_status::preview_every(1)) else { return };
    if n == 0 || step % n != 0 {
        return;
    }
    let dims = denoised_v.dims();
    if dims.len() != 5 || dims[1] != C || dims[2] == 0 {
        return;
    }
    let (ft, h, w) = (dims[2], dims[3], dims[4]);
    let mid = ft / 2;
    let _ = (|| -> Result<(), synaptix_core::error::SynaptixError> {
        let frame = denoised_v
            .narrow(2, mid, 1)?
            .contiguous()?
            .reshape((C, h * w))?
            .transpose(0, 1)?
            .contiguous()?
            .to_dtype(DType::F32)?
            .to_device(Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        gen_status::linear_preview_blocks(&frame, C, w, h, f, step, total);
        Ok(())
    })();
}

/// Отладка: сохранить видеолатент `[1, 24, F, H, W]` сырым f32 `[24, F, H, W]`
/// при заданной `H3_DUMP_LATENT` (для подбора коэффициентов превью).
pub fn dump_if_requested(video_latent: &Tensor) {
    let Some(p) = std::env::var_os("H3_DUMP_LATENT") else { return };
    let dims = video_latent.dims();
    if dims.len() != 5 || dims[1] != C {
        return;
    }
    let (f, h, w) = (dims[2], dims[3], dims[4]);
    let Ok(flat) = video_latent
        .narrow(0, 0, 1)
        .and_then(|t| t.contiguous())
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_device(Device::Cpu))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
    else {
        return;
    };
    let path = std::path::PathBuf::from(p).with_extension(format!("{f}x{h}x{w}.f32"));
    let _ = std::fs::write(path, flat.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_factors_and_emit_shape() {
        synaptix_kernels_cpu::ensure_registered();
        let f = gen_status::parse_block_factors(include_bytes!("../assets/latent_rgb.f32"), C).expect("valid H3 factors");
        assert_eq!((f.kernel, f.scale), (3, 4));
        // Несимметричный латент [1, 24, 3, 4, 6] с ярким импульсом в среднем кадре (t=1)
        // в точке (y=0, x=5) — проверяем отсутствие транспонирования H/W и верный выбор кадра.
        let (ft, h, w) = (3, 4, 6);
        let mut data = vec![0.0f32; C * ft * h * w];
        for c in 0..C {
            data[((c * ft + 1) * h + 0) * w + 5] = 3.0;
        }
        let lat = Tensor::from_vec(data, vec![1, C, ft, h, w], Device::Cpu).unwrap();
        let _ = gen_status::take_preview();
        emit(&lat, 0, 8);
        let p = gen_status::take_preview().expect("preview emitted");
        assert_eq!((p.width, p.height, p.step, p.total), (24, 16, 0, 8));
        assert_eq!(p.rgb.len(), 24 * 16 * 3);
        // Пиксель в правом верхнем углу (y=1, x=22) должен отличаться от спокойного левого нижнего (y=14, x=1).
        let top_right = &p.rgb[(1 * 24 + 22) * 3..(1 * 24 + 22) * 3 + 3];
        let bot_left = &p.rgb[(14 * 24 + 1) * 3..(14 * 24 + 1) * 3 + 3];
        assert_ne!(top_right, bot_left, "импульс в (y=0,x=5) не попал в правый верхний угол");
    }
}
