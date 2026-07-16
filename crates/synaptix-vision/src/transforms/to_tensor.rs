use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::error::{Result, VisionError};
use crate::image_buf::{ChannelOrder, RgbImage};

pub fn rgb_to_tensor_chw(img: &RgbImage, device: Device) -> Result<Tensor> {
    let chw = img.to_chw();
    let dims = (chw.channels, chw.height, chw.width);
    Tensor::from_vec(chw.data.clone(), dims, device).map_err(VisionError::from)
}

pub fn tensor_chw_to_rgb(t: &Tensor) -> Result<RgbImage> {
    let dims = t.dims();
    if dims.len() != 3 {
        return Err(VisionError::invalid_arg(format!(
            "tensor_chw_to_rgb: expected rank 3 (C,H,W), got {:?}",
            dims
        )));
    }
    if t.dtype() != DType::F32 {
        return Err(VisionError::invalid_arg(format!(
            "tensor_chw_to_rgb: expected F32, got {:?}",
            t.dtype()
        )));
    }
    let channels = dims[0];
    let height = dims[1];
    let width = dims[2];
    let flat = t
        .contiguous()
        .map_err(VisionError::from)?
        .reshape((channels * height * width,))
        .map_err(VisionError::from)?;
    let data = flat.to_vec1::<f32>().map_err(VisionError::from)?;
    Ok(RgbImage { data, width, height, channels, order: ChannelOrder::Chw })
}
