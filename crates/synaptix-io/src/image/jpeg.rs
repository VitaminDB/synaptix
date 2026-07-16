use std::path::Path;
use synaptix_core::{device::Device, tensor::Tensor};
use crate::error::Result;
use super::png::{load_image, load_image_rgba, save_image};

pub fn load_jpeg(path: impl AsRef<Path>, device: Device) -> Result<Tensor> {
    load_image(path, device)
}

pub fn save_jpeg(tensor: &Tensor, path: impl AsRef<Path>) -> Result<()> {
    save_image(tensor, path)
}
