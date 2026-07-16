use std::sync::Arc;

use cudarc::driver::{sys, CudaSlice, CudaStream};
use synaptix_core::error::{Result, SynaptixError};

pub fn make_tma_desc_2d_u8(
    stream: &Arc<CudaStream>,
    dev_ptr: sys::CUdeviceptr,
    rows: u32,
    cols_bytes: u32,
    box_rows: u32,
    box_cols_bytes: u32,
) -> Result<CudaSlice<u8>> {
    make_tma_desc_2d_u8_swz(
        stream,
        dev_ptr,
        rows,
        cols_bytes,
        box_rows,
        box_cols_bytes,
        sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    )
}

pub fn make_tma_desc_2d_u8_swz(
    stream: &Arc<CudaStream>,
    dev_ptr: sys::CUdeviceptr,
    rows: u32,
    cols_bytes: u32,
    box_rows: u32,
    box_cols_bytes: u32,
    swizzle: sys::CUtensorMapSwizzle,
) -> Result<CudaSlice<u8>> {
    make_tma_desc_2d_u8_swz_l2(
        stream,
        dev_ptr,
        rows,
        cols_bytes,
        box_rows,
        box_cols_bytes,
        swizzle,
        sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn make_tma_desc_2d_u8_swz_l2(
    stream: &Arc<CudaStream>,
    dev_ptr: sys::CUdeviceptr,
    rows: u32,
    cols_bytes: u32,
    box_rows: u32,
    box_cols_bytes: u32,
    swizzle: sys::CUtensorMapSwizzle,
    l2: sys::CUtensorMapL2promotion,
) -> Result<CudaSlice<u8>> {
    let mut map = sys::CUtensorMap { opaque: [0u64; 16] };

    let global_dim: [u64; 2] = [cols_bytes as u64, rows as u64];
    let global_strides: [u64; 1] = [cols_bytes as u64];
    let box_dim: [u32; 2] = [box_cols_bytes, box_rows];
    let elem_strides: [u32; 2] = [1, 1];
    let res = unsafe {
        sys::cuTensorMapEncodeTiled(
            &mut map as *mut _,
            sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            dev_ptr as *mut std::ffi::c_void,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            elem_strides.as_ptr(),
            sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            l2,
            sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if res != sys::CUresult::CUDA_SUCCESS {
        return Err(SynaptixError::Cuda(format!(
            "cuTensorMapEncodeTiled failed: {res:?}"
        )));
    }
    let bytes: Vec<u8> = map.opaque.iter().flat_map(|w| w.to_le_bytes()).collect();
    stream
        .clone_htod(&bytes)
        .map_err(|e| SynaptixError::Cuda(format!("htod TMA desc: {e:?}")))
}

#[allow(clippy::too_many_arguments)]
pub fn make_tma_desc_3d_u8(
    stream: &Arc<CudaStream>,
    dev_ptr: sys::CUdeviceptr,
    dim0_bytes: u32,
    dim1_count: u32,
    dim2_count: u32,
    stride1: u64,
    stride2: u64,
    box0_bytes: u32,
    box1: u32,
    box2: u32,
) -> Result<CudaSlice<u8>> {
    let mut map = sys::CUtensorMap { opaque: [0u64; 16] };
    let global_dim: [u64; 3] = [dim0_bytes as u64, dim1_count as u64, dim2_count as u64];
    let global_strides: [u64; 2] = [stride1, stride2];
    let box_dim: [u32; 3] = [box0_bytes, box1, box2];
    let elem_strides: [u32; 3] = [1, 1, 1];
    let res = unsafe {
        sys::cuTensorMapEncodeTiled(
            &mut map as *mut _,
            sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            3,
            dev_ptr as *mut std::ffi::c_void,
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
    if res != sys::CUresult::CUDA_SUCCESS {
        return Err(SynaptixError::Cuda(format!(
            "cuTensorMapEncodeTiled 3d failed: {res:?}"
        )));
    }
    let bytes: Vec<u8> = map.opaque.iter().flat_map(|w| w.to_le_bytes()).collect();
    stream
        .clone_htod(&bytes)
        .map_err(|e| SynaptixError::Cuda(format!("htod TMA desc 3d: {e:?}")))
}
