use synaptix_core::{device::Device, dtype::DType};

use crate::adaln::AdalnPlan;
use crate::dit::dit_resident_bytes;
use crate::loader::H3Checkpoint;
use crate::runtime;
use crate::H3Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3MemoryMode {
    Auto,
    AdalnPrecomputed,
    BlockOffload,
}

impl H3MemoryMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "auto" => Some(H3MemoryMode::Auto),
            "precomputed-adaln" | "adaln" | "precomputed" => Some(H3MemoryMode::AdalnPrecomputed),
            "block-offload" | "offload" | "stream" => Some(H3MemoryMode::BlockOffload),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            H3MemoryMode::Auto => "auto",
            H3MemoryMode::AdalnPrecomputed => "precomputed-adaln",
            H3MemoryMode::BlockOffload => "block-offload",
        }
    }

    pub fn to_runtime(self) -> usize {
        match self {
            H3MemoryMode::Auto => 0,
            H3MemoryMode::AdalnPrecomputed => 1,
            H3MemoryMode::BlockOffload => 2,
        }
    }

    pub fn from_runtime(v: usize) -> Self {
        match v {
            1 => H3MemoryMode::AdalnPrecomputed,
            2 => H3MemoryMode::BlockOffload,
            _ => H3MemoryMode::Auto,
        }
    }

    pub fn current() -> Self {
        Self::from_runtime(runtime::memory_mode())
    }

    pub fn install(self) {
        runtime::set_memory_mode(self.to_runtime());
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryPlan {
    pub mode: H3MemoryMode,
    pub resident_bytes: usize,
    pub adaln_cache_bytes: usize,
    pub activation_bytes: usize,
    pub free_bytes: usize,
    pub required_bytes: usize,
}

impl MemoryPlan {
    pub fn fits(&self) -> bool {
        self.required_bytes <= self.free_bytes
    }

    pub fn summary(&self) -> String {
        let gb = |b: usize| b as f64 / (1u64 << 30) as f64;
        format!(
            "режим {}: веса {:.1} ГБ + adaLN-кэш {:.1} ГБ + активации {:.1} ГБ = {:.1} ГБ, свободно {:.1} ГБ",
            self.mode.as_str(),
            gb(self.resident_bytes),
            gb(self.adaln_cache_bytes),
            gb(self.activation_bytes),
            gb(self.required_bytes),
            gb(self.free_bytes)
        )
    }
}

pub fn activation_bytes(seq_len: usize, cfg: &crate::config::H3Config, compute: DType) -> usize {
    let esz = compute.bytes_for_numel(1).max(1);
    let inner = cfg.inner_dim();
    let qkv = seq_len * inner * 3 * esz;
    let attn_out = seq_len * inner * esz;
    let hidden = seq_len * cfg.hidden_size * esz;
    let ffn = seq_len.min(16384) * cfg.ffn_hidden_size * 2 * esz;
    qkv + attn_out + hidden * 4 + ffn
}

pub fn plan(
    ckpt: &H3Checkpoint,
    adaln_plan: &AdalnPlan,
    seq_len: usize,
    quant: DType,
    compute: DType,
    cache_dtype: DType,
    requested: H3MemoryMode,
    device: Device,
) -> Result<MemoryPlan, H3Error> {
    let free = free_vram(device);
    let act = activation_bytes(seq_len, &ckpt.config, compute);
    let cache = adaln_plan.estimated_bytes(
        ckpt.config.num_layers,
        ckpt.config.hidden_size,
        cache_dtype,
    );
    let headroom = 1usize << 30;

    let pre_resident = dit_resident_bytes(ckpt, quant, compute, false);
    let pre_total = pre_resident + cache + act + headroom;

    let mode = match requested {
        H3MemoryMode::Auto => {
            if free == 0 || pre_total <= free {
                H3MemoryMode::AdalnPrecomputed
            } else {
                H3MemoryMode::BlockOffload
            }
        }
        other => other,
    };

    let (resident, cache_b) = match mode {
        H3MemoryMode::BlockOffload => (dit_resident_bytes(ckpt, quant, compute, true) / 8, 0),
        _ => (pre_resident, cache),
    };

    Ok(MemoryPlan {
        mode,
        resident_bytes: resident,
        adaln_cache_bytes: cache_b,
        activation_bytes: act,
        free_bytes: free,
        required_bytes: resident + cache_b + act + headroom,
    })
}

pub fn free_vram(device: Device) -> usize {
    match device {
        Device::Cuda(ord) => synaptix_core::device::cuda::mem_info(ord)
            .map(|(free, _)| free)
            .unwrap_or(0),
        _ => 0,
    }
}

pub fn trim_pool(device: Device) {
    if let Device::Cuda(ord) = device {
        let _ = synaptix_core::device::cuda::synchronize_all(ord);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(ord);
    }
}
