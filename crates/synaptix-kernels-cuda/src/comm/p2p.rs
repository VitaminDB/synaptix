//! NVLink peer-to-peer copy (cudaMemcpyPeer / cudaDeviceEnablePeerAccess).
//!
//! **Не реализовано — требует ≥2 GPU в одной коробке.** На single-GPU
//! self-копия = обычный D2D-memcpy на одном устройстве (нет смысла).
//! Точка расширения: при появлении 2+ GPU обернуть
//! `cuDeviceEnablePeerAccess` + `cuMemcpyPeerAsync` для tensor-parallel
//! shard exchanges.
