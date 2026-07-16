//! Multi-GPU/multi-host transport (NCCL / NVLink P2P / InfiniBand RDMA).
//!
//! **Не реализовано и не тестируется на одной GPU.** Все три транспорта
//! требуют либо нескольких устройств в одной коробке (P2P, NCCL), либо
//! сетевого фабрика (RDMA). На текущей dev-конфигурации (single sm_120)
//! bit-exact-проверка невозможна; модули сохранены как явные точки
//! расширения. См. `synaptix-distributed` для локальной математики
//! collectives (TP/CP/SP/EP/ZeRO) — она самодостаточна и проверена в
//! `synaptix-distributed/tests`.

pub mod nccl;
pub mod p2p;
pub mod rdma;
