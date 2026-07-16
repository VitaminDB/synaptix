//! InfiniBand / RoCE transport (libibverbs + GPUDirect RDMA).
//!
//! **Не реализовано — требует сетевого фабрика и libibverbs.** На
//! single-host single-GPU dev box тестирование невозможно. Точка
//! расширения: feature-флаг `rdma` + ibv_post_send/recv + регистрация
//! GPU-памяти через `ibv_reg_mr` (GPUDirect RDMA) для cross-host
//! pipeline-parallel.
