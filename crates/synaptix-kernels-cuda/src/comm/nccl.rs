//! NCCL AllReduce / AllGather / ReduceScatter.
//!
//! **Не реализовано — требует libnccl и multi-GPU runtime.** На текущей
//! single-sm_120 dev-конфигурации не пишется и не тестируется. Точка
//! расширения: при появлении multi-GPU добавить feature-флаг `nccl`,
//! линковку с `nccl` + ncclCommInitRank/ncclAllReduce обёртки, а
//! `synaptix-distributed::collectives` перевести на этот transport.
