# Vendored cudarc — why

Upstream [cudarc](https://github.com/coreylowman/cudarc) `0.19.8` (MIT OR Apache-2.0),
with a local patch that makes kernel launches safe under CUDA graph capture.

## Why the fork exists

`synaptix-infer` captures a decode step into a CUDA graph and replays it per token.
Upstream cudarc is not capture-aware: while a stream is capturing, it still issues

- `cuEventRecord` / `cuStreamWaitEvent` for its automatic cross-stream tracking
  (`LaunchArgs::arg`, `DevicePtr::device_ptr`), and
- `cuStreamSynchronize` from `SyncOnDrop::Sync`.

All three are illegal on a capturing stream: the driver returns
`CUDA_ERROR_INVALID_VALUE`, or `CUDA_ERROR_STREAM_CAPTURE_ISOLATION` when an event
recorded outside the capture is waited on inside it.

`CudaContext::disable_event_tracking()` does not solve this: events are attached to a
`CudaSlice` when it is **allocated**, and model weights are loaded long before capture
begins. Their `read`/`write` events survive and still get recorded on drop.

Dropping event tracking globally is not an option either — synaptix is genuinely
multi-stream: weights are uploaded on a dedicated loader stream so the copy engine
overlaps with compute (`synaptix-core/src/device/cuda.rs`). Cross-stream ordering must
keep working.

## The patch

Two files, `src/driver/safe/{launch.rs,core.rs}`:

1. **Same-stream elision** — when a slice's home stream is the launch stream, skip the
   wait/record entirely: stream ordering already guarantees it. Cross-stream links are
   untouched, so loader-stream → compute-stream synchronization still holds.
2. **Capture-aware events** — when `capture_status()` is ACTIVE, skip event records and
   waits, and skip `stream.synchronize()`. Ordering inside one stream is sequential by
   definition, so nothing is lost. Note that even `cuEventRecordWithFlags(EXTERNAL)` is
   not enough here: it leaves EVENT_RECORD nodes in the captured graph (~2 per kernel),
   which fail with `CUDA_ERROR_ILLEGAL_ADDRESS` on replay.

## Updating cudarc

Re-vendor the new release, then re-apply the diff for those two files. `launch.rs` was
identical across 0.19.6–0.19.8, so it usually copies over as-is; `core.rs` may need
manual fitting (0.19.8 added `view_ptr`, which the patch preserves).

Guard test: `cargo test -p synaptix --features cuda --profile fast-release graph_decode`
with `SYN_GRAPH_DECODE=1` and Qwen3-1.7B present.

3. **Suballocation hook** (`core.rs`) — `set_free_hook` lets an external
   suballocator claim a device pointer on `CudaSlice::drop`. synaptix keeps MoE
   experts in slabs of its own (`synaptix-core/src/memory/expert_arena.rs`):
   the pointers it hands out are interior to a block the driver knows nothing
   about, so they must never reach `cuMemFreeAsync`. Without the hook the only
   alternatives were a borrowed-buffer flag threaded through every storage type,
   or leaking every expert allocation.

