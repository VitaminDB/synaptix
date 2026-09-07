use std::sync::{Arc, OnceLock};

use cudarc::driver::{sys, CudaSlice, CudaStream, DevicePtr, LaunchArgs, PushKernelArg};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

/// Размер `CUtensorMap` в байтах.
pub const TMA_DESC_BYTES: usize = 128;
/// Слотов в арене одного устройства: 32768 × 128 Б = 4 МБ.
const ARENA_SLOTS: u32 = 32768;

/// TMA-дескриптор в арене устройства. Ядру уходит адрес слота
/// (`const CUtensorMap*`), поэтому `.arg(&*desc)` работает как прежде с
/// `CudaSlice<u8>`.
///
/// Зачем арена, а не отдельная аллокация на дескриптор: кэши GEMM ключуют
/// дескрипторы по АДРЕСУ тензора и живут до `release_device_caches`. При
/// стриминге блоков с хоста веса получают новый адрес на каждом forward'е,
/// и каждый чанк префилла рождал сотни новых 128-байтовых аллокаций в
/// default-пуле. Они ложились в сегменты, только что взятые под перемешанные
/// копии весов, и когда те освобождались, сегмент оставался у пула из-за
/// живого дескриптора: 07.09.2026 на промпте 77k резерв пула рос ровно на
/// сегмент за чанк (32–64 МБ) при плоском `used`, трим не возвращал ничего,
/// и ход падал OOM'ом на 40k. В арене мёртвые записи стоят 128 Б каждая,
/// сегментов весов не держат, а слоты возвращаются в оборот.
pub struct TmaDesc {
    arena: Arc<DescArena>,
    slot: u32,
    dev_ptr: u64,
}

impl TmaDesc {
    /// Адрес дескриптора на устройстве.
    pub fn device_ptr(&self) -> u64 {
        self.dev_ptr
    }

    /// Закодированные байты `CUtensorMap` → слот арены.
    fn upload(stream: &Arc<CudaStream>, bytes: &[u8], what: &str) -> Result<Self> {
        debug_assert_eq!(bytes.len(), TMA_DESC_BYTES);
        let arena = DescArena::for_stream(stream)?;
        let slot = arena.take_slot(stream.context().ordinal())?;
        let off = slot as usize * TMA_DESC_BYTES;
        {
            let mut buf = arena.buf.lock();
            let mut view = buf.slice_mut(off..off + TMA_DESC_BYTES);
            if let Err(e) = stream.memcpy_htod(bytes, &mut view) {
                arena.free.lock().push(slot);
                return Err(SynaptixError::Cuda(format!("htod {what}: {e:?}")));
            }
        }
        Ok(Self { dev_ptr: arena.base + off as u64, slot, arena })
    }
}

impl Drop for TmaDesc {
    fn drop(&mut self) {
        // Слот может ещё читать запущенное ядро — в оборот он вернётся только
        // после синка (см. `DescArena::take_slot`).
        self.arena.pending.lock().push(self.slot);
    }
}

unsafe impl<'a, 'b: 'a> PushKernelArg<&'b TmaDesc> for LaunchArgs<'a> {
    fn arg(&mut self, d: &'b TmaDesc) -> &mut Self {
        self.arg(&d.dev_ptr)
    }
}

/// Арена дескрипторов одного устройства: один буфер, слоты по 128 Б.
struct DescArena {
    buf: Mutex<CudaSlice<u8>>,
    base: u64,
    free: Mutex<Vec<u32>>,
    /// Слоты сброшенных дескрипторов — переиспользуются после синка.
    pending: Mutex<Vec<u32>>,
}

static ARENAS: OnceLock<Mutex<Vec<(usize, Arc<DescArena>)>>> = OnceLock::new();

impl DescArena {
    fn for_stream(stream: &Arc<CudaStream>) -> Result<Arc<Self>> {
        let ord = stream.context().ordinal();
        let list = ARENAS.get_or_init(|| Mutex::new(Vec::new()));
        let mut g = list.lock();
        if let Some((_, a)) = g.iter().find(|(o, _)| *o == ord) {
            return Ok(a.clone());
        }
        // Один сегмент default-пула на всё время жизни процесса — по замыслу:
        // именно он и держит все дескрипторы, а не сегменты весов.
        let buf = stream
            .alloc_zeros::<u8>(ARENA_SLOTS as usize * TMA_DESC_BYTES)
            .map_err(|e| SynaptixError::Cuda(format!("TMA desc arena alloc: {e:?}")))?;
        let (base, _) = buf.device_ptr(stream);
        let arena = Arc::new(Self {
            buf: Mutex::new(buf),
            base,
            free: Mutex::new((0..ARENA_SLOTS).rev().collect()),
            pending: Mutex::new(Vec::new()),
        });
        g.push((ord, arena.clone()));
        Ok(arena)
    }

    fn reclaim_pending(&self, ord: usize) {
        let _ = synaptix_core::device::cuda::synchronize_all(ord);
        let mut p = self.pending.lock();
        self.free.lock().extend(p.drain(..));
    }

    fn take_slot(&self, ord: usize) -> Result<u32> {
        if let Some(s) = self.free.lock().pop() {
            return Ok(s);
        }
        self.reclaim_pending(ord);
        if let Some(s) = self.free.lock().pop() {
            return Ok(s);
        }
        // Всё занято живыми записями кэшей — сбросить их: дескрипторы
        // восстанавливаются лениво на первом же вызове.
        crate::best_cu::gemm::gemm_nvfp4::clear_desc_cache();
        self.reclaim_pending(ord);
        self.free
            .lock()
            .pop()
            .ok_or_else(|| SynaptixError::Cuda("TMA desc arena exhausted".into()))
    }
}

pub fn make_tma_desc_2d_u8(
    stream: &Arc<CudaStream>,
    dev_ptr: sys::CUdeviceptr,
    rows: u32,
    cols_bytes: u32,
    box_rows: u32,
    box_cols_bytes: u32,
) -> Result<TmaDesc> {
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
) -> Result<TmaDesc> {
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
) -> Result<TmaDesc> {
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
    TmaDesc::upload(stream, &bytes, "TMA desc")
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
) -> Result<TmaDesc> {
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
    TmaDesc::upload(stream, &bytes, "TMA desc 3d")
}
