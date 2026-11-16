//! Закреплённая RAM, видимая карте как обычный буфер устройства (zero-copy).
//!
//! `cuMemHostAlloc` под единым адресным пространством отдаёт указатель,
//! одинаковый для хоста и для ядер CUDA: ядро читает такие байты прямо из
//! RAM по шине (без копии в VRAM), а адрес постоянен — захваченные CUDA-графы
//! остаются годными. Так живут хвостовые блоки модели, которой не хватило
//! видеопамяти: веса те же и ядра те же, результат бит в бит, цена — скорость
//! шины вместо скорости видеопамяти.
//!
//! Буфер оформляется как `CudaSlice` на адресе RAM. Его освобождение
//! перехватывает общий хук [`crate::memory::free_hook`]: память возвращается
//! `cuMemFreeHost`, а не `cuMemFree`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use parking_lot::Mutex;

use crate::error::{Result, SynaptixError};
use crate::memory::pinned::PinnedBuf;

static LO: AtomicU64 = AtomicU64::new(u64::MAX);
static HI: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: Mutex<Option<HashMap<u64, (PinnedBuf, usize)>>> = Mutex::new(None);

thread_local! {
    static ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Пока жив — загрузки CPU→CUDA этого потока кладут байты в закреплённую
/// RAM, а не в видеопамять.
pub struct HostMappedGuard(bool);

impl HostMappedGuard {
    pub fn new() -> Self {
        Self(ON.with(|c| c.replace(true)))
    }
}

impl Default for HostMappedGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for HostMappedGuard {
    fn drop(&mut self) {
        ON.with(|c| c.set(self.0));
    }
}

thread_local! {
    static VRAM_COPY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Пока жив — «перенос на то же устройство» буфера из закреплённой RAM даёт
/// настоящую копию в видеопамяти (стрим блока под длинный кусок префилла).
pub struct VramCopyGuard(bool);

impl VramCopyGuard {
    pub fn new() -> Self {
        Self(VRAM_COPY.with(|c| c.replace(true)))
    }
}

impl Default for VramCopyGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for VramCopyGuard {
    fn drop(&mut self) {
        VRAM_COPY.with(|c| c.set(self.0));
    }
}

pub fn vram_copy_enabled() -> bool {
    VRAM_COPY.with(|c| c.get())
}

/// Копия буфера из закреплённой RAM в видеопамять (DMA прямо из закреплённых
/// страниц, асинхронно на `stream`). `None` — адрес не наш.
pub fn to_vram(stream: &Arc<CudaStream>, ptr: u64, len: usize) -> Option<Result<CudaSlice<u8>>> {
    if ptr < LO.load(Ordering::Acquire) || ptr >= HI.load(Ordering::Acquire) {
        return None;
    }
    let g = LIVE.lock();
    let (buf, _) = g.as_ref()?.get(&ptr)?;
    let src = &buf.as_slice()[..len.min(buf.len())];
    let copy = || -> Result<CudaSlice<u8>> {
        let mut dst = unsafe { crate::device::cuda::alloc_bytes_uninit(stream, src.len()) }
            .map_err(|e| SynaptixError::Cuda(format!("host_mapped → VRAM alloc({}): {e:?}", src.len())))?;
        stream
            .memcpy_htod(src, &mut dst)
            .map_err(|e| SynaptixError::Cuda(format!("host_mapped → VRAM memcpy: {e:?}")))?;
        Ok(dst)
    };
    Some(copy())
}

pub fn enabled() -> bool {
    ON.with(|c| c.get())
}

/// Сколько байт сейчас лежит в такой памяти.
pub fn live_bytes() -> usize {
    BYTES.load(Ordering::Relaxed)
}

/// Адрес принадлежит закреплённой RAM (а не видеопамяти).
pub fn contains(ptr: u64) -> bool {
    if ptr < LO.load(Ordering::Acquire) || ptr >= HI.load(Ordering::Acquire) {
        return false;
    }
    LIVE.lock().as_ref().is_some_and(|m| m.contains_key(&ptr))
}

/// Байты буфера, если он живёт в закреплённой RAM: выгрузка на хост — обычная
/// копия памяти, без обращения к драйверу.
pub fn host_bytes(ptr: u64, len: usize) -> Option<Vec<u8>> {
    if ptr < LO.load(Ordering::Acquire) || ptr >= HI.load(Ordering::Acquire) {
        return None;
    }
    let g = LIVE.lock();
    let (buf, _) = g.as_ref()?.get(&ptr)?;
    Some(buf.as_slice()[..len.min(buf.len())].to_vec())
}

/// Положить `bytes` в закреплённую RAM и вернуть буфер «на устройстве».
pub fn htod(stream: &Arc<CudaStream>, bytes: &[u8]) -> Result<CudaSlice<u8>> {
    crate::memory::install_free_hook();
    let len = bytes.len().max(1);
    let mut buf = PinnedBuf::new_uninit(len);
    if !buf.is_pinned() {
        return Err(SynaptixError::Cuda(format!("host_mapped: не закрепилось {len} байт RAM")));
    }
    {
        use rayon::prelude::*;
        const PIECE: usize = 4 << 20;
        buf.as_mut_slice()[..bytes.len()]
            .par_chunks_mut(PIECE)
            .zip(bytes.par_chunks(PIECE))
            .for_each(|(d, s)| d.copy_from_slice(s));
    }
    let ptr = buf.as_ptr() as u64;
    LO.fetch_min(ptr, Ordering::AcqRel);
    HI.fetch_max(ptr + len as u64, Ordering::AcqRel);
    BYTES.fetch_add(len, Ordering::Relaxed);
    let ord = stream.context().ordinal();
    LIVE.lock().get_or_insert_with(HashMap::new).insert(ptr, (buf, ord));
    Ok(unsafe { stream.upgrade_device_ptr::<u8>(ptr, bytes.len()) })
}

/// Хук освобождения: наш адрес — вернуть RAM. Ядра, ещё читающие буфер,
/// обязаны закончить: освобождение обычной видеопамяти встаёт в хвост очереди
/// стрима, а закреплённая RAM отдаётся сразу.
pub fn claim_free(ptr: u64) -> bool {
    if ptr < LO.load(Ordering::Acquire) || ptr >= HI.load(Ordering::Acquire) {
        return false;
    }
    let taken = LIVE.lock().as_mut().and_then(|m| m.remove(&ptr));
    let Some((buf, ord)) = taken else { return false };
    let _ = crate::device::cuda::synchronize_all(ord);
    BYTES.fetch_sub(buf.len().max(1), Ordering::Relaxed);
    drop(buf);
    true
}
