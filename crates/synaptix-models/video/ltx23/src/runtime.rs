//! Программные ручки рантайма LTX-2.3 (вместо env-переменных): профайл-флаги
//! перф-сессий и диагностические капы. Дефолты = прежнее поведение без env.
//! CLI/инструменты выставляют их через сеттеры; чтение — через аксессоры.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static PROF: AtomicBool = AtomicBool::new(false);
static VAE_PROF: AtomicBool = AtomicBool::new(false);
static ATTN_PROF: AtomicBool = AtomicBool::new(false);
static BLK_PROF: AtomicBool = AtomicBool::new(false);
static VOC_PROF: AtomicBool = AtomicBool::new(false);

/// Кап числа DiT-блоков при загрузке (изоляция/тесты): `usize::MAX` = без капа.
static DIT_NBLOCKS_CAP: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Ручной spatial-грид VAE-декода `(nh, nw)`: упакован как `(nh<<32)|nw`, 0 = авто.
static VAE_GRID: AtomicUsize = AtomicUsize::new(0);

/// Пер-фазная разбивка denoise-петли и DiT-блоков (`[LTX_PROF]`).
pub fn set_ltx_prof(on: bool) {
    PROF.store(on, Ordering::Relaxed);
}
pub fn ltx_prof() -> bool {
    PROF.load(Ordering::Relaxed)
}

/// Пер-фазная разбивка VAE-декода (`[VAE_PROF]`/`[VAE_TILE]`/`[VAE_BUDGET]`).
pub fn set_ltx_vae_prof(on: bool) {
    VAE_PROF.store(on, Ordering::Relaxed);
}
pub fn ltx_vae_prof() -> bool {
    VAE_PROF.load(Ordering::Relaxed)
}

/// Под-фазная разбивка attention с sync-точками (`[attn-prof]`).
pub fn set_ltx_attn_prof(on: bool) {
    ATTN_PROF.store(on, Ordering::Relaxed);
}
pub fn ltx_attn_prof() -> bool {
    ATTN_PROF.load(Ordering::Relaxed)
}

/// Суб-фазы DiT-блока с sync-метками (`[BLK_PROF]`).
pub fn set_ltx_blk_prof(on: bool) {
    BLK_PROF.store(on, Ordering::Relaxed);
}
pub fn ltx_blk_prof() -> bool {
    BLK_PROF.load(Ordering::Relaxed)
}

/// Пер-уровневая разбивка вокодера (`[VOC]`).
pub fn set_ltx_voc_prof(on: bool) {
    VOC_PROF.store(on, Ordering::Relaxed);
}
pub fn ltx_voc_prof() -> bool {
    VOC_PROF.load(Ordering::Relaxed)
}

/// Кап числа DiT-блоков при загрузке `VideoDit`/`AvDit` (`None` = все блоки).
pub fn set_dit_nblocks_cap(cap: Option<usize>) {
    DIT_NBLOCKS_CAP.store(cap.unwrap_or(usize::MAX), Ordering::Relaxed);
}
pub fn dit_nblocks_cap() -> Option<usize> {
    match DIT_NBLOCKS_CAP.load(Ordering::Relaxed) {
        usize::MAX => None,
        n => Some(n),
    }
}

/// Режим стриминга DiT-блоков при dense-offload: 0 = легаси-карусель
/// (пер-блок аллокации weights-пула), 1 = слоты (2 фикс-буфера ping-pong),
/// 2 = слоты + CUDA-graph replay на блок.
static BLOCK_MODE: AtomicUsize = AtomicUsize::new(2);

pub fn set_ltx_block_mode(mode: usize) {
    BLOCK_MODE.store(mode, Ordering::Relaxed);
}
pub fn ltx_block_mode() -> usize {
    BLOCK_MODE.load(Ordering::Relaxed)
}

/// Ручной spatial-грид VAE-декода `(nh, nw)` (`None` = авто по бюджету).
pub fn set_vae_grid(grid: Option<(usize, usize)>) {
    let packed = match grid {
        Some((nh, nw)) => ((nh.max(1) as u64) << 32 | (nw.max(1) as u64 & 0xffff_ffff)) as usize,
        None => 0,
    };
    VAE_GRID.store(packed, Ordering::Relaxed);
}
pub fn vae_grid() -> Option<(usize, usize)> {
    match VAE_GRID.load(Ordering::Relaxed) {
        0 => None,
        p => Some(((p >> 32) & 0xffff_ffff, p & 0xffff_ffff)),
    }
}
