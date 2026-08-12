use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static PROF: AtomicBool = AtomicBool::new(false);
static VAE_PROF: AtomicBool = AtomicBool::new(false);
static ATTN_PROF: AtomicBool = AtomicBool::new(false);
static BLK_PROF: AtomicBool = AtomicBool::new(false);
static ADALN_PROF: AtomicBool = AtomicBool::new(false);
static MLP_PROF: AtomicBool = AtomicBool::new(false);
static PROF_BLOCK: AtomicUsize = AtomicUsize::new(0);

static MEMORY_MODE: AtomicUsize = AtomicUsize::new(0);
static NBLOCKS_CAP: AtomicUsize = AtomicUsize::new(usize::MAX);
static VAE_GRID: AtomicUsize = AtomicUsize::new(0);
static VAE_TILE: AtomicUsize = AtomicUsize::new(0);

pub fn set_h3_prof(on: bool) {
    PROF.store(on, Ordering::Relaxed);
}
pub fn h3_prof() -> bool {
    PROF.load(Ordering::Relaxed)
}

pub fn set_h3_vae_prof(on: bool) {
    VAE_PROF.store(on, Ordering::Relaxed);
}
pub fn h3_vae_prof() -> bool {
    VAE_PROF.load(Ordering::Relaxed)
}

pub fn set_h3_attn_prof(on: bool) {
    ATTN_PROF.store(on, Ordering::Relaxed);
}
pub fn h3_attn_prof() -> bool {
    ATTN_PROF.load(Ordering::Relaxed)
}

pub fn set_h3_blk_prof(on: bool) {
    BLK_PROF.store(on, Ordering::Relaxed);
}
pub fn h3_blk_prof() -> bool {
    BLK_PROF.load(Ordering::Relaxed)
}

pub fn set_h3_adaln_prof(on: bool) {
    ADALN_PROF.store(on, Ordering::Relaxed);
}
pub fn h3_adaln_prof() -> bool {
    ADALN_PROF.load(Ordering::Relaxed)
}

pub fn set_prof_block(idx: usize) {
    PROF_BLOCK.store(idx, Ordering::Relaxed);
}
pub fn prof_block() -> usize {
    PROF_BLOCK.load(Ordering::Relaxed)
}

pub fn set_h3_mlp_prof(on: bool) {
    MLP_PROF.store(on, Ordering::Relaxed);
}
pub fn h3_mlp_prof() -> bool {
    MLP_PROF.load(Ordering::Relaxed)
}

pub fn set_memory_mode(mode: usize) {
    MEMORY_MODE.store(mode, Ordering::Relaxed);
}
pub fn memory_mode() -> usize {
    MEMORY_MODE.load(Ordering::Relaxed)
}

pub fn set_nblocks_cap(cap: Option<usize>) {
    NBLOCKS_CAP.store(cap.unwrap_or(usize::MAX), Ordering::Relaxed);
}
pub fn nblocks_cap() -> Option<usize> {
    match NBLOCKS_CAP.load(Ordering::Relaxed) {
        usize::MAX => None,
        n => Some(n),
    }
}

pub fn set_vae_grid(grid: Option<(usize, usize)>) {
    let packed = match grid {
        Some((nh, nw)) => (((nh.max(1) as u64) << 32) | (nw.max(1) as u64 & 0xffff_ffff)) as usize,
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

pub fn set_vae_tile(tile: Option<usize>) {
    VAE_TILE.store(tile.unwrap_or(0), Ordering::Relaxed);
}
pub fn vae_tile() -> Option<usize> {
    match VAE_TILE.load(Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}
