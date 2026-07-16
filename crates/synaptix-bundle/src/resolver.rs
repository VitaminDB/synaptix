//! Resolving `RefSpec`s to concrete `Bundle`s on disk.
//!
//! A bundle's `BundleMeta.refs` lists external `.syn` files this bundle depends
//! on for shared backbones (e.g. VoxCPM2 and Qwen3-TTS both pointing at a
//! single Qwen0.6 base). When the parent bundle is opened, refs are not
//! auto-resolved — callers explicitly invoke `Bundle::resolve_ref(id, &resolver)`.
//!
//! The default `FsResolver` walks filesystem paths (from the RefSpec, env
//! `SYN_PATH`, and a default `~/.syn/cache`) and matches candidates by
//! `sha256` (priority) or `id`. Future resolvers may pull from HF / OCI.

use std::path::PathBuf;

use crate::bundle::Bundle;
use crate::cdir::RefSpec;
use crate::error::{Error, Result};

pub trait RefResolver {
    /// Locate the `.syn` bundle that satisfies `spec`. Implementations should
    /// surface a clear error when no match is found.
    fn resolve(&self, spec: &RefSpec) -> Result<Bundle>;
}

/// Filesystem-based resolver. Order of lookup:
///
/// 1. `spec.search_paths` (relative dirs from the bundle producer),
/// 2. `SYN_PATH` env var (`:`-separated list of dirs),
/// 3. `extra` paths supplied at construction time,
/// 4. `~/.syn/cache`.
///
/// Within each path we enumerate `*.syn`, read each candidate's header+cdir
/// (without touching tensor payload), and pick by `sha256` if the RefSpec
/// supplies one, else by `id`.
pub struct FsResolver {
    extra: Vec<PathBuf>,
}

impl FsResolver {
    pub fn new() -> Self {
        Self { extra: Vec::new() }
    }

    pub fn with_path(mut self, p: impl Into<PathBuf>) -> Self {
        self.extra.push(p.into());
        self
    }

    fn candidate_paths(&self, spec: &RefSpec) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = Vec::new();
        let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));

        // 1. RefSpec search_paths (with ~ expansion).
        for p in &spec.search_paths {
            paths.push(expand_tilde(p, home.as_deref().ok()));
        }
        // 2. User-supplied extras.
        for p in &self.extra {
            paths.push(p.clone());
        }
        // 3. Default cache.
        if let Ok(h) = &home {
            paths.push(PathBuf::from(h).join(".syn").join("cache"));
        }
        paths
    }
}

impl Default for FsResolver {
    fn default() -> Self {
        Self::new()
    }
}

fn expand_tilde(p: &str, home: Option<&str>) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(h) = home {
            return PathBuf::from(h).join(rest);
        }
    }
    PathBuf::from(p)
}

impl RefResolver for FsResolver {
    fn resolve(&self, spec: &RefSpec) -> Result<Bundle> {
        for dir in self.candidate_paths(spec) {
            if !dir.is_dir() {
                continue;
            }
            let Ok(rd) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in rd.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if path.extension().and_then(|s| s.to_str()) != Some("syn") {
                    continue;
                }
                // Cheap open: lets us see id + manifest_sha256 without touching tensor data.
                let Ok(b) = Bundle::open(&path) else {
                    continue;
                };
                if matches(&b, spec) {
                    return Ok(b);
                }
            }
        }
        Err(Error::FileNotFound(format!(
            ".syn ref `{}` not found in any search path",
            spec.id
        )))
    }
}

fn matches(b: &Bundle, spec: &RefSpec) -> bool {
    // Sha256 takes priority — if the RefSpec stored one, only an exact match
    // counts as a hit. Tampered/wrong-version bundle is invisible.
    if !spec.sha256.is_empty() {
        match &b.meta().manifest_sha256 {
            Some(actual) => return actual.as_slice() == spec.sha256.as_slice(),
            None => return false,
        }
    }
    if !spec.id.is_empty() && b.id() == spec.id {
        return true;
    }
    false
}
