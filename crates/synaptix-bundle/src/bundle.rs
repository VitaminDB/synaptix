//! Read-side: open a `.syn`, expose mmap-backed slices.

use std::borrow::Cow;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::cdir::{
    CdirFormat, CentralDirectory, ChunkEntry, ChunkType, RefSpec,
};
use crate::resolver::RefResolver;
use crate::chunk::verify_payload_crc;
use crate::error::{Error, Result};
use crate::header::{
    CdirOnDiskFormat, FileHeader, Footer, ALIGNMENT, FILE_HEADER_SIZE, FOOTER_SIZE, TRAILER_MAGIC,
};
use crate::path as syn_path;
use crate::{SUPPORTED_MAJOR, SUPPORTED_REQUIRED_CAPS};

/// One entry from `list_dir_shallow`.
#[derive(Debug, Clone)]
pub enum DirEntry<'a> {
    File(&'a ChunkEntry),
    Subdir(&'a str),
}

pub struct Bundle {
    mmap: Mmap,
    header: FileHeader,
    footer: Footer,
    cdir: CentralDirectory,
}

impl std::fmt::Debug for Bundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bundle")
            .field("id", &self.cdir.bundle_meta.id)
            .field("version", &(self.header.ver_major, self.header.ver_minor))
            .field("entries", &self.cdir.entries.len())
            .field("size", &self.mmap.len())
            .finish()
    }
}

impl Bundle {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if usize::BITS < 64 {
            return Err(Error::PlatformUnsupported("64-bit OS required"));
        }
        let file = File::open(path.as_ref())?;
        let size = file.metadata()?.len();
        if size < (FILE_HEADER_SIZE + FOOTER_SIZE) as u64 {
            return Err(Error::FileTooSmall(size));
        }
        // SAFETY: read-only mmap; we trust the OS not to mutate underneath us
        // (the file is opened read-only and there's no shared writer in this
        // process). Concurrent writers from other processes would be a misuse
        // contract violation; `BundleEditor` (Phase 2) will take a flock.
        let mmap = unsafe { Mmap::map(&file)? };

        // File header at offset 0.
        let header = FileHeader::read_from(&mmap[..FILE_HEADER_SIZE])?;

        if header.ver_major > SUPPORTED_MAJOR {
            return Err(Error::UnsupportedFormatVersion {
                found_major: header.ver_major,
                found_minor: header.ver_minor,
                supported_major: SUPPORTED_MAJOR,
            });
        }

        // Try the trailer at EOF-40 first; fall back to scan-back recovery
        // when the last commit was interrupted mid-write (the most recent
        // bytes are torn). The previous fully-fsynced trailer is still
        // somewhere earlier in the file thanks to the append-only layout.
        let (footer, footer_off) = match locate_trailer(&mmap, mmap.len() - FOOTER_SIZE) {
            Ok(pair) => pair,
            Err(_) => scan_back_for_trailer(&mmap)?,
        };
        if footer.ver_major != header.ver_major {
            return Err(Error::BadMagic { where_: "footer/header version mismatch" });
        }

        // Cdir.
        let start = footer.cdir_offset;
        let end = start.checked_add(footer.cdir_len).ok_or(Error::CdirOutOfBounds(start))?;
        if end > footer_off as u64 {
            return Err(Error::CdirOutOfBounds(end));
        }
        let cdir_bytes = &mmap[start as usize..end as usize];
        let computed = crc32c::crc32c(cdir_bytes);
        if computed != footer.cdir_crc32c {
            return Err(Error::CdirCrcMismatch {
                expected: footer.cdir_crc32c,
                got: computed,
            });
        }
        let fmt = CdirOnDiskFormat::from_u16(footer.cdir_format)
            .ok_or(Error::BadMagic { where_: "unknown cdir format" })?;
        let cdir_fmt = match fmt {
            CdirOnDiskFormat::Cbor => CdirFormat::Cbor,
            CdirOnDiskFormat::Json => CdirFormat::Json,
        };
        let cdir = CentralDirectory::decode(cdir_bytes, cdir_fmt)?;

        // Must-understand caps.
        for cap in &cdir.bundle_meta.required_caps {
            if !SUPPORTED_REQUIRED_CAPS.contains(&cap.as_str()) {
                return Err(Error::UnsupportedCapability(cap.clone()));
            }
        }

        tracing::debug!(
            id = %cdir.bundle_meta.id,
            ver = format!("{}.{}", header.ver_major, header.ver_minor),
            entries = cdir.entries.len(),
            size = mmap.len(),
            "loaded .syn bundle"
        );

        Ok(Self { mmap, header, footer, cdir })
    }

    pub fn id(&self) -> &str { &self.cdir.bundle_meta.id }
    pub fn version(&self) -> (u16, u16) { (self.header.ver_major, self.header.ver_minor) }
    pub fn refs(&self) -> &[RefSpec] { &self.cdir.bundle_meta.refs }
    pub fn meta(&self) -> &crate::cdir::BundleMeta { &self.cdir.bundle_meta }
    pub fn cdir(&self) -> &CentralDirectory { &self.cdir }
    pub fn size(&self) -> u64 { self.mmap.len() as u64 }
    pub fn footer(&self) -> &Footer { &self.footer }
    pub fn file_header(&self) -> &FileHeader { &self.header }

    /// Raw chunk payload as a borrowed slice (no CRC verification, no decompression).
    /// Used by tooling that wants to dump the bundle's bytes verbatim.
    pub fn read_raw_chunk<'a>(&'a self, entry: &ChunkEntry) -> Result<Cow<'a, [u8]>> {
        Ok(Cow::Borrowed(self.chunk_slice(entry)?))
    }

    /// Raw slice of a chunk's payload, *without* CRC verification.
    /// The slice borrows the mmap.
    fn chunk_slice(&self, entry: &ChunkEntry) -> Result<&[u8]> {
        let start = entry.payload_off as usize;
        let end = start.checked_add(entry.payload_len as usize)
            .ok_or(Error::CdirOutOfBounds(entry.payload_off))?;
        if end > self.mmap.len() {
            return Err(Error::CdirOutOfBounds(end as u64));
        }
        Ok(&self.mmap[start..end])
    }

    /// Find the canonical Tensors chunk. Convention: name == "tensors:main".
    /// Falls back to the first alive Tensors chunk.
    fn find_tensors_entry(&self) -> Result<&ChunkEntry> {
        if let Some(e) = self.cdir.find_alive("tensors:main") {
            if matches!(e.kind_typed(), ChunkType::Tensors) {
                return Ok(e);
            }
        }
        self.cdir
            .entries
            .iter()
            .rev()
            .find(|e| e.is_alive() && matches!(e.kind_typed(), ChunkType::Tensors))
            .ok_or(Error::TensorsChunkMissing)
    }

    /// Find a named Tensors chunk: looks up an alive `tensors:<name>` chunk
    /// of type [`ChunkType::Tensors`]. Used by multi-component bundles
    /// (OmniVoice lm+codec, …) to address each component's chunk directly.
    fn find_tensors_entry_named(&self, name: &str) -> Result<&ChunkEntry> {
        let full = format!("tensors:{name}");
        self.cdir
            .entries
            .iter()
            .find(|e| e.is_alive()
                  && matches!(e.kind_typed(), ChunkType::Tensors)
                  && e.name == full)
            .ok_or(Error::TensorsChunkMissing)
    }

    /// Найти первый alive quantized-chunk (`CHUNK_TYPE_QUANTIZED_TENSORS = 6`).
    /// Convention: name == `QUANTIZED_CHUNK_NAME` ("quantized:main"). Fallback —
    /// первый alive `QuantizedTensors` chunk.
    fn find_quantized_entry(&self) -> Option<&ChunkEntry> {
        if let Some(e) = self.cdir.find_alive(crate::quantized::QUANTIZED_CHUNK_NAME) {
            if matches!(e.kind_typed(), ChunkType::QuantizedTensors) {
                return Some(e);
            }
        }
        self.cdir
            .entries
            .iter()
            .rev()
            .find(|e| e.is_alive() && matches!(e.kind_typed(), ChunkType::QuantizedTensors))
    }

    /// Прочитать quantized-chunk: `(header, payload_slice)`. Возвращает `None`,
    /// если в bundle нет quantized chunk'ов — caller должен fallback на
    /// [`Self::tensor_var_builder`].
    pub fn quantized_chunk(
        &self,
    ) -> Result<Option<(crate::quantized::QuantizedChunkHeader, &[u8])>> {
        let Some(entry) = self.find_quantized_entry() else {
            return Ok(None);
        };
        let slice = self.chunk_slice(entry)?;
        if slice.len() < crate::quantized::QuantizedChunkHeader::SIZE {
            return Err(Error::CborDecode(format!(
                "quantized chunk too small: {} < {}",
                slice.len(),
                crate::quantized::QuantizedChunkHeader::SIZE
            )));
        }
        let header = crate::quantized::QuantizedChunkHeader::decode(
            &slice[..crate::quantized::QuantizedChunkHeader::SIZE],
        )?;
        let payload = &slice[crate::quantized::QuantizedChunkHeader::SIZE..];
        Ok(Some((header, payload)))
    }

    /// Raw payload slice of a named Tensors chunk. Borrows the mmap.
    pub fn tensors_slice_named(&self, name: &str) -> Result<&[u8]> {
        let entry = self.find_tensors_entry_named(name)?;
        self.chunk_slice(entry)
    }

    /// Resolve the byte slice for a named component:
    /// 1. **New layout** — if there's a dedicated `tensors:<component>` chunk, return it.
    /// 2. **Legacy layout** — fall back to the single canonical Tensors chunk.
    ///    The legacy prefix from `bundle_meta.components[component]` is returned
    ///    alongside so callers can route by it.
    pub fn tensors_slice_for(&self, component: &str) -> Result<(&[u8], Option<String>)> {
        if let Ok(slice) = self.tensors_slice_named(component) {
            return Ok((slice, None));
        }
        let prefix = self
            .cdir
            .bundle_meta
            .components
            .get(component)
            .cloned()
            .unwrap_or_else(|| component.to_string());
        let slice = self.tensors_slice()?;
        Ok((slice, if prefix.is_empty() { None } else { Some(prefix) }))
    }

    /// Read an auxiliary file from the bundle by path.
    pub fn read_file(&self, path: &str) -> Result<Cow<'_, [u8]>> {
        let norm = syn_path::normalize(path)?;
        let entry = self
            .cdir
            .find_alive(&norm)
            .ok_or_else(|| Error::FileNotFound(norm.clone()))?;
        let slice = self.chunk_slice(entry)?;
        verify_payload_crc(entry.id, slice, entry.crc32c)?;
        // Bit 0 reserved (was: zstd). If anyone produces a bundle with an
        // unknown bit set, refuse loudly rather than silently decompressing
        // garbage.
        if entry.flags & crate::chunk::CHUNK_FLAG_RESERVED_0 != 0 {
            return Err(Error::Safetensors(format!(
                "chunk `{}` has reserved flag bit 0 set; bundle was likely produced \
                 by a newer reader with an unsupported per-chunk transform",
                entry.name,
            )));
        }
        Ok(Cow::Borrowed(slice))
    }

    /// Iterator over all alive non-tensor files.
    pub fn list_files(&self) -> impl Iterator<Item = &ChunkEntry> + '_ {
        self.cdir
            .entries
            .iter()
            .filter(|e| e.is_alive() && matches!(e.kind_typed(), ChunkType::File))
    }

    /// All alive files under `prefix` (recursive).
    pub fn list_dir<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a ChunkEntry> + 'a {
        self.list_files().filter(move |e| syn_path::is_under(&e.name, prefix))
    }

    /// Immediate children of `prefix` (files and subdir names). Each subdir
    /// name is reported once even if multiple files live under it.
    pub fn list_dir_shallow<'a>(&'a self, prefix: &'a str) -> Vec<DirEntry<'a>> {
        let mut seen_dirs: std::collections::BTreeSet<&str> = Default::default();
        let mut out = Vec::new();
        for e in self.list_files() {
            if let Some((seg, is_dir)) = syn_path::shallow_child(&e.name, prefix) {
                if is_dir {
                    if seen_dirs.insert(seg) {
                        out.push(DirEntry::Subdir(seg));
                    }
                } else {
                    out.push(DirEntry::File(e));
                }
            }
        }
        out
    }

    /// Quick integrity check: revalidates footer/cdir CRC. No payload reads.
    /// (Footer/cdir CRCs are already validated on `open`, so this is mostly
    /// useful to confirm an externally-passed `Bundle` is healthy.)
    pub fn verify_quick(&self) -> Result<()> {
        let _ = self.footer; // already validated on open
        Ok(())
    }

    /// Full integrity check: walk every alive chunk and verify CRC32C.
    /// O(total alive bytes); ~1 GB/s software, faster with hardware CRC32C.
    pub fn verify_full(&self) -> Result<()> {
        for e in self.cdir.entries.iter().filter(|e| e.is_alive()) {
            let slice = self.chunk_slice(e)?;
            verify_payload_crc(e.id, slice, e.crc32c)?;
        }
        Ok(())
    }

    /// Tensors-chunk slice as a borrowed view into the mmap. Returns the
    /// payload-only byte range (no chunk header). **Does not verify CRC32C**
    /// — see comment on `tensor_var_builder`.
    pub fn tensors_slice(&self) -> Result<&[u8]> {
        let entry = self.find_tensors_entry()?;
        self.chunk_slice(entry)
    }

    /// Resolve a single ref by id via `resolver`. The returned `Bundle` owns
    /// its own mmap; callers must keep it alive as long as any derived
    /// view borrows from it.
    pub fn resolve_ref(&self, id: &str, resolver: &dyn RefResolver) -> Result<Bundle> {
        let spec = self
            .cdir
            .bundle_meta
            .refs
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| Error::FileNotFound(format!("ref `{id}` not in bundle_meta")))?;
        resolver.resolve(spec)
    }

    /// Cryptographic integrity check (requires `sha256` feature). Walks every
    /// alive chunk, verifies each `entry.sha256` matches the SHA-256 of its
    /// payload, then recomputes the manifest (hash-of-hashes) and compares to
    /// `BundleMeta.manifest_sha256`.
    ///
    /// Returns `Ok(false)` if the bundle was built without SHA-256 (no
    /// `manifest_sha256` in meta) — caller can decide whether to treat that
    /// as success or as "skipped".
    /// Blake3 равнозначно `verify_sha256` но в 5-10× быстрее. Если bundle
    /// упакован c `with_blake3(true)`, эта проверка предпочтительнее.
    #[cfg(feature = "blake3")]
    pub fn verify_blake3(&self) -> Result<bool> {
        let Some(manifest_expected) = &self.cdir.bundle_meta.manifest_blake3 else {
            return Ok(false);
        };
        let mut h = blake3::Hasher::new();
        for e in self.cdir.entries.iter() {
            let Some(stored) = &e.blake3 else {
                if e.is_alive() {
                    return Err(Error::Safetensors(format!(
                        "chunk #{} missing blake3 in cdir but manifest claims blake3-chunks",
                        e.id
                    )));
                }
                continue;
            };
            if e.is_alive() {
                let slice = self.chunk_slice(e)?;
                let got = blake3::hash(slice);
                if got.as_bytes().as_slice() != stored.as_slice() {
                    return Err(Error::ChunkCrcMismatch {
                        id: e.id,
                        expected: u32::from_be_bytes(stored[0..4].try_into().unwrap()),
                        got: u32::from_be_bytes(got.as_bytes()[0..4].try_into().unwrap()),
                    });
                }
            }
            h.update(stored);
        }
        if h.finalize().as_bytes().as_slice() != manifest_expected.as_slice() {
            return Err(Error::Safetensors(
                "manifest_blake3 mismatch (cdir was tampered)".into(),
            ));
        }
        Ok(true)
    }

    #[cfg(feature = "sha256")]
    pub fn verify_sha256(&self) -> Result<bool> {
        use sha2::{Digest, Sha256};
        let Some(manifest_expected) = &self.cdir.bundle_meta.manifest_sha256 else {
            return Ok(false);
        };
        let mut h = Sha256::new();
        for e in self.cdir.entries.iter() {
            let Some(stored) = &e.sha256 else {
                if e.is_alive() {
                    return Err(Error::Safetensors(format!(
                        "chunk #{} missing sha256 in cdir but manifest claims sha256-manifest",
                        e.id
                    )));
                }
                continue;
            };
            if e.is_alive() {
                let slice = self.chunk_slice(e)?;
                let mut hch = Sha256::new();
                hch.update(slice);
                let got = hch.finalize();
                if got.as_slice() != stored.as_slice() {
                    return Err(Error::ChunkCrcMismatch {
                        id: e.id,
                        expected: u32::from_be_bytes(stored[0..4].try_into().unwrap()),
                        got: u32::from_be_bytes(got[0..4].try_into().unwrap()),
                    });
                }
            }
            h.update(stored);
        }
        let got = h.finalize();
        if got.as_slice() != manifest_expected.as_slice() {
            return Err(Error::Safetensors(
                "manifest_sha256 mismatch (cdir was tampered)".into(),
            ));
        }
        Ok(true)
    }
}

/// Try to parse a footer that *should* be at `offset`. Returns the parsed
/// footer + its absolute byte offset. Caller must ensure
/// `offset + FOOTER_SIZE <= mmap.len()`.
fn locate_trailer(mmap: &[u8], offset: usize) -> Result<(Footer, usize)> {
    let footer = Footer::read_from(&mmap[offset..offset + FOOTER_SIZE])?;
    Ok((footer, offset))
}

/// Walk backwards from EOF in `ALIGNMENT`-byte steps looking for the trailer
/// magic. Used when a recent commit was interrupted mid-write and the latest
/// trailer at `EOF - FOOTER_SIZE` is torn. Append-only commits guarantee an
/// earlier, fully-fsynced trailer exists somewhere in the file.
///
/// Bounded by `MAX_SCAN`: refuses to scan past 256 MB to avoid pathological
/// behaviour on intentionally-broken files. In practice the previous trailer
/// is at most one edit's worth of bytes earlier — usually well under 1 MB.
fn scan_back_for_trailer(mmap: &[u8]) -> Result<(Footer, usize)> {
    const MAX_SCAN: usize = 256 * 1024 * 1024;
    let len = mmap.len();
    if len < FILE_HEADER_SIZE + FOOTER_SIZE {
        return Err(Error::FileTooSmall(len as u64));
    }
    let limit = len.saturating_sub(MAX_SCAN).max(FILE_HEADER_SIZE);
    // Cdir length is variable, so the footer never lands on a predictable
    // alignment. Walk byte-by-byte from EOF back to `limit`. ~256 MB/s on
    // a modern CPU — acceptable for the recovery path, only triggered on
    // a torn tail.
    let mut tail_off = len.saturating_sub(FOOTER_SIZE);
    loop {
        let magic_at = tail_off + FOOTER_SIZE - 8;
        if magic_at + 8 <= len && &mmap[magic_at..magic_at + 8] == TRAILER_MAGIC {
            if let Ok(f) = Footer::read_from(&mmap[tail_off..tail_off + FOOTER_SIZE]) {
                tracing::warn!(
                    "scan-back recovered trailer at offset {} (EOF-{} bytes); the last commit was interrupted",
                    tail_off,
                    len - tail_off
                );
                let _ = ALIGNMENT;
                return Ok((f, tail_off));
            }
        }
        if tail_off == 0 || tail_off <= limit {
            break;
        }
        tail_off -= 1;
    }
    Err(Error::BadMagic { where_: "no valid trailer found via scan-back" })
}
