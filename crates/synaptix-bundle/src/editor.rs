//! In-place editing of `.syn` bundles.
//!
//! **Append-only journal layout**: every `commit()` only writes past the
//! previous EOF; nothing already on disk is overwritten. After a sequence
//! of edits the file looks like:
//!
//! ```text
//! [hdr] [chunks v0] [cdir v0] [footer v0]   ← original state
//!                            [chunks v1+] [cdir v1] [footer v1]   ← edit 1
//!                                                  [chunks v2+] [cdir v2] [footer v2]
//! ```
//!
//! Readers normally consume only `footer-vN` (last 40 bytes of file). On
//! crash mid-write, the *previous* footer is still the last fully-fsynced
//! `SYNEND\0\0` record — `Bundle::open` finds it via scan-back recovery
//! (see `bundle.rs`).
//!
//! `add_file` / `remove_file` / `rename` are O(new_bytes + new_cdir_size).
//! Disk consumed per edit grows; reclaim via `compact()`.
//!
//! Concurrent editing: `BundleEditor::open` takes an advisory exclusive
//! `flock` via `fs4` — a second `open` on the same file returns
//! `Error::BundleBusy` without blocking.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::bundle::Bundle;
use crate::cdir::{
    BundleMeta, CdirFormat, CentralDirectory, ChunkEntry, ChunkType, FileTag, CHUNK_STATUS_ALIVE,
    CHUNK_STATUS_TOMBSTONE,
};
use crate::chunk::{chunk_total_size, header_len, ChunkHeader};
use crate::error::{Error, Result};
use crate::header::{align_up, CdirOnDiskFormat, Footer};
use crate::path as syn_path;

struct PendingChunk {
    name: String,
    kind: ChunkType,
    payload: Vec<u8>,
    tag: Option<FileTag>,
}

pub struct BundleEditor {
    path: PathBuf,
    file: File,
    bundle: Bundle,
    pending_chunks: Vec<PendingChunk>,
    pending_tombstones: BTreeSet<u64>,
    pending_renames: BTreeMap<u64, String>,
    /// Свежий `BundleMeta`, который заменит существующий при `commit`. `None` —
    /// meta не меняется (значит `commit` запишет `self.bundle.meta().clone()`).
    pending_meta: Option<BundleMeta>,
    next_id: u64,
    cdir_format: CdirFormat,
}

impl BundleEditor {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        use fs4::fs_std::FileExt;
        let p = path.as_ref().to_path_buf();
        let bundle = Bundle::open(&p)?;
        // Re-open R/W and take an advisory exclusive lock. Two parallel
        // `BundleEditor::open` calls on the same file would otherwise race
        // and corrupt the cdir tail. `try_lock_exclusive` is non-blocking —
        // a busy bundle returns `BundleBusy` immediately rather than
        // deadlocking.
        let file = OpenOptions::new().read(true).write(true).open(&p)?;
        match FileExt::try_lock_exclusive(&file) {
            Ok(true) => {}
            Ok(false) | Err(_) => return Err(Error::BundleBusy(p)),
        }
        let next_id = bundle.cdir().entries.iter().map(|e| e.id).max().unwrap_or(0) + 1;
        let cdir_format = match bundle.footer().cdir_format {
            0 => CdirFormat::Cbor,
            1 => CdirFormat::Json,
            _ => CdirFormat::Cbor,
        };
        Ok(Self {
            path: p,
            file,
            bundle,
            pending_chunks: Vec::new(),
            pending_tombstones: BTreeSet::new(),
            pending_renames: BTreeMap::new(),
            pending_meta: None,
            next_id,
            cdir_format,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn meta(&self) -> &BundleMeta {
        self.pending_meta.as_ref().unwrap_or_else(|| self.bundle.meta())
    }

    /// Поставить новые `BundleMeta` для применения на ближайшем `commit`.
    /// `id`, `version`, `arch`, `purpose`, `components`, `refs`, capabilities —
    /// всё перезаписывается целиком. Поля `manifest_sha256`/`manifest_blake3`
    /// пересчитывать здесь не нужно: текущая реализация bundle не пишет их в
    /// commit (только в builder.write). При необходимости полной перепаковки
    /// с новой meta используйте `compact()`.
    pub fn set_meta(&mut self, meta: BundleMeta) {
        self.pending_meta = Some(meta);
    }

    /// Append a new file chunk. Effective only after `commit()`.
    pub fn add_file(&mut self, bundle_path: &str, bytes: Vec<u8>, tag: FileTag) -> Result<()> {
        let name = syn_path::normalize(bundle_path)?;
        if self.is_alive_after_pending(&name) {
            return Err(Error::InvalidPath {
                path: name,
                reason: "already exists; use replace_file or remove_file first",
            });
        }
        self.pending_chunks.push(PendingChunk { name, kind: ChunkType::File, payload: bytes, tag: Some(tag) });
        Ok(())
    }

    /// Добавить (или заменить) quantized-tensors chunk
    /// (`CHUNK_TYPE_QUANTIZED_TENSORS = 6`). `payload` уже должен включать
    /// 16-байтный [`crate::quantized::QuantizedChunkHeader`] в начале —
    /// этот метод просто кладёт байты в chunk с именем
    /// [`crate::quantized::QUANTIZED_CHUNK_NAME`].
    ///
    /// Если в bundle уже есть alive quantized chunk — он tombstone'ится
    /// и новый кладётся рядом (commit обработает компакт).
    pub fn add_quantized_chunk(&mut self, payload: Vec<u8>) -> Result<()> {
        let name = crate::quantized::QUANTIZED_CHUNK_NAME.to_string();

        // Tombstone существующий quantized chunk, если есть.
        if let Some(id) = self
            .bundle
            .cdir()
            .entries
            .iter()
            .rev()
            .find(|e| {
                e.is_alive()
                    && matches!(e.kind_typed(), ChunkType::QuantizedTensors)
                    && !self.pending_tombstones.contains(&e.id)
            })
            .map(|e| e.id)
        {
            self.pending_tombstones.insert(id);
        }

        // Удалить pending с тем же именем (overwrite).
        if let Some(idx) = self
            .pending_chunks
            .iter()
            .rposition(|p| p.name == name && matches!(p.kind, ChunkType::QuantizedTensors))
        {
            self.pending_chunks.remove(idx);
        }

        self.pending_chunks.push(PendingChunk {
            name,
            kind: ChunkType::QuantizedTensors,
            payload,
            tag: None,
        });
        Ok(())
    }

    /// Tombstone the chunk that currently holds `bundle_path`.
    pub fn remove_file(&mut self, bundle_path: &str) -> Result<()> {
        let name = syn_path::normalize(bundle_path)?;
        // Are we removing a not-yet-committed addition? Just drop it.
        if let Some(idx) = self.pending_chunks.iter().rposition(|p| p.name == name) {
            self.pending_chunks.remove(idx);
            return Ok(());
        }
        let id = self
            .bundle
            .cdir()
            .entries
            .iter()
            .rev()
            .find(|e| {
                e.is_alive()
                    && !self.pending_tombstones.contains(&e.id)
                    && self.effective_name(e) == name
            })
            .map(|e| e.id)
            .ok_or_else(|| Error::FileNotFound(name.clone()))?;
        self.pending_tombstones.insert(id);
        Ok(())
    }

    /// Rename an existing file. Renaming a not-yet-committed addition is
    /// allowed (just updates its pending name).
    pub fn rename(&mut self, old: &str, new: &str) -> Result<()> {
        let old = syn_path::normalize(old)?;
        let new = syn_path::normalize(new)?;
        if self.is_alive_after_pending(&new) {
            return Err(Error::InvalidPath {
                path: new,
                reason: "destination already exists",
            });
        }
        if let Some(p) = self.pending_chunks.iter_mut().rfind(|p| p.name == old) {
            p.name = new;
            return Ok(());
        }
        let id = self
            .bundle
            .cdir()
            .entries
            .iter()
            .rev()
            .find(|e| {
                e.is_alive()
                    && !self.pending_tombstones.contains(&e.id)
                    && self.effective_name(e) == old
            })
            .map(|e| e.id)
            .ok_or_else(|| Error::FileNotFound(old.clone()))?;
        self.pending_renames.insert(id, new);
        Ok(())
    }

    /// Tombstone the existing entry and queue a new one with the same name.
    pub fn replace_file(
        &mut self,
        bundle_path: &str,
        bytes: Vec<u8>,
        tag: FileTag,
    ) -> Result<()> {
        let name = syn_path::normalize(bundle_path)?;
        // If there's an already-queued add with this name, just overwrite its bytes.
        if let Some(p) = self.pending_chunks.iter_mut().rfind(|p| p.name == name) {
            p.payload = bytes;
            p.tag = Some(tag);
            return Ok(());
        }
        // Otherwise tombstone the existing alive entry (if any) and queue the new one.
        if let Some(e) = self
            .bundle
            .cdir()
            .entries
            .iter()
            .rev()
            .find(|e| {
                e.is_alive()
                    && !self.pending_tombstones.contains(&e.id)
                    && self.effective_name(e) == name
            })
        {
            self.pending_tombstones.insert(e.id);
        }
        self.pending_chunks.push(PendingChunk { name, kind: ChunkType::File, payload: bytes, tag: Some(tag) });
        Ok(())
    }

    /// Number of queued operations.
    pub fn pending_count(&self) -> usize {
        self.pending_chunks.len()
            + self.pending_tombstones.len()
            + self.pending_renames.len()
            + usize::from(self.pending_meta.is_some())
    }

    /// Flush all pending changes to disk.
    ///
    /// **Append-only layout**: nothing already written is overwritten.
    /// New chunks land beyond the current EOF; a fresh cdir+footer is
    /// appended after them. The previous cdir+footer stay in place mid-file
    /// — readers walking back from EOF for the trailer magic skip them.
    ///
    /// Crash semantics: if the process is killed between writing new chunks
    /// and the final footer, the old trailer is still the **last** valid
    /// `SYNEND\0\0`-marked record in the file. `Bundle::open`'s scan-back
    /// recovery finds it. The bundle is never observable in a corrupt
    /// intermediate state.
    ///
    /// The trade-off is that an edit grows the file by `(new_payload +
    /// new_cdir + 40)` bytes even when it merely tombstones an entry —
    /// reclaim via `compact()` when the tail metadata becomes large.
    pub fn commit(mut self) -> Result<()> {
        if self.pending_count() == 0 {
            return Ok(());
        }

        // 1. Compute the new central directory entry list.
        let mut new_entries: Vec<ChunkEntry> = self.bundle.cdir().entries.clone();
        for e in &mut new_entries {
            if self.pending_tombstones.contains(&e.id) {
                e.status = CHUNK_STATUS_TOMBSTONE;
            }
            if let Some(new_name) = self.pending_renames.get(&e.id) {
                e.name = new_name.clone();
            }
        }

        // 2. Append-only: start past the previous EOF so nothing is overwritten.
        let mut cursor = self.file.metadata()?.len();
        self.file.seek(SeekFrom::Start(cursor))?;

        // 3. Append new chunks.
        for pc in &self.pending_chunks {
            let name_bytes = pc.name.as_bytes();
            if name_bytes.len() > syn_path::MAX_PATH_LEN {
                return Err(Error::InvalidPath { path: pc.name.clone(), reason: "name too long" });
            }
            let hlen = header_len(name_bytes.len());
            let crc = crc32c::crc32c(&pc.payload);
            let entry = ChunkEntry {
                id: self.next_id,
                kind: pc.kind.to_u8(),
                name: pc.name.clone(),
                offset: cursor,
                header_len: hlen as u32,
                payload_off: cursor + hlen,
                payload_len: pc.payload.len() as u64,
                raw_len: pc.payload.len() as u64,
                flags: 0,
                crc32c: crc,
                status: CHUNK_STATUS_ALIVE,
                sha256: None,
                blake3: None,
                tag: pc.tag,
                metadata: Default::default(),
            };
            let chunk_hdr = ChunkHeader {
                id: entry.id,
                kind: entry.kind as u16,
                flags: entry.flags,
                payload_len: entry.payload_len,
                raw_len: entry.raw_len,
                payload_crc32c: entry.crc32c,
                name_len: name_bytes.len() as u16,
            };
            chunk_hdr.write(&mut self.file, name_bytes)?;
            self.file.write_all(&pc.payload)?;
            let written = hlen + pc.payload.len() as u64;
            let pad = align_up(written) - written;
            if pad > 0 {
                self.file.write_all(&vec![0u8; pad as usize])?;
            }
            cursor += chunk_total_size(name_bytes.len(), entry.payload_len);
            self.next_id += 1;
            new_entries.push(entry);
        }

        // 4. Pad to 64-byte boundary before cdir.
        let cdir_off = align_up(cursor);
        let pad = cdir_off - cursor;
        if pad > 0 {
            self.file.write_all(&vec![0u8; pad as usize])?;
        }

        // 5. Encode + write the new cdir. Если `set_meta` был вызван — берём
        //    свежий BundleMeta, иначе сохраняем существующий из mmap.
        let bundle_meta = self
            .pending_meta
            .clone()
            .unwrap_or_else(|| self.bundle.meta().clone());
        let cdir = CentralDirectory {
            bundle_meta,
            entries: new_entries,
        };
        let cdir_bytes = cdir.encode(self.cdir_format)?;
        self.file.write_all(&cdir_bytes)?;
        let cdir_crc = crc32c::crc32c(&cdir_bytes);

        // 6. Write new footer at EOF.
        let footer = Footer {
            cdir_offset: cdir_off,
            cdir_len: cdir_bytes.len() as u64,
            cdir_crc32c: cdir_crc,
            cdir_format: match self.cdir_format {
                CdirFormat::Cbor => CdirOnDiskFormat::Cbor as u16,
                CdirFormat::Json => CdirOnDiskFormat::Json as u16,
            },
            ver_major: self.bundle.file_header().ver_major,
            bundle_crc32c: 0,
        };
        // fsync chunks + cdir bytes before the footer — guarantees that a
        // crash between "footer is fully written" and "fsync happens" still
        // leaves the previous trailer as the last valid record on disk.
        self.file.sync_data()?;
        footer.write_to(&mut self.file)?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Look up the effective name of a chunk taking pending renames into account.
    fn effective_name<'a>(&'a self, e: &'a ChunkEntry) -> &'a str {
        self.pending_renames.get(&e.id).map(String::as_str).unwrap_or(&e.name)
    }

    /// Whether a path is currently held by either a committed-alive entry or
    /// a pending addition (and not pending tombstoned).
    fn is_alive_after_pending(&self, name: &str) -> bool {
        if self.pending_chunks.iter().any(|p| p.name == name) {
            return true;
        }
        self.bundle.cdir().entries.iter().any(|e| {
            e.is_alive()
                && !self.pending_tombstones.contains(&e.id)
                && self.effective_name(e) == name
        })
    }
}

/// Rebuild a bundle without tombstoned chunks. Reads `src` and writes a
/// fresh bundle at `dst`, then atomically renames into place. Useful after
/// many `remove_file` / `replace_file` operations to reclaim disk space.
///
/// `src` and `dst` can be the same path (the function uses a sibling temp).
pub fn compact(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<()> {
    use crate::builder::BundleBuilder;
    let src = src.as_ref();
    let dst = dst.as_ref();
    let b = Bundle::open(src)?;
    let mut builder = BundleBuilder::new(b.meta().id.clone(), b.meta().version.clone())
        .arch(b.meta().arch.clone())
        .purpose(b.meta().purpose.clone())
        .cdir_format(match b.footer().cdir_format {
            0 => CdirFormat::Cbor,
            1 => CdirFormat::Json,
            _ => CdirFormat::Cbor,
        });
    for (k, v) in &b.meta().components {
        builder = builder.component(k.clone(), v.clone());
    }
    for r in &b.meta().refs {
        builder = builder.add_ref(r.clone());
    }
    for cap in &b.meta().required_caps {
        builder = builder.require_capability(cap.clone());
    }
    for cap in &b.meta().optional_caps {
        builder = builder.optional_capability(cap.clone());
    }

    // Preserve SHA-256 manifest setting: if the source bundle had
    // `manifest_sha256` populated, regenerate it on the compacted output too.
    // Without this, every compact() silently downgrades cryptographic
    // integrity guarantees of the original pack.
    #[cfg(feature = "sha256")]
    if b.meta().manifest_sha256.is_some() {
        builder = builder.with_sha256(true);
    }

    // Stage the tensors stream to a sibling temp so the builder can validate it
    // as plain safetensors at pack time.
    let tmp_tensors = dst.with_extension("tensors.tmp");
    let mut tmp_written = false;
    for e in b.cdir().entries.iter().filter(|e| e.is_alive()) {
        match e.kind_typed() {
            ChunkType::Tensors => {
                let bytes = b.read_raw_chunk(e)?;
                std::fs::write(&tmp_tensors, &*bytes)?;
                builder = builder.add_tensors_from_safetensors(&tmp_tensors);
                tmp_written = true;
            }
            ChunkType::File => {
                let bytes = b.read_file(&e.name)?;
                let tag = e.tag.unwrap_or(FileTag::Inference);
                builder = builder.add_file_bytes(&e.name, bytes.into_owned(), tag)?;
            }
            _ => {}
        }
    }
    if !tmp_written {
        let _ = std::fs::remove_file(&tmp_tensors);
        return Err(Error::TensorsChunkMissing);
    }
    builder.write(dst)?;
    let _ = std::fs::remove_file(&tmp_tensors);
    Ok(())
}
