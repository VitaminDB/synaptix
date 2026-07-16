//! Write-side: pack a HuggingFace-style snapshot into a `.syn` bundle.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::cdir::{
    BundleMeta, CdirFormat, CentralDirectory, ChunkEntry, ChunkType, FileTag, LoraOverlay, RefSpec,
    CHUNK_STATUS_ALIVE,
};
use crate::chunk::{chunk_total_size, header_len, ChunkHeader};
use crate::error::{Error, Result};
use crate::header::{
    align_up, CdirOnDiskFormat, FileHeader, Footer, FILE_HEADER_SIZE, FLAG_HAS_SHA256_MANIFEST,
};
use crate::path as syn_path;
use crate::CURRENT_MINOR;

/// Размер блока для стриминговой копии payload'а: достаточно большой, чтобы
/// амортизировать sys-call overhead, и достаточно мелкий, чтобы прогресс
/// тикал плавно (≈30 тиков на гигабайт).
const COPY_CHUNK: usize = 4 * 1024 * 1024;

/// Событие прогресса сборки бандла. Эмитится из `BundleBuilder::write` через
/// callback, установленный `with_progress`. Callback вызывается в worker-thread
/// — потребитель отвечает за переход на UI-поток (например через
/// `run_on_main_thread`).
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// План: `total_bytes` — объём «работы» (для tensor-источников копирование
    /// идёт в два прохода: stage + pack, поэтому учитывается дважды) — он
    /// используется для расчёта fraction прогресса. `payload_bytes` — реальный
    /// размер payload'а на диске (без удвоения), для пользовательского
    /// отображения «X из Y ГБ». `total_items` — количество логических этапов
    /// (≈ количество чанков). Эмитится **один раз** перед стартом, до первого
    /// ItemStart.
    Plan { total_bytes: u64, total_items: usize, payload_bytes: u64 },
    /// Начался этап `index`: имя (например `tensors:main` или `tokenizer.json`)
    /// и размер payload'а текущего этапа в байтах.
    ItemStart { index: usize, name: String, bytes: u64 },
    /// Скопировано `delta` дополнительных байт текущего этапа (нарастающие
    /// дельты, не абсолютное значение). Сумма дельт равна `bytes` из ItemStart.
    Bytes { delta: u64 },
    /// Этап `index` завершён и (если `with_delete_sources_after_pack` включено)
    /// его исходные файлы удалены с диска.
    ItemDone { index: usize, name: String, deleted_sources: bool },
    /// Все чанки записаны, остаётся cdir+footer и финальный rename — это
    /// быстро и без отчётности.
    Finalizing,
    /// Бандл успешно записан и переименован в `out`.
    Done,
}

/// Callback прогресса. Должен быть Send+Sync, поскольку вызывается из
/// worker-thread в `write()`. Для интеграции с UI оборачивайте обновление
/// сигналов в `run_on_main_thread`.
pub type ProgressCallback = Arc<dyn Fn(ProgressEvent) + Send + Sync>;

/// One safetensors source (one or more shards) optionally prefixed during merge.
/// `name` is the suffix of the resulting Tensors chunk: each source produces
/// a separate chunk named `tensors:<name>`. For single-component bundles the
/// default is `"main"` → `tensors:main` (backward compatible).
struct TensorSource {
    name: String,
    paths: Vec<PathBuf>,
    prefix: Option<String>,
}

/// Auxiliary file pending write.
struct FilePending {
    name: String,
    tag: Option<FileTag>,
    payload: FilePayload,
}

enum FilePayload {
    Owned(Vec<u8>),
    Path(PathBuf),
}

/// Builder for a fresh bundle.
pub struct BundleBuilder {
    meta: BundleMeta,
    tensor_sources: Vec<TensorSource>,
    files: Vec<FilePending>,
    cdir_format: CdirFormat,
    sha256: bool,
    blake3: bool,
    progress: Option<ProgressCallback>,
    /// Удалять исходные файлы (safetensors-шарды + `FilePayload::Path`) с диска
    /// сразу после успешной записи соответствующего чанка в `out.tmp`. По
    /// умолчанию `false`: это деструктивная операция, и UI обязан явно её
    /// подтвердить чекбоксом.
    delete_sources: bool,
}

impl BundleBuilder {
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            meta: BundleMeta {
                id: id.into(),
                version: version.into(),
                ..Default::default()
            },
            tensor_sources: Vec::new(),
            files: Vec::new(),
            cdir_format: CdirFormat::Cbor,
            sha256: false,
            blake3: false,
            progress: None,
            delete_sources: false,
        }
    }

    /// Установить callback прогресса. Вызывается из `write()` в worker-thread:
    /// последовательно Plan → ItemStart/Bytes/ItemDone … → Finalizing → Done.
    /// При ошибке `write()` Done не вызывается.
    pub fn with_progress(mut self, cb: ProgressCallback) -> Self {
        self.progress = Some(cb);
        self
    }

    /// Удалять исходные файлы (safetensors-шарды каждого `TensorSource` и
    /// файлы добавленные через `add_file_path`) сразу после успешной записи
    /// соответствующего чанка в `out.tmp`. По умолчанию `false`. ВКЛЮЧАТЬ
    /// только по явному запросу пользователя.
    pub fn with_delete_sources_after_pack(mut self, enabled: bool) -> Self {
        self.delete_sources = enabled;
        self
    }

    /// Оценка суммарного объёма payload'а, который попадёт в bundle: сумма
    /// размеров safetensors-шардов всех `TensorSource` и размеров файлов
    /// (`FilePayload::Path` → metadata, `FilePayload::Owned` → `bytes.len()`).
    /// Метаданные (FileHeader, ChunkHeader'ы, cdir, footer, align-паддинг) сюда
    /// не входят — это ~единицы килобайт оверхеда на типичный пакет.
    /// `Result` — потому что `fs::metadata` может упасть на битых ссылках.
    pub fn estimate_payload_bytes(&self) -> Result<u64> {
        let mut total: u64 = 0;
        for src in &self.tensor_sources {
            for p in &src.paths {
                total = total.saturating_add(std::fs::metadata(p)?.len());
            }
        }
        for fp in &self.files {
            match &fp.payload {
                FilePayload::Owned(b) => total = total.saturating_add(b.len() as u64),
                FilePayload::Path(p) => {
                    total = total.saturating_add(std::fs::metadata(p)?.len())
                }
            }
        }
        Ok(total)
    }

    /// Максимальный размер одного `TensorSource` — нужен для оценки пикового
    /// потребления свободного места на разделе `out`: во время записи
    /// промежуточный `out.syn.tensors_stage{idx}.tmp` живёт параллельно с
    /// `out.syn.tmp`. Возвращает 0 если тензорных источников нет.
    pub fn max_tensor_source_bytes(&self) -> Result<u64> {
        let mut max: u64 = 0;
        for src in &self.tensor_sources {
            let mut sum: u64 = 0;
            for p in &src.paths {
                sum = sum.saturating_add(std::fs::metadata(p)?.len());
            }
            if sum > max {
                max = sum;
            }
        }
        Ok(max)
    }

    /// Количество логических этапов: один на каждый `TensorSource` + один на
    /// каждый `FilePending`. UI использует это число для подписи прогресс-бара.
    pub fn item_count(&self) -> usize {
        self.tensor_sources.len() + self.files.len()
    }

    pub fn arch(mut self, arch: impl Into<String>) -> Self {
        self.meta.arch = arch.into();
        self
    }

    pub fn purpose(mut self, purpose: impl Into<String>) -> Self {
        self.meta.purpose = purpose.into();
        self
    }

    pub fn component(mut self, name: impl Into<String>, tensor_prefix: impl Into<String>) -> Self {
        self.meta.components.insert(name.into(), tensor_prefix.into());
        self
    }

    pub fn add_ref(mut self, spec: RefSpec) -> Self {
        self.meta.refs.push(spec);
        self
    }

    /// Register a LoRA-style overlay. The merged var-builder will materialise
    /// `W_effective = W_base + α · (B @ A)` for every requested tensor named
    /// `base_tensor`. Adds `tensor-delta-overlays` to `required_caps` since
    /// readers ignoring overlays would silently load incomplete weights.
    pub fn add_lora_overlay(mut self, overlay: LoraOverlay) -> Self {
        self.meta.overlays.push(overlay);
        let cap = crate::CAP_TENSOR_DELTA.to_string();
        if !self.meta.required_caps.contains(&cap) {
            self.meta.required_caps.push(cap);
        }
        self
    }

    pub fn require_capability(mut self, cap: impl Into<String>) -> Self {
        self.meta.required_caps.push(cap.into());
        self
    }

    pub fn optional_capability(mut self, cap: impl Into<String>) -> Self {
        self.meta.optional_caps.push(cap.into());
        self
    }

    pub fn cdir_format(mut self, f: CdirFormat) -> Self {
        self.cdir_format = f;
        self
    }

    /// Записать SHA-256 на каждый chunk и manifest (`hash-of-hashes`) в
    /// `BundleMeta.manifest_sha256`. Требует feature `sha256`; без неё этот
    /// метод компилируется в `panic!` чтобы поломку нельзя было пропустить.
    /// Также добавляет `sha256-manifest` в `optional_caps`.
    pub fn with_sha256(mut self, enabled: bool) -> Self {
        if enabled {
            #[cfg(not(feature = "sha256"))]
            panic!("BundleBuilder::with_sha256 requires feature `sha256`");
            #[cfg(feature = "sha256")]
            {
                self.sha256 = true;
                let cap = crate::CAP_SHA256_MANIFEST.to_string();
                if !self.meta.optional_caps.contains(&cap) {
                    self.meta.optional_caps.push(cap);
                }
            }
        } else {
            self.sha256 = false;
            self.meta.optional_caps.retain(|c| c != crate::CAP_SHA256_MANIFEST);
        }
        self
    }

    /// То же, но Blake3 (~5–10× быстрее SHA-256 для verify). Cap
    /// `blake3-chunks` в `optional_caps`. Можно сочетать с `with_sha256`.
    pub fn with_blake3(mut self, enabled: bool) -> Self {
        if enabled {
            #[cfg(not(feature = "blake3"))]
            panic!("BundleBuilder::with_blake3 requires feature `blake3`");
            #[cfg(feature = "blake3")]
            {
                self.blake3 = true;
                let cap = crate::CAP_BLAKE3_MANIFEST.to_string();
                if !self.meta.optional_caps.contains(&cap) {
                    self.meta.optional_caps.push(cap);
                }
            }
        } else {
            self.blake3 = false;
            self.meta.optional_caps.retain(|c| c != crate::CAP_BLAKE3_MANIFEST);
        }
        self
    }

    /// Single-file safetensors as the tensors chunk. Convenience wrapper for the
    /// common case of one monolithic `model.safetensors`. The resulting chunk
    /// is named `tensors:main`.
    pub fn add_tensors_from_safetensors(mut self, path: impl AsRef<Path>) -> Self {
        self.tensor_sources.push(TensorSource {
            name: "main".into(),
            paths: vec![path.as_ref().to_path_buf()],
            prefix: None,
        });
        self
    }

    /// Multi-shard safetensors merged under an optional tensor-name prefix.
    /// Use this for HuggingFace `model.safetensors.index.json` layouts.
    /// Produces a single `tensors:main` chunk.
    pub fn add_safetensors_shards(
        mut self,
        paths: Vec<PathBuf>,
        prefix: Option<String>,
    ) -> Self {
        self.tensor_sources.push(TensorSource { name: "main".into(), paths, prefix });
        self
    }

    /// One component within a multi-component bundle. Each call produces a
    /// **separate** Tensors chunk named `tensors:<name>`. Tensors inside that
    /// chunk are namespaced under `prefix.*` if `prefix` is provided (commonly
    /// you want `prefix=None` since each component lives in its own chunk and
    /// name collisions across chunks are impossible). Call repeatedly to
    /// register more components.
    ///
    /// Example for OmniVoice (LM + codec):
    /// ```ignore
    /// BundleBuilder::new("omnivoice", "1.0.0")
    ///     .component("lm", "")
    ///     .component("codec", "")
    ///     .add_safetensors_component("lm",    vec![lm_path],    None)
    ///     .add_safetensors_component("codec", vec![codec_path], None)
    ///     .write(out)?;
    /// ```
    pub fn add_safetensors_component(
        mut self,
        name: &str,
        paths: Vec<PathBuf>,
        prefix: Option<&str>,
    ) -> Self {
        self.tensor_sources.push(TensorSource {
            name: name.to_string(),
            paths,
            prefix: prefix.map(String::from),
        });
        self
    }

    /// Auto-resolve safetensors files in `dir`:
    /// 1. single `model.safetensors`;
    /// 2. shards via `model.safetensors.index.json` (HF convention);
    /// 3. fallback: glob `*.safetensors` in `dir`.
    /// Produces a single `tensors:main` chunk.
    pub fn add_safetensors_from_dir(
        mut self,
        dir: impl AsRef<Path>,
        prefix: Option<&str>,
    ) -> Result<Self> {
        let paths = resolve_safetensors_in_dir(dir.as_ref())?;
        self.tensor_sources.push(TensorSource {
            name: "main".into(),
            paths,
            prefix: prefix.map(String::from),
        });
        Ok(self)
    }

    /// Auxiliary file from in-memory bytes.
    pub fn add_file_bytes(
        mut self,
        bundle_path: &str,
        bytes: Vec<u8>,
        tag: FileTag,
    ) -> Result<Self> {
        let normalized = syn_path::normalize(bundle_path)?;
        self.files.push(FilePending {
            name: normalized,
            tag: Some(tag),
            payload: FilePayload::Owned(bytes),
        });
        Ok(self)
    }

    /// Auxiliary file read lazily from disk during `write()`.
    pub fn add_file_path(
        mut self,
        bundle_path: &str,
        on_disk: impl AsRef<Path>,
        tag: FileTag,
    ) -> Result<Self> {
        let normalized = syn_path::normalize(bundle_path)?;
        self.files.push(FilePending {
            name: normalized,
            tag: Some(tag),
            payload: FilePayload::Path(on_disk.as_ref().to_path_buf()),
        });
        Ok(self)
    }

    /// Serialise the bundle to `out`. Writes via `out.tmp` then atomically renames.
    ///
    /// При установленных `with_progress` / `with_delete_sources_after_pack`
    /// эмитит события и удаляет исходные файлы после успешной записи каждого
    /// чанка. Удаление выполняется **до** финального rename — если процесс
    /// упадёт между удалением исходников и rename, в `out.syn.tmp` уже лежит
    /// валидный bundle, но `out` ещё не создан; пользователь должен будет
    /// сам переименовать `out.syn.tmp` → `out`.
    pub fn write(mut self, out: impl AsRef<Path>) -> Result<()> {
        let out = out.as_ref();
        let tmp = out.with_extension("syn.tmp");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.meta.created_at = Some(now);

        let f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)?;
        let mut w = BufWriter::new(f);

        let flags = if self.sha256 { FLAG_HAS_SHA256_MANIFEST } else { 0 };
        let header = FileHeader {
            ver_major: crate::SUPPORTED_MAJOR,
            ver_minor: CURRENT_MINOR,
            flags,
            created_at: now,
        };
        header.write_to(&mut w)?;
        debug_assert_eq!(written_so_far(&mut w)?, FILE_HEADER_SIZE as u64);

        // Emit Plan: для tensor sources прогресс тикает дважды (copy-to-stage
        // и pack-to-bundle), поэтому total = 2 * tensor_bytes + file_bytes.
        // Это даёт линейный прогресс через всю операцию.
        let total_items = self.tensor_sources.len() + self.files.len();
        let total_tensor_bytes: u64 = self
            .tensor_sources
            .iter()
            .flat_map(|s| s.paths.iter())
            .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
            .sum();
        let total_file_bytes: u64 = self
            .files
            .iter()
            .map(|fp| match &fp.payload {
                FilePayload::Owned(b) => b.len() as u64,
                FilePayload::Path(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            })
            .sum();
        let total_bytes = total_tensor_bytes.saturating_mul(2).saturating_add(total_file_bytes);
        let payload_bytes = total_tensor_bytes.saturating_add(total_file_bytes);
        emit(
            &self.progress,
            ProgressEvent::Plan { total_bytes, total_items, payload_bytes },
        );

        let mut cursor: u64 = FILE_HEADER_SIZE as u64;
        let mut entries = Vec::<ChunkEntry>::new();
        let mut next_id: u64 = 1;
        let mut item_index: usize = 0;

        // 1. Tensors chunks — one per `TensorSource`, named `tensors:<source.name>`.
        //    For backward compat, single-source bundles default to `tensors:main`.
        //    Multi-source bundles (OmniVoice lm+codec, …) get separate named
        //    chunks so each component can be mmap'd zero-copy without prefix
        //    routing, and storage tools (BundleEditor::compact) can stream-copy
        //    via mmap instead of materialising a 805-MB heap copy from a
        //    File-chunk fallback.
        let mut tensors_tmps: Vec<PathBuf> = Vec::new();
        for (idx, src) in self.tensor_sources.iter().enumerate() {
            let chunk_name = format!("tensors:{}", src.name);
            let src_bytes: u64 = src
                .paths
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
                .sum();
            // Двойной размер — copy-to-stage + pack-to-bundle. Если merge меняет
            // header, фактический pack может оказаться чуть больше; UI всё равно
            // clamp'ит fraction.
            emit(
                &self.progress,
                ProgressEvent::ItemStart {
                    index: item_index,
                    name: chunk_name.clone(),
                    bytes: src_bytes.saturating_mul(2),
                },
            );
            let tmp_t = out.with_extension(format!("syn.tensors_stage{idx}.tmp"));
            materialise_safetensors_to_file(
                std::slice::from_ref(src),
                &tmp_t,
                self.progress.as_ref(),
            )?;
            cursor = write_chunk_streaming(
                &mut w,
                cursor,
                next_id,
                ChunkType::Tensors,
                &chunk_name,
                None,
                &tmp_t,
                self.sha256,
                self.blake3,
                &mut entries,
                self.progress.as_ref(),
            )?;
            // Stage tmp удаляется сразу — это наш промежуточный буфер.
            let _ = std::fs::remove_file(&tmp_t);
            // Опциональное удаление исходных шардов.
            if self.delete_sources {
                for p in &src.paths {
                    if let Err(e) = std::fs::remove_file(p) {
                        tracing::warn!(
                            "[syn-format] не удалось удалить исходник `{}`: {e}",
                            p.display()
                        );
                    }
                }
            }
            emit(
                &self.progress,
                ProgressEvent::ItemDone {
                    index: item_index,
                    name: chunk_name,
                    deleted_sources: self.delete_sources,
                },
            );
            tensors_tmps.push(tmp_t);
            next_id += 1;
            item_index += 1;
        }

        // 2. File chunks in registration order.
        let files = std::mem::take(&mut self.files);
        for fp in files {
            let name_for_event = fp.name.clone();
            let payload_size: u64 = match &fp.payload {
                FilePayload::Owned(b) => b.len() as u64,
                FilePayload::Path(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            };
            emit(
                &self.progress,
                ProgressEvent::ItemStart {
                    index: item_index,
                    name: name_for_event.clone(),
                    bytes: payload_size,
                },
            );
            match fp.payload {
                FilePayload::Owned(bytes) => {
                    cursor = write_chunk_owned(
                        &mut w,
                        cursor,
                        next_id,
                        ChunkType::File,
                        &fp.name,
                        fp.tag,
                        &bytes,
                        self.sha256,
                        self.blake3,
                        &mut entries,
                        self.progress.as_ref(),
                    )?;
                }
                FilePayload::Path(path) => {
                    cursor = write_chunk_streaming(
                        &mut w,
                        cursor,
                        next_id,
                        ChunkType::File,
                        &fp.name,
                        fp.tag,
                        &path,
                        self.sha256,
                        self.blake3,
                        &mut entries,
                        self.progress.as_ref(),
                    )?;
                    if self.delete_sources {
                        if let Err(e) = std::fs::remove_file(&path) {
                            tracing::warn!(
                                "[syn-format] не удалось удалить исходник `{}`: {e}",
                                path.display()
                            );
                        }
                    }
                }
            }
            emit(
                &self.progress,
                ProgressEvent::ItemDone {
                    index: item_index,
                    name: name_for_event,
                    deleted_sources: self.delete_sources,
                },
            );
            next_id += 1;
            item_index += 1;
        }

        emit(&self.progress, ProgressEvent::Finalizing);

        // Compute manifest hash-of-hashes if sha256 enabled.
        if self.sha256 {
            self.meta.manifest_sha256 = Some(compute_manifest_sha256(&entries));
        }
        if self.blake3 {
            self.meta.manifest_blake3 = Some(compute_manifest_blake3(&entries));
        }

        // 3. Cdir, padded to 64-byte boundary for hygiene.
        let cdir = CentralDirectory { bundle_meta: self.meta.clone(), entries };
        let cdir_bytes = cdir.encode(self.cdir_format)?;
        let cdir_off = align_up(cursor);
        let pad = cdir_off - cursor;
        if pad > 0 {
            let zeros = vec![0u8; pad as usize];
            w.write_all(&zeros)?;
        }
        w.write_all(&cdir_bytes)?;
        let cdir_crc = crc32c::crc32c(&cdir_bytes);

        // 4. Footer.
        let footer = Footer {
            cdir_offset: cdir_off,
            cdir_len: cdir_bytes.len() as u64,
            cdir_crc32c: cdir_crc,
            cdir_format: match self.cdir_format {
                CdirFormat::Cbor => CdirOnDiskFormat::Cbor as u16,
                CdirFormat::Json => CdirOnDiskFormat::Json as u16,
            },
            ver_major: crate::SUPPORTED_MAJOR,
            bundle_crc32c: 0,
        };
        footer.write_to(&mut w)?;
        w.flush()?;
        let f = w.into_inner().map_err(|e| Error::Io(e.into_error()))?;
        f.sync_all()?;
        drop(f);

        std::fs::rename(&tmp, out)?;
        if let Some(dir) = out.parent() {
            if let Ok(d) = File::open(dir) {
                let _ = d.sync_all();
            }
        }
        for t in tensors_tmps {
            let _ = std::fs::remove_file(t);
        }
        emit(&self.progress, ProgressEvent::Done);
        Ok(())
    }
}

#[inline]
fn emit(cb: &Option<ProgressCallback>, ev: ProgressEvent) {
    if let Some(cb) = cb {
        cb(ev);
    }
}

/// Скопировать `src` → `dst` блоками `COPY_CHUNK`, эмитя `Bytes` на каждый
/// блок. Используется в materialise fast-path и для File-чанков с диска.
fn chunked_copy_with_progress(
    src: &Path,
    dst: &Path,
    progress: Option<&ProgressCallback>,
) -> std::io::Result<u64> {
    let mut r = File::open(src).map(std::io::BufReader::new)?;
    let mut w = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(dst)
        .map(BufWriter::new)?;
    let mut buf = vec![0u8; COPY_CHUNK];
    let mut total: u64 = 0;
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        w.write_all(&buf[..n])?;
        if let Some(cb) = progress {
            cb(ProgressEvent::Bytes { delta: n as u64 });
        }
        total = total.saturating_add(n as u64);
    }
    w.flush()?;
    Ok(total)
}

/// Streaming version of `write_chunk` that copies the payload from `src_path`
/// via mmap, computing CRC32C (and optional SHA-256) on the fly. Used for the
/// Tensors chunk so a 19 GB ACE-Step pack doesn't allocate 19 GB on the heap.
/// `progress` ticks `Bytes { delta }` on every `COPY_CHUNK` slice copied to
/// `w` so a large pack-phase shows linear movement.
#[allow(clippy::too_many_arguments)]
fn write_chunk_streaming(
    w: &mut BufWriter<File>,
    cursor: u64,
    id: u64,
    kind: ChunkType,
    name: &str,
    tag: Option<FileTag>,
    src_path: &Path,
    with_sha256: bool,
    with_blake3: bool,
    entries: &mut Vec<ChunkEntry>,
    progress: Option<&ProgressCallback>,
) -> Result<u64> {
    let name_bytes = name.as_bytes();
    if name_bytes.len() > syn_path::MAX_PATH_LEN {
        return Err(Error::InvalidPath { path: name.into(), reason: "name exceeds MAX_PATH_LEN" });
    }
    let f = File::open(src_path)?;
    let payload_len = f.metadata()?.len();
    let mmap = unsafe { memmap2::Mmap::map(&f)? };
    let crc = crc32c::crc32c(&mmap);
    let sha256 = if with_sha256 { Some(sha256_of(&mmap)) } else { None };
    let blake3 = if with_blake3 { Some(blake3_of(&mmap)) } else { None };

    let hlen = header_len(name_bytes.len());
    let entry = ChunkEntry {
        id,
        kind: kind.to_u8(),
        name: name.into(),
        offset: cursor,
        header_len: hlen as u32,
        payload_off: cursor + hlen,
        payload_len,
        raw_len: payload_len,
        flags: 0,
        crc32c: crc,
        status: CHUNK_STATUS_ALIVE,
        sha256,
        blake3,
        tag,
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
    chunk_hdr.write(&mut *w, name_bytes)?;
    // Стримим mmap блоками COPY_CHUNK — это нужно для прогресса и не зависит
    // от размера BufWriter'а: пользователь видит плавное движение даже на
    // 19 ГБ тензорах.
    let mut offset: usize = 0;
    let mmap_slice: &[u8] = &mmap;
    while offset < mmap_slice.len() {
        let end = (offset + COPY_CHUNK).min(mmap_slice.len());
        w.write_all(&mmap_slice[offset..end])?;
        if let Some(cb) = progress {
            cb(ProgressEvent::Bytes { delta: (end - offset) as u64 });
        }
        offset = end;
    }
    let written = hlen + payload_len;
    let pad = align_up(written) - written;
    if pad > 0 {
        let zeros = vec![0u8; pad as usize];
        w.write_all(&zeros)?;
    }
    entries.push(entry);
    Ok(cursor + chunk_total_size(name_bytes.len(), payload_len))
}

fn written_so_far<W: std::io::Seek>(w: &mut W) -> Result<u64> {
    Ok(w.stream_position()?)
}

/// Запись чанка с in-memory payload'ом. Используется для `FilePayload::Owned`.
/// `Bytes { delta }` эмитится одним событием по факту: in-memory payload'ы
/// мелкие (десятки/сотни КБ — tokenizer.json, config.json), и дробить там
/// нечего.
#[allow(clippy::too_many_arguments)]
fn write_chunk_owned(
    w: &mut BufWriter<File>,
    cursor: u64,
    id: u64,
    kind: ChunkType,
    name: &str,
    tag: Option<FileTag>,
    payload: &[u8],
    with_sha256: bool,
    with_blake3: bool,
    entries: &mut Vec<ChunkEntry>,
    progress: Option<&ProgressCallback>,
) -> Result<u64> {
    let name_bytes = name.as_bytes();
    if name_bytes.len() > syn_path::MAX_PATH_LEN {
        return Err(Error::InvalidPath { path: name.into(), reason: "name exceeds MAX_PATH_LEN" });
    }
    let hlen = header_len(name_bytes.len());
    let sha256 = if with_sha256 { Some(sha256_of(payload)) } else { None };
    let blake3 = if with_blake3 { Some(blake3_of(payload)) } else { None };
    let entry = ChunkEntry {
        id,
        kind: kind.to_u8(),
        name: name.into(),
        offset: cursor,
        header_len: hlen as u32,
        payload_off: cursor + hlen,
        payload_len: payload.len() as u64,
        raw_len: payload.len() as u64,
        flags: 0,
        crc32c: crc32c::crc32c(payload),
        status: CHUNK_STATUS_ALIVE,
        sha256,
        blake3,
        tag,
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
    chunk_hdr.write(&mut *w, name_bytes)?;
    w.write_all(payload)?;
    if let Some(cb) = progress {
        cb(ProgressEvent::Bytes { delta: payload.len() as u64 });
    }
    let written = hlen + payload.len() as u64;
    let pad = align_up(written) - written;
    if pad > 0 {
        let zeros = vec![0u8; pad as usize];
        w.write_all(&zeros)?;
    }
    entries.push(entry);
    Ok(cursor + chunk_total_size(name_bytes.len(), payload.len() as u64))
}

/// SHA-256 of `bytes` as a 32-byte vector. Without the `sha256` feature this
/// is unreachable: `write_chunk` only calls it when `with_sha256=true`, which
/// only `with_sha256(true)` can set, and the latter panics without the feature.
#[cfg(feature = "sha256")]
fn sha256_of(bytes: &[u8]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().to_vec()
}

#[cfg(not(feature = "sha256"))]
fn sha256_of(_: &[u8]) -> Vec<u8> {
    unreachable!("sha256_of called without `sha256` feature; with_sha256(true) panics earlier");
}

/// SHA-256 over the concatenation of per-chunk SHA-256 hashes, in cdir order.
/// Catches reorderings and tombstone forgeries.
#[cfg(feature = "sha256")]
fn compute_manifest_sha256(entries: &[ChunkEntry]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for e in entries {
        if let Some(s) = &e.sha256 {
            h.update(s);
        }
    }
    h.finalize().to_vec()
}

#[cfg(not(feature = "sha256"))]
fn compute_manifest_sha256(_: &[ChunkEntry]) -> Vec<u8> {
    unreachable!()
}

#[cfg(feature = "blake3")]
fn blake3_of(bytes: &[u8]) -> Vec<u8> {
    blake3::hash(bytes).as_bytes().to_vec()
}

#[cfg(not(feature = "blake3"))]
fn blake3_of(_: &[u8]) -> Vec<u8> {
    unreachable!("blake3_of called without `blake3` feature");
}

#[cfg(feature = "blake3")]
fn compute_manifest_blake3(entries: &[ChunkEntry]) -> Vec<u8> {
    let mut h = blake3::Hasher::new();
    for e in entries {
        if let Some(s) = &e.blake3 {
            h.update(s);
        }
    }
    h.finalize().as_bytes().to_vec()
}

#[cfg(not(feature = "blake3"))]
fn compute_manifest_blake3(_: &[ChunkEntry]) -> Vec<u8> {
    unreachable!()
}

/// Merge every safetensors shard into one safetensors-1.0 stream, optionally
/// renaming tensors with a per-source prefix, writing the result to
/// `out_path`. mmap each shard so memory cost stays at ~O(metadata),
/// independent of total tensor data size.
///
/// Fast path: one shard, no prefix — verbatim copy from disk to disk.
/// При наличии `progress` fast-path использует chunked-copy с побайтовым
/// тиканьем; merge-path выдаёт суммарный `Bytes`-тик одним блоком по факту
/// готового stage'а (serialize_to_file непрозрачен).
fn materialise_safetensors_to_file(
    sources: &[TensorSource],
    out_path: &Path,
    progress: Option<&ProgressCallback>,
) -> Result<()> {
    if sources.is_empty() {
        return Err(Error::Safetensors("no safetensors paths provided".into()));
    }

    // Fast path: single shard with no prefix.
    if sources.len() == 1 && sources[0].paths.len() == 1 && sources[0].prefix.is_none() {
        let src = &sources[0].paths[0];
        // Validate via mmap-deserialise (constant-time on header).
        let f = std::fs::File::open(src)?;
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        safetensors::SafeTensors::deserialize(&mmap)?;
        drop(mmap);
        drop(f);
        // Chunked copy с прогрессом.
        chunked_copy_with_progress(src, out_path, progress)?;
        return Ok(());
    }

    // Mmap every shard. SafeTensors views below borrow from these mmaps —
    // they all live for the duration of this function.
    struct ShardMmap {
        mmap: memmap2::Mmap,
        prefix: Option<String>,
    }
    let mut mmaps: Vec<ShardMmap> = Vec::new();
    for src in sources {
        if src.paths.is_empty() {
            return Err(Error::Safetensors("empty paths in TensorSource".into()));
        }
        for p in &src.paths {
            let f = std::fs::File::open(p)?;
            let mmap = unsafe { memmap2::Mmap::map(&f)? };
            mmaps.push(ShardMmap { mmap, prefix: src.prefix.clone() });
        }
    }

    let mut sts: Vec<(safetensors::SafeTensors<'_>, &Option<String>)> =
        Vec::with_capacity(mmaps.len());
    for m in &mmaps {
        let st = safetensors::SafeTensors::deserialize(&m.mmap)?;
        sts.push((st, &m.prefix));
    }
    let mut merged: BTreeMap<String, safetensors::tensor::TensorView<'_>> = BTreeMap::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (st, prefix) in &sts {
        for (name, view) in st.tensors() {
            let final_name = match prefix {
                // Treat Some("") the same as None — useful when the CLI lets
                // a user write `--component foo:dir:` for an explicitly
                // unprefixed component (no `.name` cosmetic prefix).
                Some(p) if !p.is_empty() => format!("{p}.{name}"),
                _ => name.to_string(),
            };
            if !seen.insert(final_name.clone()) {
                return Err(Error::Safetensors(format!(
                    "duplicate tensor name across shards/components: `{final_name}`"
                )));
            }
            merged.insert(final_name, view);
        }
    }

    // Stream-write the merged stream: safetensors::serialize_to_file writes
    // the header and then each tensor's `.data()` slice via a `BufWriter`.
    // Since `.data()` is a mmap-backed `&[u8]`, this is a kernel pagecache
    // sequential read → write copy. Peak RSS stays near zero.
    //
    // `serialize_to_file` непрозрачен для прогресса — поднимаем фоновый
    // поток, который раз в 250 мс читает `metadata(out_path).len()` и
    // эмитит дельту роста размера. На выходе досылаем оставшийся хвост
    // (между последним поллом и финальным fsync).
    let monitor_cancel = Arc::new(AtomicBool::new(false));
    let monitor_handle = progress.cloned().map(|cb| {
        let cancel = monitor_cancel.clone();
        let path = out_path.to_path_buf();
        std::thread::spawn(move || -> u64 {
            let mut last: u64 = 0;
            while !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(250));
                let cur = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                if cur > last {
                    cb(ProgressEvent::Bytes { delta: cur - last });
                    last = cur;
                }
            }
            last
        })
    });

    let result = safetensors::serialize_to_file(merged, None, out_path)
        .map_err(|e| Error::Safetensors(e.to_string()));

    monitor_cancel.store(true, Ordering::Relaxed);
    let reported_so_far = monitor_handle
        .and_then(|h| h.join().ok())
        .unwrap_or(0);
    result?;

    if let Some(cb) = progress {
        let final_size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
        if final_size > reported_so_far {
            cb(ProgressEvent::Bytes { delta: final_size - reported_so_far });
        }
    }
    Ok(())
}

/// Resolve safetensors files in a HuggingFace-style directory:
/// 1. `model.safetensors` (single shard) → returns just that.
/// 2. `model.safetensors.index.json` → returns every distinct shard referenced
///    in `weight_map`, sorted lexicographically.
/// 3. fallback: glob `*.safetensors` in `dir`, sorted.
///
/// Returned paths are absolute (relative to `dir`).
pub fn resolve_safetensors_in_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Err(Error::Safetensors(format!("not a directory: {}", dir.display())));
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let raw = std::fs::read(&index)?;
        let v: serde_json::Value = serde_json::from_slice(&raw)
            .map_err(|e| Error::Safetensors(format!("model.safetensors.index.json: {e}")))?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| Error::Safetensors("index.json missing weight_map".into()))?;
        let mut files: BTreeSet<PathBuf> = BTreeSet::new();
        for (_, file) in map {
            if let Some(name) = file.as_str() {
                files.insert(dir.join(name));
            }
        }
        if files.is_empty() {
            return Err(Error::Safetensors("weight_map empty".into()));
        }
        return Ok(files.into_iter().collect());
    }
    // Fallback glob.
    let mut files: BTreeSet<PathBuf> = BTreeSet::new();
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let p = e.path();
        if p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            files.insert(p);
        }
    }
    if files.is_empty() {
        return Err(Error::Safetensors(format!("no .safetensors in {}", dir.display())));
    }
    Ok(files.into_iter().collect())
}
