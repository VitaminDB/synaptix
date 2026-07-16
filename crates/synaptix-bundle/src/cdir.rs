//! Central directory: CBOR-encoded list of chunk entries plus bundle metadata.
//!
//! The cdir is the single source of truth for "what lives in this bundle";
//! chunk-internal headers are a redundant copy used only for sanity checks
//! after a corrupt-cdir recovery.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

pub const CHUNK_TYPE_TENSORS: u8 = 1;
pub const CHUNK_TYPE_FILE: u8 = 2;
pub const CHUNK_TYPE_REF: u8 = 3;
pub const CHUNK_TYPE_META: u8 = 4;
pub const CHUNK_TYPE_TENSOR_DELTA: u8 = 5;
/// Quantized tensors chunk. Payload prefix — [`QuantizedChunkHeader`]
/// (16 bytes), затем формат-зависимые данные. Поддерживаемые форматы — см.
/// `quantized.rs`. ID=1 (бывший GGUF) deprecated, ID≥2 — native syn-quant.
pub const CHUNK_TYPE_QUANTIZED_TENSORS: u8 = 6;

pub const CHUNK_STATUS_ALIVE: u8 = 0;
pub const CHUNK_STATUS_TOMBSTONE: u8 = 1;

/// File-purpose hint. Loaders skip non-`Inference` chunks unless asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FileTag {
    #[default]
    Inference,
    Doc,
    Example,
    Asset,
}

impl FileTag {
    pub fn as_str(self) -> &'static str {
        match self {
            FileTag::Inference => "inference",
            FileTag::Doc => "doc",
            FileTag::Example => "example",
            FileTag::Asset => "asset",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    Tensors,
    File,
    Ref,
    Meta,
    TensorDelta,
    QuantizedTensors,
    Unknown(u8),
}

impl ChunkType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            CHUNK_TYPE_TENSORS => Self::Tensors,
            CHUNK_TYPE_FILE => Self::File,
            CHUNK_TYPE_REF => Self::Ref,
            CHUNK_TYPE_META => Self::Meta,
            CHUNK_TYPE_TENSOR_DELTA => Self::TensorDelta,
            CHUNK_TYPE_QUANTIZED_TENSORS => Self::QuantizedTensors,
            other => Self::Unknown(other),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Tensors => CHUNK_TYPE_TENSORS,
            Self::File => CHUNK_TYPE_FILE,
            Self::Ref => CHUNK_TYPE_REF,
            Self::Meta => CHUNK_TYPE_META,
            Self::TensorDelta => CHUNK_TYPE_TENSOR_DELTA,
            Self::QuantizedTensors => CHUNK_TYPE_QUANTIZED_TENSORS,
            Self::Unknown(v) => v,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStatus {
    Alive,
    Tombstone,
}

impl ChunkStatus {
    pub fn from_u8(v: u8) -> Self {
        if v == CHUNK_STATUS_TOMBSTONE { Self::Tombstone } else { Self::Alive }
    }
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Alive => CHUNK_STATUS_ALIVE,
            Self::Tombstone => CHUNK_STATUS_TOMBSTONE,
        }
    }
    pub fn is_alive(self) -> bool {
        matches!(self, Self::Alive)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkEntry {
    pub id: u64,
    /// `ChunkType::to_u8()`
    #[serde(rename = "type")]
    pub kind: u8,
    pub name: String,
    /// Offset of the chunk header in the bundle file.
    pub offset: u64,
    /// On-disk header length (chunk header + padding to 64B).
    pub header_len: u32,
    /// Offset where payload bytes begin (= offset + header_len).
    pub payload_off: u64,
    /// Length of payload as stored on disk (compressed if `flags & ZSTD`).
    pub payload_len: u64,
    /// Length of payload after decompression (== payload_len when not compressed).
    pub raw_len: u64,
    pub flags: u16,
    pub crc32c: u32,
    /// `ChunkStatus::to_u8()`
    #[serde(default)]
    pub status: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<Vec<u8>>,
    /// Per-chunk Blake3 hash (32 bytes). Independent of `sha256` — both can
    /// coexist; `verify_blake3()` checks this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blake3: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<FileTag>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, ciborium::Value>,
}

impl ChunkEntry {
    pub fn kind_typed(&self) -> ChunkType {
        ChunkType::from_u8(self.kind)
    }
    pub fn status_typed(&self) -> ChunkStatus {
        ChunkStatus::from_u8(self.status)
    }
    pub fn is_alive(&self) -> bool {
        self.status_typed().is_alive()
    }
}

/// LoRA-style low-rank delta applied to a base tensor at load time:
/// `W_effective = W_base + α · (B @ A)` where `A: (rank, in)`, `B: (out, rank)`.
///
/// Tensors `lora_a` and `lora_b` are looked up in the merged backend (which
/// may resolve them in the bundle itself or in any of the cross-bundle refs).
/// `base_tensor` names the tensor in the parent bundle to overlay.
///
/// Use case: a fine-tune lives as a small `*.syn` containing just A/B
/// matrices + an overlay list, plus a `RefSpec` pointing at the frozen base
/// bundle (e.g. `voxcpm2.syn`). Loader merges at `tensor_var_builder` time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoraOverlay {
    /// Name of the base-bundle tensor being overlaid.
    pub base_tensor: String,
    /// Tensor name for the LoRA A matrix (rank, in_features).
    pub lora_a: String,
    /// Tensor name for the LoRA B matrix (out_features, rank).
    pub lora_b: String,
    /// Scaling factor (commonly `lora_alpha / rank`).
    pub alpha: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RefSpec {
    pub id: String,
    /// 32 bytes. Empty means "match by id only".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sha256: Vec<u8>,
    #[serde(default)]
    pub version_range: String,
    #[serde(default)]
    pub purpose: String,
    #[serde(default)]
    pub tensor_prefix: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub search_paths: Vec<String>,
}

/// Bundle-level metadata (model id, version, capabilities, refs, components).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BundleMeta {
    pub id: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub purpose: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
    /// `tensor_prefix` per logical component, e.g. `{"audiovae": "audiovae", "dit": "locdit"}`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub components: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refs: Vec<RefSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_caps: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub optional_caps: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_sha256: Option<Vec<u8>>,
    /// Blake3 hash-of-hashes over per-chunk `blake3` fields, in cdir order.
    /// Independent from manifest_sha256 — both can be present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_blake3: Option<Vec<u8>>,
    /// LoRA-style tensor overlays applied at load time. Non-empty here
    /// requires the `tensor-delta-overlays` capability in `required_caps`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub overlays: Vec<LoraOverlay>,
    /// Any additional producer-supplied metadata (kept as-is).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, ciborium::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CentralDirectory {
    pub bundle_meta: BundleMeta,
    pub entries: Vec<ChunkEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdirFormat {
    Cbor,
    Json,
}

impl CentralDirectory {
    pub fn encode(&self, format: CdirFormat) -> Result<Vec<u8>> {
        match format {
            CdirFormat::Cbor => {
                let mut buf = Vec::with_capacity(4096);
                ciborium::into_writer(self, &mut buf).map_err(|e| Error::CborEncode(e.to_string()))?;
                Ok(buf)
            }
            CdirFormat::Json => serde_json::to_vec(self).map_err(|e| Error::CborEncode(e.to_string())),
        }
    }

    pub fn decode(bytes: &[u8], format: CdirFormat) -> Result<Self> {
        match format {
            CdirFormat::Cbor => {
                ciborium::from_reader(bytes).map_err(|e| Error::CborDecode(e.to_string()))
            }
            CdirFormat::Json => serde_json::from_slice(bytes).map_err(|e| Error::CborDecode(e.to_string())),
        }
    }

    /// Find the first alive entry with the given name.
    pub fn find_alive(&self, name: &str) -> Option<&ChunkEntry> {
        self.entries.iter().rev().find(|e| e.is_alive() && e.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdir_round_trip_cbor() {
        let cdir = CentralDirectory {
            bundle_meta: BundleMeta {
                id: "test-model".into(),
                version: "1.0.0".into(),
                arch: "xlm-roberta".into(),
                purpose: "embed".into(),
                ..Default::default()
            },
            entries: vec![ChunkEntry {
                id: 1,
                kind: CHUNK_TYPE_TENSORS,
                name: "tensors:main".into(),
                offset: 64,
                header_len: 64,
                payload_off: 128,
                payload_len: 4096,
                raw_len: 4096,
                flags: 0,
                crc32c: 0xdead_beef,
                status: CHUNK_STATUS_ALIVE,
                sha256: None,
                blake3: None,
                tag: None,
                metadata: BTreeMap::new(),
            }],
        };
        let bytes = cdir.encode(CdirFormat::Cbor).unwrap();
        let decoded = CentralDirectory::decode(&bytes, CdirFormat::Cbor).unwrap();
        assert_eq!(decoded.bundle_meta.id, "test-model");
        assert_eq!(decoded.entries.len(), 1);
        assert!(decoded.entries[0].is_alive());
    }

    #[test]
    fn find_alive_skips_tombstones() {
        let cdir = CentralDirectory {
            bundle_meta: BundleMeta::default(),
            entries: vec![
                ChunkEntry {
                    id: 1,
                    kind: CHUNK_TYPE_FILE,
                    name: "README.md".into(),
                    offset: 0,
                    header_len: 0,
                    payload_off: 0,
                    payload_len: 0,
                    raw_len: 0,
                    flags: 0,
                    crc32c: 0,
                    status: CHUNK_STATUS_TOMBSTONE,
                    sha256: None,
                    blake3: None,
                    tag: None,
                    metadata: BTreeMap::new(),
                },
                ChunkEntry {
                    id: 2,
                    kind: CHUNK_TYPE_FILE,
                    name: "README.md".into(),
                    offset: 100,
                    header_len: 64,
                    payload_off: 164,
                    payload_len: 10,
                    raw_len: 10,
                    flags: 0,
                    crc32c: 0,
                    status: CHUNK_STATUS_ALIVE,
                    sha256: None,
                    blake3: None,
                    tag: None,
                    metadata: BTreeMap::new(),
                },
            ],
        };
        let e = cdir.find_alive("README.md").unwrap();
        assert_eq!(e.id, 2);
    }
}
