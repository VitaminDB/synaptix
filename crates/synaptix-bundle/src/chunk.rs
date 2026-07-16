//! Chunk on-disk encoding.
//!
//! A chunk = self-describing header + payload + trailing alignment pad.
//! Header layout (variable due to name, but padded to 64-byte alignment):
//!
//! ```text
//! [ chunk_magic    : 4B  = b"SYNK"                          ]
//! [ chunk_id       : u64 LE                                 ]
//! [ chunk_type     : u16 LE                                 ]
//! [ flags          : u16 LE                                 ]
//! [ payload_len    : u64 LE                                 ]
//! [ raw_len        : u64 LE                                 ]
//! [ payload_crc32c : u32 LE                                 ]
//! [ name_len       : u16 LE                                 ]
//! [ reserved       : 6B  zero                               ]
//! [ name           : <name_len> UTF-8                       ]
//! [ pad to 64-byte align                                    ]
//! ```
//!
//! Then payload, then a final pad to the next 64-byte boundary so the next
//! chunk header is aligned.

use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{Error, Result};
use crate::header::align_up;

pub const CHUNK_MAGIC: &[u8; 4] = b"SYNK";

/// Fixed prefix size: 4 + 8 + 2 + 2 + 8 + 8 + 4 + 2 + 6 = 44 bytes
pub const CHUNK_HEADER_FIXED: usize = 44;

/// Chunk flag bits.
/// Bit 0 reserved for future per-chunk transforms (was: zstd compression,
/// dropped from the roadmap — weights aren't compressible and aux files are
/// negligible, see plan).
pub const CHUNK_FLAG_RESERVED_0: u16 = 1 << 0;
pub const CHUNK_FLAG_ENCRYPTED: u16 = 1 << 1;

/// Compute the on-disk length of a chunk header given its name length.
/// Includes the 64-byte padding so the payload starts at an aligned offset.
pub fn header_len(name_len: usize) -> u64 {
    align_up(CHUNK_HEADER_FIXED as u64 + name_len as u64)
}

/// Distance from a chunk's start offset to the start of the next chunk:
/// header (aligned) + payload + trailing align-up.
pub fn chunk_total_size(name_len: usize, payload_len: u64) -> u64 {
    let hlen = header_len(name_len);
    align_up(hlen + payload_len)
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkHeader {
    pub id: u64,
    pub kind: u16,
    pub flags: u16,
    pub payload_len: u64,
    pub raw_len: u64,
    pub payload_crc32c: u32,
    pub name_len: u16,
}

impl ChunkHeader {
    pub fn write<W: Write>(&self, mut w: W, name: &[u8]) -> io::Result<()> {
        debug_assert_eq!(name.len(), self.name_len as usize);
        w.write_all(CHUNK_MAGIC)?;
        w.write_u64::<LittleEndian>(self.id)?;
        w.write_u16::<LittleEndian>(self.kind)?;
        w.write_u16::<LittleEndian>(self.flags)?;
        w.write_u64::<LittleEndian>(self.payload_len)?;
        w.write_u64::<LittleEndian>(self.raw_len)?;
        w.write_u32::<LittleEndian>(self.payload_crc32c)?;
        w.write_u16::<LittleEndian>(self.name_len)?;
        w.write_all(&[0u8; 6])?; // reserved
        w.write_all(name)?;
        let written = CHUNK_HEADER_FIXED as u64 + name.len() as u64;
        let pad = align_up(written) - written;
        if pad > 0 {
            let zeros = vec![0u8; pad as usize];
            w.write_all(&zeros)?;
        }
        Ok(())
    }

    pub fn read<R: Read>(mut r: R) -> Result<(Self, Vec<u8>)> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != CHUNK_MAGIC {
            return Err(Error::ChunkBadMagic { id: 0, got: magic });
        }
        let id = r.read_u64::<LittleEndian>()?;
        let kind = r.read_u16::<LittleEndian>()?;
        let flags = r.read_u16::<LittleEndian>()?;
        let payload_len = r.read_u64::<LittleEndian>()?;
        let raw_len = r.read_u64::<LittleEndian>()?;
        let payload_crc32c = r.read_u32::<LittleEndian>()?;
        let name_len = r.read_u16::<LittleEndian>()?;
        let mut reserved = [0u8; 6];
        r.read_exact(&mut reserved)?;
        let mut name = vec![0u8; name_len as usize];
        r.read_exact(&mut name)?;
        let consumed = CHUNK_HEADER_FIXED as u64 + name_len as u64;
        let pad = align_up(consumed) - consumed;
        if pad > 0 {
            let mut skip = vec![0u8; pad as usize];
            r.read_exact(&mut skip)?;
        }
        Ok((
            Self {
                id,
                kind,
                flags,
                payload_len,
                raw_len,
                payload_crc32c,
                name_len,
            },
            name,
        ))
    }
}

/// Verify CRC32C of a slice matches `expected`. Returns the expected/got pair
/// inside a `ChunkCrcMismatch` if it doesn't.
pub fn verify_payload_crc(id: u64, payload: &[u8], expected: u32) -> Result<()> {
    let got = crc32c::crc32c(payload);
    if got != expected {
        return Err(Error::ChunkCrcMismatch { id, expected, got });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_header_round_trip() {
        let payload = b"hello world";
        let crc = crc32c::crc32c(payload);
        let name = b"config.json";
        let h = ChunkHeader {
            id: 7,
            kind: 2,
            flags: 0,
            payload_len: payload.len() as u64,
            raw_len: payload.len() as u64,
            payload_crc32c: crc,
            name_len: name.len() as u16,
        };
        let mut buf = Vec::new();
        h.write(&mut buf, name).unwrap();
        assert_eq!(buf.len() as u64, header_len(name.len()));
        assert!(buf.len() % 64 == 0);
        let (parsed, parsed_name) = ChunkHeader::read(&buf[..]).unwrap();
        assert_eq!(parsed.id, 7);
        assert_eq!(parsed.kind, 2);
        assert_eq!(parsed.payload_len, payload.len() as u64);
        assert_eq!(parsed_name, name);
        verify_payload_crc(parsed.id, payload, parsed.payload_crc32c).unwrap();
    }

    #[test]
    fn crc_mismatch_detected() {
        let err = verify_payload_crc(1, b"abc", 0xdeadbeef).unwrap_err();
        assert!(matches!(err, Error::ChunkCrcMismatch { .. }));
    }

    #[test]
    fn chunk_total_size_aligned() {
        assert!(chunk_total_size(11, 100) % 64 == 0);
        assert!(chunk_total_size(0, 0) % 64 == 0);
        assert!(chunk_total_size(1024, 1) % 64 == 0);
    }
}
