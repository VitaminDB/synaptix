//! Fixed-size on-disk records: file header and footer.
//!
//! Both are bit-exact little-endian and never grow within a major version —
//! `ver_minor` bumps may *use* the reserved space, never resize it.

use std::io::{self, Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{Error, Result};

pub const FILE_MAGIC: &[u8; 8] = b"SYNBNDL\0";
pub const TRAILER_MAGIC: &[u8; 8] = b"SYNEND\0\0";

pub const FILE_HEADER_SIZE: usize = 64;
pub const FOOTER_SIZE: usize = 40;

/// Chunks and the central directory all start on a 64-byte boundary so that
/// tensor payloads (which live inside the Tensors chunk) inherit the alignment
/// safetensors expects.
pub const ALIGNMENT: u64 = 64;

/// Bundle-level flag bits (in `FileHeader.flags`).
pub const FLAG_HAS_SHA256_MANIFEST: u32 = 1 << 0;
pub const FLAG_ENCRYPTED: u32 = 1 << 1;

/// On-disk file header at offset 0. Always exactly 64 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    pub ver_major: u16,
    pub ver_minor: u16,
    pub flags: u32,
    pub created_at: u64,
}

impl FileHeader {
    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        w.write_all(FILE_MAGIC)?;
        w.write_u16::<LittleEndian>(self.ver_major)?;
        w.write_u16::<LittleEndian>(self.ver_minor)?;
        w.write_u32::<LittleEndian>(self.flags)?;
        w.write_u64::<LittleEndian>(self.created_at)?;
        w.write_all(&[0u8; 40])?;
        Ok(())
    }

    pub fn read_from<R: Read>(mut r: R) -> Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != FILE_MAGIC {
            return Err(Error::BadMagic { where_: "file header" });
        }
        let ver_major = r.read_u16::<LittleEndian>()?;
        let ver_minor = r.read_u16::<LittleEndian>()?;
        let flags = r.read_u32::<LittleEndian>()?;
        let created_at = r.read_u64::<LittleEndian>()?;
        let mut reserved = [0u8; 40];
        r.read_exact(&mut reserved)?;
        Ok(Self { ver_major, ver_minor, flags, created_at })
    }
}

/// Cdir serialisation format, stored in the footer's `cdir_format` field.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdirOnDiskFormat {
    Cbor = 0,
    Json = 1,
}

impl CdirOnDiskFormat {
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0 => Some(Self::Cbor),
            1 => Some(Self::Json),
            _ => None,
        }
    }
}

/// On-disk footer at `EOF - 40`. Always exactly 40 bytes.
#[derive(Debug, Clone, Copy)]
pub struct Footer {
    pub cdir_offset: u64,
    pub cdir_len: u64,
    pub cdir_crc32c: u32,
    pub cdir_format: u16,
    pub ver_major: u16,
    pub bundle_crc32c: u32,
}

impl Footer {
    /// Bytes covered by `bundle_crc32c` — everything before the crc field itself.
    pub const CRC_REGION_LEN: usize = 8 + 8 + 4 + 2 + 2;

    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        let mut region = [0u8; Self::CRC_REGION_LEN];
        {
            let mut cur = &mut region[..];
            cur.write_u64::<LittleEndian>(self.cdir_offset)?;
            cur.write_u64::<LittleEndian>(self.cdir_len)?;
            cur.write_u32::<LittleEndian>(self.cdir_crc32c)?;
            cur.write_u16::<LittleEndian>(self.cdir_format)?;
            cur.write_u16::<LittleEndian>(self.ver_major)?;
        }
        w.write_all(&region)?;
        let crc = crc32c::crc32c(&region);
        w.write_u32::<LittleEndian>(crc)?;
        w.write_all(&[0u8; 4])?; // reserved
        w.write_all(TRAILER_MAGIC)?;
        Ok(())
    }

    pub fn read_from(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != FOOTER_SIZE {
            return Err(Error::FileTooSmall(bytes.len() as u64));
        }
        let region = &bytes[..Self::CRC_REGION_LEN];
        let stored_crc = u32::from_le_bytes(
            bytes[Self::CRC_REGION_LEN..Self::CRC_REGION_LEN + 4]
                .try_into()
                .unwrap(),
        );
        let trailer = &bytes[FOOTER_SIZE - 8..];
        if trailer != TRAILER_MAGIC {
            return Err(Error::BadMagic { where_: "footer trailer" });
        }
        let computed = crc32c::crc32c(region);
        if computed != stored_crc {
            return Err(Error::FooterCrcMismatch { expected: stored_crc, got: computed });
        }
        let mut r = &region[..];
        let cdir_offset = r.read_u64::<LittleEndian>()?;
        let cdir_len = r.read_u64::<LittleEndian>()?;
        let cdir_crc32c = r.read_u32::<LittleEndian>()?;
        let cdir_format = r.read_u16::<LittleEndian>()?;
        let ver_major = r.read_u16::<LittleEndian>()?;
        Ok(Self {
            cdir_offset,
            cdir_len,
            cdir_crc32c,
            cdir_format,
            ver_major,
            bundle_crc32c: stored_crc,
        })
    }
}

/// Round `n` up to the next multiple of `ALIGNMENT`.
pub const fn align_up(n: u64) -> u64 {
    let rem = n % ALIGNMENT;
    if rem == 0 { n } else { n + (ALIGNMENT - rem) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_header_round_trip() {
        let h = FileHeader {
            ver_major: 1,
            ver_minor: 0,
            flags: 0,
            created_at: 1_700_000_000,
        };
        let mut buf = Vec::new();
        h.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), FILE_HEADER_SIZE);
        let parsed = FileHeader::read_from(&buf[..]).unwrap();
        assert_eq!(parsed.ver_major, 1);
        assert_eq!(parsed.created_at, 1_700_000_000);
    }

    #[test]
    fn footer_round_trip_and_crc() {
        let f = Footer {
            cdir_offset: 1024,
            cdir_len: 512,
            cdir_crc32c: 0xdeadbeef,
            cdir_format: 0,
            ver_major: 1,
            bundle_crc32c: 0,
        };
        let mut buf = Vec::new();
        f.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), FOOTER_SIZE);
        let parsed = Footer::read_from(&buf).unwrap();
        assert_eq!(parsed.cdir_offset, 1024);
        assert_eq!(parsed.cdir_len, 512);
        assert_eq!(parsed.cdir_format, 0);
    }

    #[test]
    fn footer_rejects_corruption() {
        let f = Footer {
            cdir_offset: 1024,
            cdir_len: 512,
            cdir_crc32c: 0xdeadbeef,
            cdir_format: 0,
            ver_major: 1,
            bundle_crc32c: 0,
        };
        let mut buf = Vec::new();
        f.write_to(&mut buf).unwrap();
        buf[0] ^= 0xff;
        let err = Footer::read_from(&buf).unwrap_err();
        assert!(matches!(err, Error::FooterCrcMismatch { .. }));
    }

    #[test]
    fn align_up_works() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 64);
        assert_eq!(align_up(64), 64);
        assert_eq!(align_up(65), 128);
    }
}
