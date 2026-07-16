pub struct ParquetStub;

impl ParquetStub {
    pub fn not_available() -> &'static str {
        "parquet feature not yet implemented; add parquet crate to deps"
    }
}

pub fn open_parquet(_path: impl AsRef<std::path::Path>) -> crate::error::Result<ParquetStub> {
    Err(crate::error::IoError::Data("parquet crate not in deps".into()))
}
