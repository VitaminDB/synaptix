//! POSIX-style path normalisation for in-bundle paths.
//!
//! Bundle paths look like `examples/voice_ref/sample.wav` — slash-separated,
//! no leading slash, no `..`, no `.`, no empty segments, no NUL, ≤ 1024 bytes.

use crate::error::{Error, Result};

pub const MAX_PATH_LEN: usize = 1024;

/// Validate and canonicalise an in-bundle path:
///
/// - converts `\\` → `/` (Windows ergonomics);
/// - strips leading `./`;
/// - rejects `..`, empty segments, `//`, NUL, leading `/`, trailing `/`;
/// - rejects paths longer than `MAX_PATH_LEN` bytes (UTF-8);
/// - returns the canonical form as `String`.
pub fn normalize(input: &str) -> Result<String> {
    let mut s: String = input.replace('\\', "/");
    while let Some(rest) = s.strip_prefix("./") {
        s = rest.to_string();
    }
    if s.is_empty() {
        return Err(Error::InvalidPath { path: input.to_string(), reason: "empty path" });
    }
    if s.starts_with('/') {
        return Err(Error::InvalidPath {
            path: input.to_string(),
            reason: "leading slash",
        });
    }
    if s.ends_with('/') {
        return Err(Error::InvalidPath {
            path: input.to_string(),
            reason: "trailing slash (paths in a bundle are files, not directories)",
        });
    }
    if s.len() > MAX_PATH_LEN {
        return Err(Error::InvalidPath { path: input.to_string(), reason: "too long (> 1024 bytes)" });
    }
    if s.as_bytes().contains(&0) {
        return Err(Error::InvalidPath { path: input.to_string(), reason: "contains NUL byte" });
    }
    for seg in s.split('/') {
        if seg.is_empty() {
            return Err(Error::InvalidPath {
                path: input.to_string(),
                reason: "empty path segment (// in path)",
            });
        }
        if seg == ".." {
            return Err(Error::InvalidPath {
                path: input.to_string(),
                reason: "'..' segment not allowed",
            });
        }
        if seg == "." {
            return Err(Error::InvalidPath {
                path: input.to_string(),
                reason: "'.' segment not allowed",
            });
        }
    }
    Ok(s)
}

/// Check whether `path` lies under `prefix` (treating `prefix` as a directory
/// path). `prefix == ""` matches everything. `prefix` must itself end *without*
/// a trailing slash; we add one internally for the comparison.
pub fn is_under(path: &str, prefix: &str) -> bool {
    if prefix.is_empty() {
        return true;
    }
    let with_slash = if prefix.ends_with('/') {
        prefix.to_string()
    } else {
        format!("{prefix}/")
    };
    path.starts_with(&with_slash)
}

/// Immediate child segment of `path` under `prefix`, if any.
/// Returns `(segment, is_dir)` where `is_dir = true` iff `path` has more
/// segments after `segment`.
pub fn shallow_child<'a>(path: &'a str, prefix: &str) -> Option<(&'a str, bool)> {
    let stripped = if prefix.is_empty() {
        path
    } else {
        let with_slash = if prefix.ends_with('/') {
            prefix.to_string()
        } else {
            format!("{prefix}/")
        };
        path.strip_prefix(&with_slash)?
    };
    if stripped.is_empty() {
        return None;
    }
    if let Some(idx) = stripped.find('/') {
        Some((&stripped[..idx], true))
    } else {
        Some((stripped, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_basic() {
        assert_eq!(normalize("foo/bar.txt").unwrap(), "foo/bar.txt");
        assert_eq!(normalize("./foo/bar.txt").unwrap(), "foo/bar.txt");
        assert_eq!(normalize("foo\\bar.txt").unwrap(), "foo/bar.txt");
    }

    #[test]
    fn normalize_rejects() {
        assert!(normalize("").is_err());
        assert!(normalize("/abs").is_err());
        assert!(normalize("trail/").is_err());
        assert!(normalize("a//b").is_err());
        assert!(normalize("../escape").is_err());
        assert!(normalize("a/./b").is_err());
        assert!(normalize("a/../b").is_err());
        assert!(normalize("with\0nul").is_err());
    }

    #[test]
    fn shallow_child_works() {
        assert_eq!(shallow_child("a/b/c", ""), Some(("a", true)));
        assert_eq!(shallow_child("a/b/c", "a"), Some(("b", true)));
        assert_eq!(shallow_child("a/b/c", "a/"), Some(("b", true)));
        assert_eq!(shallow_child("a/b", "a"), Some(("b", false)));
        assert_eq!(shallow_child("a", ""), Some(("a", false)));
        assert_eq!(shallow_child("a/b", "x"), None);
    }
}
