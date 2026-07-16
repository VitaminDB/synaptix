use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

pub fn walk_folder(root: impl AsRef<Path>, extensions: &[&str]) -> impl Iterator<Item = PathBuf> {
    let exts: Vec<String> = extensions.iter().map(|e| e.to_lowercase()).collect();
    WalkBuilder::new(root)
        .build()
        .filter_map(move |entry| {
            let entry = entry.ok()?;
            let path = entry.into_path();
            if !path.is_file() {
                return None;
            }
            if exts.is_empty() {
                return Some(path);
            }
            let ext = path.extension()?.to_str()?.to_lowercase();
            if exts.iter().any(|e| e == &ext) {
                Some(path)
            } else {
                None
            }
        })
}

pub fn walk_folder_sorted(root: impl AsRef<Path>, extensions: &[&str]) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = walk_folder(root, extensions).collect();
    paths.sort();
    paths
}
