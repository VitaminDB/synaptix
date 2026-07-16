use std::sync::atomic::{AtomicUsize, Ordering};

static KV_PREALLOC_SEQ_LEN: AtomicUsize = AtomicUsize::new(0);

pub fn set_kv_prealloc_seq_len(n: usize) {
    KV_PREALLOC_SEQ_LEN.store(n, Ordering::Relaxed);
}

pub fn kv_prealloc_seq_len() -> Option<usize> {
    let v = KV_PREALLOC_SEQ_LEN.load(Ordering::Relaxed);
    if v == 0 { None } else { Some(v) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_none() {
        set_kv_prealloc_seq_len(0);
        assert_eq!(kv_prealloc_seq_len(), None);
        set_kv_prealloc_seq_len(4096);
        assert_eq!(kv_prealloc_seq_len(), Some(4096));
        set_kv_prealloc_seq_len(0);
    }
}
