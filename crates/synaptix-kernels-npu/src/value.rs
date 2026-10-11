use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::storage::{NpuBuf, Storage};
use synaptix_npu::NpuBuffer;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Origin {
    Param,
    Host,
    Graph,
}

pub struct Ready {
    pub buf: Arc<NpuBuffer>,
    pub origin: Origin,
    pub uses: u32,
    pub fingerprint: Option<u64>,
}

impl Ready {
    pub fn fingerprint(&mut self, bytes: usize) -> u64 {
        if let Some(f) = self.fingerprint {
            return f;
        }
        self.buf.cpu_begin();
        let f = fingerprint(&self.buf.as_slice()[..bytes]);
        self.buf.cpu_end();
        self.fingerprint = Some(f);
        f
    }
}

pub enum State {
    Unset,
    Pending(u64),
    Ready(Ready),
}

pub struct Value {
    pub id: u64,
    pub bytes: usize,
    pub state: Mutex<State>,
}

pub type ValueRef = Arc<Value>;

static NEXT: AtomicU64 = AtomicU64::new(1);

impl Value {
    pub fn new(bytes: usize, state: State) -> ValueRef {
        Arc::new(Value { id: NEXT.fetch_add(1, Ordering::Relaxed), bytes, state: Mutex::new(state) })
    }
}

pub fn storage(v: ValueRef, ordinal: usize) -> Storage {
    Storage::Npu(NpuBuf { ordinal, byte_len: v.bytes, inner: Box::new(v) })
}

pub fn value_of(s: &Storage) -> Result<ValueRef> {
    s.as_npu()
        .and_then(|b| b.inner.downcast_ref::<ValueRef>())
        .cloned()
        .ok_or(SynaptixError::Unsupported("ожидалось хранилище NPU"))
}

pub fn fingerprint(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ b.len() as u64;
    let mut eat = |c: &[u8]| {
        for w in c.chunks(8) {
            let mut x = [0u8; 8];
            x[..w.len()].copy_from_slice(w);
            h = (h ^ u64::from_le_bytes(x)).wrapping_mul(0x100_0000_01b3);
        }
    };
    if b.len() <= 1 << 20 {
        eat(b);
    } else {
        eat(&b[..64 << 10]);
        let step = b.len() / 256;
        for i in 0..256 {
            let s = i * step;
            eat(&b[s..(s + 512).min(b.len())]);
        }
        eat(&b[b.len() - (64 << 10)..]);
    }
    h
}
