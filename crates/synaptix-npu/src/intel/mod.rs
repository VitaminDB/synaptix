//! Intel NPU (Core Ultra, «AI Boost») через OpenVINO: граф synaptix переводится в
//! OpenVINO IR v11 (XML + веса в памяти), собирается плагином NPU и исполняется
//! с буферами synaptix как входами и выходами (тензоры поверх памяти хоста).
//!
//! OpenVINO подгружается во время работы (`libopenvino_c.so`, C API) — без
//! зависимости при сборке; нет библиотеки, плагина NPU или `/dev/accel` —
//! драйвер просто не находится. Нужны пакеты `openvino`,
//! `openvino-intel-npu-plugin`, `intel-npu-driver`, `intel-npu-compiler`.
//!
//! Собранные графы кэширует сам OpenVINO (`CACHE_DIR`, по умолчанию
//! `~/.cache/synaptix/npu-intel`): повторная сборка того же графа — чтение блоба.

mod ir;

use crate::buffer::NpuBuffer;
use crate::graph::{DType, Graph, TensorId, TensorKind};
use crate::models;
use crate::{Binding, CompiledGraph, IoTensor, Npu, NpuError, NpuInfo, Result};
use std::ffi::{c_char, c_void, CStr, CString};
use std::time::{Duration, Instant};

type Status = i32;
type P = *mut c_void;

/// Функции OpenVINO C API (сигнатуры — `openvino/c/*.h`).
#[allow(clippy::type_complexity)]
struct Ov {
    _lib: *mut c_void,
    core_create: unsafe extern "C" fn(*mut P) -> Status,
    core_free: unsafe extern "C" fn(P),
    read_model_mem: unsafe extern "C" fn(P, *const c_char, usize, P, *mut P) -> Status,
    compile: unsafe extern "C" fn(P, P, *const c_char, usize, *mut P, ...) -> Status,
    model_free: unsafe extern "C" fn(P),
    compiled_free: unsafe extern "C" fn(P),
    create_request: unsafe extern "C" fn(P, *mut P) -> Status,
    request_free: unsafe extern "C" fn(P),
    set_input: unsafe extern "C" fn(P, usize, P) -> Status,
    set_output: unsafe extern "C" fn(P, usize, P) -> Status,
    infer: unsafe extern "C" fn(P) -> Status,
    shape_create: unsafe extern "C" fn(i64, *const i64, *mut Shape) -> Status,
    shape_free: unsafe extern "C" fn(*mut Shape) -> Status,
    tensor_from_ptr: unsafe extern "C" fn(u32, Shape, *mut c_void, *mut P) -> Status,
    tensor_free: unsafe extern "C" fn(P),
    last_err: unsafe extern "C" fn() -> *const c_char,
    get_versions: unsafe extern "C" fn(P, *const c_char, *mut CoreVersionList) -> Status,
    versions_free: unsafe extern "C" fn(*mut CoreVersionList),
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Shape {
    rank: i64,
    dims: *mut i64,
}

#[repr(C)]
struct Version {
    build_number: *const c_char,
    description: *const c_char,
}

#[repr(C)]
struct CoreVersion {
    device_name: *const c_char,
    version: Version,
}

#[repr(C)]
struct CoreVersionList {
    versions: *mut CoreVersion,
    size: usize,
}

unsafe impl Send for Ov {}
unsafe impl Sync for Ov {}

/// Коды `ov_element_type_e` OpenVINO 2026 (с U2/U3/U6 в перечислении).
fn ov_type(d: DType) -> u32 {
    match d {
        DType::F16 => 3,
        DType::F32 => 4,
        DType::I4 => 6,
        DType::I8 => 7,
        DType::I32 => 9,
        DType::U8 => 16,
        DType::U16 => 17,
        DType::U32 => 18,
    }
}

impl Ov {
    fn load() -> Result<Ov> {
        let lib = ["libopenvino_c.so", "libopenvino_c.so.2641"]
            .iter()
            .find_map(|n| {
                let c = CString::new(*n).unwrap();
                let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
                (!h.is_null()).then_some(h)
            })
            .ok_or_else(|| NpuError::Unavailable("нет libopenvino_c.so (пакет openvino)".into()))?;
        let sym = |name: &str| -> Result<*mut c_void> {
            let c = CString::new(name).unwrap();
            let p = unsafe { libc::dlsym(lib, c.as_ptr()) };
            if p.is_null() {
                Err(NpuError::Unavailable(format!("OpenVINO без {name}")))
            } else {
                Ok(p)
            }
        };
        macro_rules! f {
            ($n:literal) => {
                unsafe { std::mem::transmute::<*mut c_void, _>(sym($n)?) }
            };
        }
        Ok(Ov {
            _lib: lib,
            core_create: f!("ov_core_create"),
            core_free: f!("ov_core_free"),
            read_model_mem: f!("ov_core_read_model_from_memory_buffer"),
            compile: f!("ov_core_compile_model"),
            model_free: f!("ov_model_free"),
            compiled_free: f!("ov_compiled_model_free"),
            create_request: f!("ov_compiled_model_create_infer_request"),
            request_free: f!("ov_infer_request_free"),
            set_input: f!("ov_infer_request_set_input_tensor_by_index"),
            set_output: f!("ov_infer_request_set_output_tensor_by_index"),
            infer: f!("ov_infer_request_infer"),
            shape_create: f!("ov_shape_create"),
            shape_free: f!("ov_shape_free"),
            tensor_from_ptr: f!("ov_tensor_create_from_host_ptr"),
            tensor_free: f!("ov_tensor_free"),
            last_err: f!("ov_get_last_err_msg"),
            get_versions: f!("ov_core_get_versions_by_device_name"),
            versions_free: f!("ov_core_versions_free"),
        })
    }

    fn check(&self, st: Status, what: &str) -> Result<()> {
        if st == 0 {
            return Ok(());
        }
        let msg = unsafe {
            let p = (self.last_err)();
            if p.is_null() {
                String::new()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(NpuError::Driver { driver: "openvino", msg: format!("{what}: код {st}: {}", msg.trim()) })
    }

    /// Тензор OpenVINO поверх памяти хоста (без копии).
    fn host_tensor(&self, dt: DType, shape: &[u32], ptr: *mut u8) -> Result<OwnedTensor<'_>> {
        let dims: Vec<i64> = if shape.is_empty() { vec![1] } else { shape.iter().map(|&d| d as i64).collect() };
        let mut s = Shape { rank: 0, dims: std::ptr::null_mut() };
        self.check(unsafe { (self.shape_create)(dims.len() as i64, dims.as_ptr(), &mut s) }, "ov_shape_create")?;
        let mut t: P = std::ptr::null_mut();
        let st = unsafe { (self.tensor_from_ptr)(ov_type(dt), s, ptr as *mut c_void, &mut t) };
        unsafe { (self.shape_free)(&mut s) };
        self.check(st, "ov_tensor_create_from_host_ptr")?;
        Ok(OwnedTensor { ov: self, t })
    }
}

struct OwnedTensor<'a> {
    ov: &'a Ov,
    t: P,
}

impl Drop for OwnedTensor<'_> {
    fn drop(&mut self) {
        unsafe { (self.ov.tensor_free)(self.t) }
    }
}

pub struct IntelNpu {
    ov: &'static Ov,
    core: P,
    info: NpuInfo,
    cache_dir: CString,
}

unsafe impl Send for IntelNpu {}
unsafe impl Sync for IntelNpu {}

fn cache_dir() -> String {
    if let Ok(d) = std::env::var("SYNAPTIX_NPU_CACHE") {
        return d;
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.cache")))
        .unwrap_or_else(|| "/tmp".into());
    format!("{base}/synaptix/npu-intel")
}

impl IntelNpu {
    pub fn connect() -> Result<Self> {
        if std::env::var("SYNAPTIX_NPU_INTEL").as_deref() == Ok("0") {
            return Err(NpuError::Unavailable("Intel NPU выключен (SYNAPTIX_NPU_INTEL=0)".into()));
        }
        if !std::path::Path::new("/dev/accel/accel0").exists() {
            return Err(NpuError::Unavailable("нет /dev/accel/accel0".into()));
        }
        static OV: std::sync::OnceLock<std::result::Result<Ov, String>> = std::sync::OnceLock::new();
        let ov = OV.get_or_init(|| Ov::load().map_err(|e| e.to_string())).as_ref().map_err(|e| NpuError::Unavailable(e.clone()))?;
        let mut core: P = std::ptr::null_mut();
        ov.check(unsafe { (ov.core_create)(&mut core) }, "ov_core_create")?;
        // Есть ли плагин NPU: версия устройства «NPU».
        let dev = CString::new("NPU").unwrap();
        let mut list = CoreVersionList { versions: std::ptr::null_mut(), size: 0 };
        if let Err(e) = ov.check(unsafe { (ov.get_versions)(core, dev.as_ptr(), &mut list) }, "плагин NPU") {
            unsafe { (ov.core_free)(core) };
            return Err(NpuError::Unavailable(e.to_string()));
        }
        let runtime = unsafe {
            let b = if list.size > 0 && !list.versions.is_null() {
                CStr::from_ptr((*list.versions).version.build_number).to_string_lossy().into_owned()
            } else {
                String::new()
            };
            (ov.versions_free)(&mut list);
            format!("OpenVINO {b}")
        };
        let model = models::MODELS.iter().find(|m| m.id == "intel-npu").expect("модель intel-npu");
        let dir = cache_dir();
        let _ = std::fs::create_dir_all(&dir);
        Ok(IntelNpu { ov, core, info: NpuInfo { model, soc: 0, vtcm_mb: 0, runtime }, cache_dir: CString::new(dir).unwrap() })
    }
}

impl Drop for IntelNpu {
    fn drop(&mut self) {
        unsafe { (self.ov.core_free)(self.core) }
    }
}

struct IntelGraph {
    ov: &'static Ov,
    compiled: P,
    request: parking_lot::Mutex<P>,
    inputs: Vec<IoTensor>,
    outputs: Vec<IoTensor>,
    build: Duration,
}

unsafe impl Send for IntelGraph {}
unsafe impl Sync for IntelGraph {}

impl Npu for IntelNpu {
    fn info(&self) -> &NpuInfo {
        &self.info
    }

    fn alloc(&self, bytes: usize) -> Result<NpuBuffer> {
        NpuBuffer::new(bytes)
    }

    fn compile(&self, _key: &str, g: &Graph) -> Result<Box<dyn CompiledGraph>> {
        let t0 = Instant::now();
        let lowered = ir::lower(g)?;
        if let Ok(dir) = std::env::var("SYNAPTIX_NPU_DUMP") {
            let h = g.structure_hash();
            let _ = std::fs::write(format!("{dir}/{h:016x}.xml"), &lowered.xml);
            let _ = std::fs::write(format!("{dir}/{h:016x}.bin"), &lowered.bin);
        }
        let ov = self.ov;
        let mut bin = lowered.bin;
        let wt = ov.host_tensor(DType::U8, &[bin.len().max(1) as u32], bin.as_mut_ptr())?;
        let mut model: P = std::ptr::null_mut();
        let xml = lowered.xml.as_bytes();
        ov.check(unsafe { (ov.read_model_mem)(self.core, xml.as_ptr() as *const c_char, xml.len(), wt.t, &mut model) }, "чтение IR")?;
        let dev = CString::new("NPU").unwrap();
        let (k_cache, k_hint) = (CString::new("CACHE_DIR").unwrap(), CString::new("PERFORMANCE_HINT").unwrap());
        let v_hint = CString::new("LATENCY").unwrap();
        let mut compiled: P = std::ptr::null_mut();
        let st = unsafe {
            (ov.compile)(self.core, model, dev.as_ptr(), 4, &mut compiled, k_cache.as_ptr(), self.cache_dir.as_ptr(), k_hint.as_ptr(), v_hint.as_ptr())
        };
        unsafe { (ov.model_free)(model) };
        drop(wt);
        drop(bin);
        ov.check(st, "сборка графа на NPU")?;
        let mut req: P = std::ptr::null_mut();
        if let Err(e) = ov.check(unsafe { (ov.create_request)(compiled, &mut req) }, "запрос исполнения") {
            unsafe { (ov.compiled_free)(compiled) };
            return Err(e);
        }
        Ok(Box::new(IntelGraph {
            ov,
            compiled,
            request: parking_lot::Mutex::new(req),
            inputs: lowered.inputs,
            outputs: lowered.outputs,
            build: t0.elapsed(),
        }))
    }
}

impl CompiledGraph for IntelGraph {
    fn inputs(&self) -> &[IoTensor] {
        &self.inputs
    }

    fn outputs(&self) -> &[IoTensor] {
        &self.outputs
    }

    fn from_cache(&self) -> bool {
        false
    }

    fn build_time(&self) -> Duration {
        self.build
    }

    fn run(&self, bindings: &[Binding<'_>]) -> Result<Duration> {
        let find = |id: TensorId| -> Result<&Binding<'_>> {
            bindings.iter().find(|b| b.tensor == id).ok_or_else(|| NpuError::Graph(format!("тензор {} без буфера", id.0)))
        };
        let ov = self.ov;
        let req = self.request.lock();
        let mut keep = Vec::with_capacity(self.inputs.len() + self.outputs.len());
        for (set, list, what) in [(ov.set_input, &self.inputs, "вход"), (ov.set_output, &self.outputs, "выход")] {
            for (i, io) in list.iter().enumerate() {
                let b = find(io.id)?;
                if b.offset + io.bytes() > b.buffer.len() {
                    return Err(NpuError::Graph(format!("{what} {}: буфер меньше тензора", io.id.0)));
                }
                let ptr = unsafe { b.buffer.as_slice().as_ptr().add(b.offset) as *mut u8 };
                let t = ov.host_tensor(io.dtype, &io.shape, ptr)?;
                ov.check(unsafe { set(*req, i, t.t) }, what)?;
                keep.push(t);
            }
        }
        let t0 = Instant::now();
        ov.check(unsafe { (ov.infer)(*req) }, "исполнение")?;
        Ok(t0.elapsed())
    }
}

impl Drop for IntelGraph {
    fn drop(&mut self) {
        unsafe {
            (self.ov.request_free)(*self.request.lock());
            (self.ov.compiled_free)(self.compiled);
        }
    }
}

/// Входы и выходы графа в порядке параметров и результатов IR.
fn io_of(g: &Graph, kind: TensorKind, live: &[bool]) -> Vec<IoTensor> {
    g.tensors
        .iter()
        .filter(|t| t.kind == kind && live[t.id.0 as usize])
        .map(|t| IoTensor { id: t.id, dtype: t.dtype, shape: t.shape.clone() })
        .collect()
}
