use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileError, CompileOptions, Ptx};
use synaptix_core::error::{Result, SynaptixError};

static CUDA_INCLUDE: RwLock<Option<String>> = RwLock::new(None);

pub fn set_cuda_include(path: Option<String>) {
    *CUDA_INCLUDE.write().unwrap() = path;
}

/// Компиляция под цель карты (см. [`crate::caps::DeviceCaps`]): `sm_120a` на
/// Blackwell, `sm_80` на Ampere, а с `SYN_FORCE_ARCH` — под указанную цель.
pub fn compile_module(
    ctx: &Arc<CudaContext>,
    src: &str,
    tag: &'static str,
) -> Result<Arc<CudaModule>> {
    compile_module_with_opts(ctx, src, tag, &[], None)
}

/// Как [`compile_module_with_opts`] под цель карты, но сначала проверяет,
/// что карта (или принудительная цель) имеет возможность `need`; иначе —
/// `Unsupported` с понятным сообщением вместо лога NVRTC. Так гейтятся
/// модули, целиком написанные на block-scale MMA (NVFP4/MXFP8 GEMM/GEMV,
/// fused NVFP4-проекции) и на TMA.
pub fn compile_module_req(
    ctx: &Arc<CudaContext>,
    src: &str,
    tag: &'static str,
    extra_options: &[&str],
    need: crate::caps::Feature,
) -> Result<Arc<CudaModule>> {
    crate::caps::DeviceCaps::for_context(ctx).require(need)?;
    compile_module_with_opts(ctx, src, tag, extra_options, None)
}

// ── Дисковый кэш NVRTC (source→PTX) ─────────────────────────────────────────
//
// NVRTC-компиляция .cu → PTX стоит сотни мс на модуль; холодный старт LLM
// платит секунды на каждый процесс (PTX→SASS кэширует драйверный
// ComputeCache, а source→PTX — нет). Кэшируем PTX на диск: ключ — полный
// исходник + опции + arch + версия NVRTC; имя файла — FNV-64 от ключа, сам
// ключ хранится в файле и сверяется побайтово при чтении (коллизия → miss).
// Выключить: SYN_NVRTC_CACHE=0; каталог: SYN_NVRTC_CACHE_DIR (дефолт
// $XDG_CACHE_HOME/synaptix/nvrtc или ~/.cache/synaptix/nvrtc).

const CACHE_MAGIC: &[u8; 8] = b"SYNPTX1\0";

fn cache_enabled() -> bool {
    std::env::var("SYN_NVRTC_CACHE").as_deref() != Ok("0")
}

fn cache_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("SYN_NVRTC_CACHE_DIR") {
        return Some(PathBuf::from(d));
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| std::env::var("HOME").ok().map(|h| Path::new(&h).join(".cache")))?;
    Some(base.join("synaptix").join("nvrtc"))
}

fn nvrtc_version() -> (i32, i32) {
    let mut major = 0i32;
    let mut minor = 0i32;
    let rc = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    if rc != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        return (0, 0);
    }
    (major, minor)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Полный ключ кэша: всё, что влияет на PTX. Опции склеены с \x1f
/// (не встречается в валидных флагах), версия NVRTC — прокси для версии
/// заголовков toolkit'а (cuda_fp16.h и т.п. едут вместе с ним).
fn cache_key(src: &str, options: &[String], arch: &str) -> Vec<u8> {
    let (maj, min) = nvrtc_version();
    let mut key = Vec::with_capacity(src.len() + 128);
    key.extend_from_slice(format!("nvrtc={maj}.{min}\x1farch={arch}\x1f").as_bytes());
    for o in options {
        key.extend_from_slice(o.as_bytes());
        key.push(0x1f);
    }
    key.push(0x1e);
    key.extend_from_slice(src.as_bytes());
    key
}

fn cache_read(path: &Path, key: &[u8]) -> Option<String> {
    let data = std::fs::read(path).ok()?;
    let rest = data.strip_prefix(CACHE_MAGIC.as_slice())?;
    let (klen_b, rest) = rest.split_at_checked(4)?;
    let klen = u32::from_le_bytes(klen_b.try_into().ok()?) as usize;
    let (stored_key, rest) = rest.split_at_checked(klen)?;
    if stored_key != key {
        return None;
    }
    let (plen_b, rest) = rest.split_at_checked(4)?;
    let plen = u32::from_le_bytes(plen_b.try_into().ok()?) as usize;
    let ptx = rest.get(..plen)?;
    String::from_utf8(ptx.to_vec()).ok()
}

fn cache_write(path: &Path, key: &[u8], ptx: &str) {
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let mut buf = Vec::with_capacity(16 + key.len() + ptx.len());
    buf.extend_from_slice(CACHE_MAGIC);
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&(ptx.len() as u32).to_le_bytes());
    buf.extend_from_slice(ptx.as_bytes());
    // Атомарно: tmp с pid → rename (параллельные процессы не бьют друг друга).
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, &buf).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

/// `arch = None` — цель карты из [`crate::caps::DeviceCaps`] (с учётом
/// `SYN_FORCE_ARCH`); `Some(..)` — явная цель (тесты матрицы архитектур).
/// В опции всегда добавляются `-DSYN_CC=…` и `-DSYN_ARCH_A=…` под выбранную
/// цель, чтобы `.cu` мог гейтить ветки препроцессором.
pub fn compile_module_with_opts(
    ctx: &Arc<CudaContext>,
    src: &str,
    tag: &'static str,
    extra_options: &[&str],
    arch: Option<&'static str>,
) -> Result<Arc<CudaModule>> {
    let mut options: Vec<String> = extra_options.iter().map(|s| (*s).to_string()).collect();
    let arch_s = match arch {
        Some(a) => {
            options.extend(defines_for(a, tag)?);
            a
        }
        None => {
            let caps = crate::caps::DeviceCaps::for_context(ctx);
            options.extend(caps.defines());
            caps.arch
        }
    };
    let (ptx, cached) = compile_to_ptx(src, tag, options.clone(), arch_s, true)?;
    match ctx.load_module(ptx) {
        Ok(m) => Ok(m),
        // Битый/несовместимый PTX из кэша — перекомпилируем и перезапишем.
        Err(_) if cached => {
            let (ptx, _) = compile_to_ptx(src, tag, options, arch_s, false)?;
            ctx.load_module(ptx)
                .map_err(|e| SynaptixError::Cuda(format!("load_module {tag}: {e:?}")))
        }
        Err(e) => Err(SynaptixError::Cuda(format!("load_module {tag}: {e:?}"))),
    }
}

/// Только NVRTC, без загрузки в контекст: для матрицы архитектур в тестах
/// (PTX под `sm_90a` на карте sm_120 не загрузить, а собрать — можно).
/// Возвращает текст PTX.
pub fn compile_ptx_only(
    src: &str,
    tag: &'static str,
    extra_options: &[&str],
    arch: &'static str,
) -> Result<String> {
    let mut options: Vec<String> = extra_options.iter().map(|s| (*s).to_string()).collect();
    options.extend(defines_for(arch, tag)?);
    Ok(compile_to_ptx(src, tag, options, arch, true)?.0.to_src())
}

/// `-DSYN_CC=…`/`-DSYN_ARCH_A=…` под явную цель.
fn defines_for(arch: &str, tag: &str) -> Result<[String; 2]> {
    let (cc, accel) = crate::caps::parse_force_arch(arch)
        .ok_or_else(|| SynaptixError::Cuda(format!("nvrtc {tag}: цель '{arch}' не разобрана")))?;
    let accel = accel && cc >= (9, 0);
    Ok([
        format!("-DSYN_CC={}", cc.0 * 10 + cc.1),
        format!("-DSYN_ARCH_A={}", u8::from(accel)),
    ])
}

/// `(PTX, взят из кэша)`. `allow_cache = false` — принудительная
/// перекомпиляция с перезаписью кэша.
fn compile_to_ptx(
    src: &str,
    tag: &'static str,
    mut options: Vec<String>,
    arch_s: &'static str,
    allow_cache: bool,
) -> Result<(Ptx, bool)> {
    let mut include_paths = Vec::new();
    if let Some(p) = CUDA_INCLUDE.read().unwrap().clone() {
        include_paths.push(p);
    }
    for candidate in &[
        "/opt/cuda/include",
        "/usr/local/cuda/include",
        "/usr/include/cuda",
    ] {
        if Path::new(candidate).join("cuda_fp16.h").is_file() {
            include_paths.push((*candidate).to_string());
        }
    }
    options.push("-lineinfo".to_string());

    let (cache_path, key) = if cache_enabled() {
        let key = cache_key(src, &options, arch_s);
        let path = cache_dir().map(|d| d.join(format!("{tag}-{:016x}.ptx", fnv1a64(&key))));
        (path, Some(key))
    } else {
        (None, None)
    };

    if allow_cache {
        if let (Some(path), Some(key)) = (cache_path.as_ref(), key.as_ref()) {
            if let Some(ptx_src) = cache_read(path, key) {
                return Ok((Ptx::from_src(ptx_src), true));
            }
        }
    }

    let opts = CompileOptions {
        arch: Some(arch_s),
        include_paths,
        options,
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(src, opts).map_err(|e| {
        let log_str = match &e {
            CompileError::CompileError { log, .. } => log.to_string_lossy().to_string(),
            other => format!("{other:?}"),
        };
        SynaptixError::Cuda(format!("nvrtc {tag} [{arch_s}]: {log_str}"))
    })?;
    if let (Some(path), Some(key)) = (cache_path.as_ref(), key.as_ref()) {
        cache_write(path, key, &ptx.to_src());
    }
    Ok((ptx, false))
}

pub fn load_fn(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction> {
    module
        .load_function(name)
        .map_err(|e| SynaptixError::Cuda(format!("load_function {name}: {e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_roundtrip_and_key_mismatch() {
        let dir = std::env::temp_dir().join(format!("syn-nvrtc-test-{}", std::process::id()));
        let path = dir.join("t-abc.ptx");
        let key = cache_key("__global__ void k(){}", &["-O3".into()], "sm_80");
        assert_eq!(cache_read(&path, &key), None);
        cache_write(&path, &key, "// ptx body");
        assert_eq!(cache_read(&path, &key).as_deref(), Some("// ptx body"));
        let other = cache_key("__global__ void k2(){}", &["-O3".into()], "sm_80");
        assert_eq!(cache_read(&path, &other), None);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
