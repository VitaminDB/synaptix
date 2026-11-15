//! Кодировщик NVFP4 на карте с выбором масштабов и поколоночной компенсацией
//! ошибки (GPTQ внутри блока): `[n, width]` f32 → полубайты, коды масштабов
//! построчно и деквант. Модуль собирается с `--fmad=false` и повторяет
//! CPU-эталон `transcode::nvfp4_encode_rows` бит в бит (тест
//! `cuda_nvfp4_encode`): коды E4M3 — по таблице порогов, посчитанной здесь
//! же CPU-кодировщиком.

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

pub struct Nvfp4EncodeKernels {
    _module: Arc<CudaModule>,
    encode: CudaFunction,
    /// Пороги кодов E4M3 `[127]` и значения кодов `[127]` на карте.
    thr: CudaSlice<f32>,
    dec: CudaSlice<f32>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Nvfp4EncodeKernels>)>>> = OnceLock::new();

/// `thr[c]` — наименьшее положительное f32, у которого `encode_e4m3 ≥ c`
/// (c = 1..=0x7e; код монотонен по x); `dec[c]` — `decode_e4m3(c)`.
pub fn e4m3_tables() -> (Vec<f32>, Vec<f32>) {
    use synaptix_kernels_cpu::quant::fp8_e4m3::{decode_e4m3, encode_e4m3};
    let mut thr = vec![0f32; 0x7f];
    for (c, t) in thr.iter_mut().enumerate().skip(1) {
        // Бисекция по битам положительных f32 (порядок битов = порядок чисел).
        let (mut lo, mut hi) = (0u32, 448f32.to_bits());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if encode_e4m3(f32::from_bits(mid)) as usize >= c { hi = mid } else { lo = mid + 1 }
        }
        *t = f32::from_bits(lo);
    }
    let dec = (0..0x7f).map(|c| decode_e4m3(c as u8)).collect();
    (thr, dec)
}

impl Nvfp4EncodeKernels {
    pub fn for_context(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>) -> Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        if let Some((_, v)) = cache.lock().unwrap().iter().find(|(k, _)| *k == key) {
            return Ok(v.clone());
        }
        let module = compile_module_with_opts(ctx, include_str!("../cu/elementwise/nvfp4_encode.cu"), "nvfp4_encode.cu", &["--fmad=false"], None)?;
        let (thr, dec) = e4m3_tables();
        let err = |e| SynaptixError::Cuda(format!("nvfp4_encode: таблицы: {e:?}"));
        let new = Arc::new(Self {
            encode: load_fn(&module, "nvfp4_encode_rows")?,
            _module: module,
            thr: stream.clone_htod(&thr).map_err(err)?,
            dec: stream.clone_htod(&dec).map_err(err)?,
        });
        cache.lock().unwrap().push((key, new.clone()));
        Ok(new)
    }
}

pub use synaptix_core::backend::Nvfp4EncodeArgs;

/// Закодировать `w` (`n × width` f32 на карте). Возвращает полубайты
/// `[n, width/2]`, коды масштабов `[n, width/16]` (на хосте) и деквант
/// `[n, width]` f32 на карте (байтовый буфер `n·width·4`).
pub fn nvfp4_encode_rows(
    kernels: &Nvfp4EncodeKernels,
    stream: &Arc<CudaStream>,
    w: &cudarc::driver::CudaView<'_, f32>,
    a: &Nvfp4EncodeArgs<'_>,
) -> Result<(Vec<u8>, Vec<u8>, CudaSlice<u8>)> {
    let (n, width) = (a.n, a.width);
    if width % 16 != 0 || w.len() < n * width {
        return Err(SynaptixError::Unsupported("nvfp4_encode: ширина не кратна 16 или буфер короче формы"));
    }
    if a.u.is_some() && (width != 128 || a.u.unwrap().len() != width * width) {
        return Err(SynaptixError::Unsupported("nvfp4_encode: компенсация ошибки — только блок 128×128"));
    }
    if a.d.is_some_and(|d| d.len() != width) || a.wimp.is_some_and(|m| m.len() != width) {
        return Err(SynaptixError::Unsupported("nvfp4_encode: d/важность не длины width"));
    }
    let cuerr = |what: &'static str| move |e| SynaptixError::Cuda(format!("nvfp4_encode: {what}: {e:?}"));
    let up = |v: Option<&[f32]>| -> Result<Option<CudaSlice<f32>>> { v.map(|v| stream.clone_htod(v).map_err(cuerr("загрузка"))).transpose() };
    let (u, d, m) = (up(a.u)?, up(a.d)?, up(a.wimp)?);
    let mut packed = stream.alloc_zeros::<u8>(n * width / 2).map_err(cuerr("alloc"))?;
    let mut scales = stream.alloc_zeros::<u8>(n * width / 16).map_err(cuerr("alloc"))?;
    let mut deq = stream.alloc_zeros::<u8>(n * width * 4).map_err(cuerr("alloc"))?;
    if n > 0 {
        let null: u64 = 0;
        let (width_i, search, noclip) = (width as i32, a.search as i32, a.noclip as i32);
        let mut b = stream.launch_builder(&kernels.encode);
        b.arg(w).arg(&width_i);
        for s in [&u, &d, &m] {
            match s {
                Some(s) => b.arg(s),
                None => b.arg(&null),
            };
        }
        let mut deq_f = unsafe { deq.transmute_mut::<f32>(n * width) }.ok_or(SynaptixError::Unsupported("nvfp4_encode: вид f32"))?;
        b.arg(&kernels.thr).arg(&kernels.dec).arg(&search).arg(&noclip).arg(&mut packed).arg(&mut scales).arg(&mut deq_f);
        let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        unsafe { b.launch(cfg) }.map_err(|e| SynaptixError::Cuda(format!("launch nvfp4_encode_rows: {e:?}")))?;
    }
    let p = stream.clone_dtoh(&packed).map_err(cuerr("выгрузка"))?;
    let s = stream.clone_dtoh(&scales).map_err(cuerr("выгрузка"))?;
    Ok((p, s, deq))
}
