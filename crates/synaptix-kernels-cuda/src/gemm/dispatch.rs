use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::storage::{CudaBuf, Storage};

use crate::best_cu::gemm::gemm_nvfp4::{
    gemm_nvfp4_full_cfg_view, GemmNvfp4FullKernels, Nvfp4FullCfg,
};
use crate::best_cu::gemm::gemm_nvfp4::{
    nvfp4_mma_gemm_shuf_2d_f16_view, nvfp4_mma_gemm_shuf_2dr_f16_view,
    nvfp4_mma_gemm_shuf_f16_view, nvfp4_mma_gemm_shuf_n8_f16_view, Gemm2drConfig,
    Nvfp4MmaGemmShufKernels,
};
use crate::best_cu::gemv::gemv_nvfp4::{
    nvfp4_mma_gemv_shuf_f16_view, nvfp4_w_repack, Nvfp4MmaGemvShufKernels,
};
use crate::elementwise::quant::{
    nvfp4_scale_buffer_size, quantize_f16_to_nvfp4_view, Nvfp4QuantKernels,
};
use crate::best_cu::gemm::gemm_mxfp8::{gemm_mxfp8, GemmMxFp8Kernels};
use crate::elementwise::quant::Mxfp8QuantKernels;
use crate::wsalloc::WsAlloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nvfp4Plan {
    Gemv,
    Reg,
    Coop,
    N8,
    Broadcast,
    Full(Nvfp4FullCfg),
}

impl Nvfp4Plan {
    pub fn name(&self) -> &'static str {
        match self {
            Nvfp4Plan::Gemv => "gemv_shuf",
            Nvfp4Plan::Reg => "2dr",
            Nvfp4Plan::Coop => "2d_coop",
            Nvfp4Plan::N8 => "n8",
            Nvfp4Plan::Broadcast => "4a_broadcast",
            Nvfp4Plan::Full(cfg) => cfg.fname(),
        }
    }
}

#[inline]
fn dr_block(c: Gemm2drConfig) -> (u32, u32) {
    (c.warps_m * c.mu * 16, c.warps_n * c.nu * 8)
}

const DR_CFGS: [Gemm2drConfig; 5] = [
    Gemm2drConfig::R_2X2_M4N8,
    Gemm2drConfig::R_2X2_M4N4,
    Gemm2drConfig::R_4X2_M2N4,
    Gemm2drConfig::R_4X4_M2N2,
    Gemm2drConfig::R_2X2_M2N2,
];

fn fits_2dr(n: u32, batch: u32) -> bool {
    DR_CFGS.iter().any(|&c| {
        let (bm, bn) = dr_block(c);
        n % bm == 0 && batch % bn == 0
    })
}

fn fits_n8(n: u32, batch: u32) -> bool {
    batch % 8 == 0 && n % 64 == 0
}

fn fits_2d(n: u32, batch: u32) -> bool {
    const COOP: [(u32, u32); 5] = [(8, 4), (4, 8), (4, 4), (4, 2), (2, 2)];
    COOP.iter()
        .any(|&(wm, wn)| n % (wm * 16) == 0 && batch % (wn * 8) == 0)
}

const NVFP4_L2_WEIGHT_BYTES: u64 = 24 * 1024 * 1024;

fn pick_nvfp4_full(m: u32, n: u32, k: u32) -> Option<Nvfp4FullCfg> {
    // ROT = порт k64-шедулинга CUTLASS sm120 (sm120_blockscaled_mma_tma.hpp):
    // регистровый double-buffer фрагментов + ранний release стадии + wait перед
    // последней gemm-пачкой + staged-эпилог + ::cta-TMA (::cluster глушил setmaxnreg).
    // Свип 2026-06-04 (bit-exact per-row 0): лучший/паритет на ВСЕХ LTX-формах,
    // M=512..26624 — attn 510→594-617, ff_up 487→572, ff_down 552→561 (паритет)/635;
    // persist проигрывает rot даже на «своей» ff_up/26624 (372 vs 572 TF).
    // Зонирование mid-M ПЕРЕПИСАНО по events-протоколу (события + SM-флаш,
    // свип 2026-06-05 вечер): старые зоны (b64 321-448 и др.) строились на
    // wall+sync+CE-флаше, который ронял SM-клок и врал порядком конфигов.
    // Победители (наш pure vs qutlass pure): attn 1.00/0.96/1.01/1.08/0.90/0.95,
    // ff_up 1.03/1.10/1.11/1.08/1.08, ff_down 1.03/1.15/1.13/1.17/1.10 (m 128..1024).
    let pick_first = |cands: &[Nvfp4FullCfg]| -> Option<Nvfp4FullCfg> {
        cands.iter().copied().find(|c| c.fits(m, n, k))
    };
    {
        // ROT — чемпион m>=512 (свип 2026-06-04: attn 594-617, ff_up 572);
        // исключение ffu-512: persist 508.8 vs rot 472.7 (зона n>k ниже).
        let persist_512 = n > k && m == 512;
        let c = Nvfp4FullCfg::C_128_256_S3_SWZ_ROT;
        if m >= 512 && !persist_512 && c.fits(m, n, k) {
            return Some(c);
        }
    }
    // Батч-тайл 64: малые M на всех формах (32-64: attn 46.8/93.8, ffd 73.5/144.3,
    // ffu 72.1/142.6 — ВСЕ бьют их pure; N8-план был артефактом кривого протокола)
    // + зона 65-128 на узком N (attn 187.1, ffd 263.1); на ff_up с m>=97 b64
    // проигрывает c256/persist (215 vs 272-279).
    let b64_on = (32..=128).contains(&m) && (n <= k || m <= 96);
    if m < 512 && b64_on {
        if let Some(c) =
            pick_first(&[Nvfp4FullCfg::C_128_64_S4_SWZ, Nvfp4FullCfg::C_128_64_S3_SWZ])
        {
            return Some(c);
        }
    }
    // Зоны n>k свипованы на ff_up LTX (вес 32MB); гиганты (qwen gate 70MB,
    // lm_head 635MB) — вне свипа, остаются на legacy-пути (persist-raster).
    let w_in_sweep = (n as u64) * (k as u64) / 2 <= 40 * 1024 * 1024;
    if (97..=512).contains(&m) && (n <= k || w_in_sweep) {
        let cands: &[Nvfp4FullCfg] = if n > k {
            // ff_up-класс (широкий N): persist gridless-тайлы душат wave-quant.
            match m {
                97..=160 => &[
                    Nvfp4FullCfg::C_PERSIST_C256_S4_SWZ,
                    Nvfp4FullCfg::C_128_128_C256_S4_SWZ_DROT,
                    Nvfp4FullCfg::C_128_128_C256_S4_SWZ,
                ],
                161..=320 => &[
                    Nvfp4FullCfg::C_128_256_S3_SWZ_DROT,
                    Nvfp4FullCfg::C_128_256_S3_SWZ_ROT,
                    Nvfp4FullCfg::C_128_128_C256_S4_SWZ,
                ],
                _ => &[
                    Nvfp4FullCfg::C_PERSIST_C256_S4_SWZ,
                    Nvfp4FullCfg::C_128_256_S3_SWZ_DROT,
                    Nvfp4FullCfg::C_128_128_C256_S4_SWZ,
                ],
            }
        } else {
            // attn/ff_down-класс (узкий N=4096).
            match m {
                129..=320 => &[
                    Nvfp4FullCfg::C_128_128_C256_S4_SWZ_DROT,
                    Nvfp4FullCfg::C_128_128_C256_S4_SWZ,
                ],
                321..=511 => &[
                    Nvfp4FullCfg::C_128_256_S3_SWZ_DROT,
                    Nvfp4FullCfg::C_128_256_S3_SWZ_ROT,
                    Nvfp4FullCfg::C_128_128_C256_S4_SWZ,
                ],
                _ => &[],
            }
        };
        if let Some(c) = pick_first(cands) {
            return Some(c);
        }
    }
    // Вес > L2-порога: persist выигрывает только при n>=k (ff_up-класс, 526 vs 334);
    // при k>n (ff_down-класс) 128×128 вдвое быстрее (440 vs 216) — A/B 2026-06-04.
    let weight_fits_l2 = (n as u64) * (k as u64) / 2 <= NVFP4_L2_WEIGHT_BYTES || k > n;
    // C_128_256 (batch-256, конфиг qutlass-ядра): sfb-мультитайл в шаблоне починен
    // (per-row 0). C_256_128 корректен, но медленнее 128×128 — не в кандидатах.
    let candidates: &[Nvfp4FullCfg] = if weight_fits_l2 {
        &[
            Nvfp4FullCfg::C_128_128_C256_S4_SWZ,
            Nvfp4FullCfg::C_128_128_C256_S3_SWZ,
        ]
    } else {
        &[
            Nvfp4FullCfg::C_PERSIST_C256_S4_SWZ,
            Nvfp4FullCfg::C_PERSIST_C256_S3_SWZ,
        ]
    };
    candidates.iter().copied().find(|c| c.fits(m, n, k))
}

pub fn pick_nvfp4(m: u32, n: u32, k: u32) -> Nvfp4Plan {
    if m == 1 {
        return Nvfp4Plan::Gemv;
    }
    // Full-ядро (gn_nvfp4_full_*, M%128==0, persistent-CTA+TMA, самый быстрый large-M)
    // ПОЧИНЕНО: было ДВА бага (оба скрыты cos) — (1) sfb=lane&7 вместо lane>>2 (тот же
    // класс, что N8/2dr); (2) swizzle off>>6 вместо off>>7 (off-by-one vs cute
    // Swizzle<2,4,3> для SWIZZLE_64B). Теперь per-row 0.12 = бит-идентичен Broadcast
    // (diag full_perrow_vs_dense). Включено по умолчанию.
    // Порог m>=32 (был 128): Full 128×64 с TMA-OOB бьёт их pure и на малых M
    // (events-свип 2026-06-05: attn-32/64 46.8/93.8 vs их 45.5/89.0, ffd 73.5/144.3
    // vs 66.3/130.9, ffu 72.1/142.6 vs 69.9/137.7); N8-план был выбором кривого
    // протокола (e2e attn-32 39 / ffd-64 83 — Full в 1.2-1.7× быстрее).
    if m >= 32 && k % 128 == 0 {
        if let Some(cfg) = pick_nvfp4_full(m, n, k) {
            return Nvfp4Plan::Full(cfg);
        }
    }
    // Reg(2dr)/N8/Coop — баг B-scale (sfb читал scale ЧУЖОЙ batch-строки: sfb=lane&7
    // при B-data n_t=lane>>2) ПОЧИНЕН (sfb=lane>>2, совпал). diag_perrow: per-row 0.12
    // (=Broadcast, квант-шум). Строко-корректны и быстры → используем по умолчанию.
    // m<=64 НА УЗКОМ N: N8 быстрее Reg в 1.5× (свип 2026-06-05: attn-32
    // 24.4→38.9, attn-64 48.8→72.4, ffd-64 54.9→85.7); на широком N=16384
    // НАОБОРОТ (ff_up-64 134.7→80.3) — Reg-грид там и так заполняет машину.
    if m <= 64 && n <= 8192 && fits_n8(n, m) {
        Nvfp4Plan::N8
    } else if fits_2dr(n, m) {
        Nvfp4Plan::Reg
    } else if fits_n8(n, m) {
        Nvfp4Plan::N8
    } else if n % 64 == 0 {
        Nvfp4Plan::Broadcast
    } else if fits_2d(n, m) {
        Nvfp4Plan::Coop
    } else {
        Nvfp4Plan::Broadcast
    }
}

/// TODO(perf): x/out копируются в типизированные f16-буферы (kernel-обёртки ждут

#[allow(clippy::too_many_arguments)]
pub fn nvfp4_linear_f16(
    gemm_k: &Nvfp4MmaGemmShufKernels,
    gemv_k: &Nvfp4MmaGemvShufKernels,
    quant_k: &Nvfp4QuantKernels,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    ordinal: usize,
    x_u8: &CudaSlice<u8>,
    out_u8: &mut CudaSlice<u8>,
    w: &QuantWeight,
    m: u32,
    out_bf16: bool,
    // Prequant: УЖЕ квантованная активация (packed, scales) — пропускает quantize.
    // Позволяет квантовать общий `h` один раз и переиспользовать во всех проекциях.
    prequant: Option<(&CudaSlice<u8>, &CudaSlice<u8>)>,
) -> Result<Nvfp4Plan> {
    if w.dtype() != DType::NVFP4 {
        return Err(SynaptixError::Unsupported(
            "nvfp4_linear_f16: вес должен быть NVFP4",
        ));
    }
    let n = w.n() as u32;
    let k = w.k() as u32;
    if k % 64 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "nvfp4_linear_f16: K={k} должно быть кратно 64"
        )));
    }
    if m == 0 || n == 0 {
        return Ok(pick_nvfp4(m.max(1), n, k));
    }

    // M-ПАДДИНГ (prefill, m>1): GEMM-ядра требуют выровненного M; невыровненный
    // large-M падает в медленный Broadcast (8-9× медленнее — был корень «prefill 50×»).
    // m>=128 → паддим до 128 (→ Full, самый быстрый large-M; Reg на small-M вдвое
    // медленнее); меньшие m → до 16 (→ Reg; round-128 дал бы слишком большой overhead).
    // НИКОГДА Broadcast. Хвостовые строки m..m_run нулевые (alloc_zeros packed+scales)
    // → их выход отбрасывается копией ниже. decode (m=1 / prequant) не паддим (→ Gemv).
    // Корректность: scale-layout позиция строки не зависит от outer-M,
    // scale_buffer_size(m_run)>=size(m) → quantize m строк в m_run-буфер пишет верные
    // позиции, хвост остаётся нулём.
    // Паддим И prequant-ветку (qkv/gate/up шарят квант-активацию h): без этого
    // невыровненный prequant-M падал в Broadcast (был корень «350/1660 prefill 8×»).
    // m>=1280 паддим до 256 (Full-тайл 128×256 ROT требует M%256==0; overhead
    // ≤255 нулевых строк ≤10% при m≥1280, выигрыш rot-ядра +8-17% перекрывает).
    // Full non-persist (m>=128): батч ЛЮБОЙ — TMA OOB-нули + BATCH-гард
    // эпилогов → без паддинга, без out_pad-скретча и 218MB-копии. Persist и
    // мелкие классы (Reg/N8) — старый паддинг.
    // Порог 32 (был 128) — синхронно с pick_nvfp4: Full на малых M бьёт N8/Reg.
    let full_direct = m >= 32
        && k % 128 == 0
        && pick_nvfp4_full(m, n, k).map(|c| !c.persistent).unwrap_or(false);
    let m_run = if full_direct {
        m
    } else if m > 1 {
        if m >= 1280 {
            (m + 255) & !255
        } else if m >= 128 {
            (m + 127) & !127
        } else {
            (m + 15) & !15
        }
    } else {
        m
    };
    let padded = m_run != m;
    let mk_run = (m_run as usize) * (k as usize);
    let mn = (m as usize) * (n as usize);
    let mn_run = (m_run as usize) * (n as usize);

    // Активация в NVFP4: либо переданная prequant (общий `h`, квантован 1×), либо
    // квантуем здесь. owned_self — самоквантованные (packed,scales); owned_pad_p —
    // padded-копия prequant-packed. packed_x/scales_x — ссылки на тот источник, что есть.
    let owned_self: Option<(CudaSlice<u8>, CudaSlice<u8>)> = if prequant.is_none() {
        let x_view = unsafe { x_u8.transmute::<f16>((m as usize) * (k as usize)) }
            .ok_or_else(|| SynaptixError::Cuda("nvfp4_linear: transmute x→f16".into()))?;
        // uninit + zero ТОЛЬКО хвоста (alloc_zeros делал CE-memset всего буфера
        // на каждый вызов — 109MB на 26624; квант пишет первые m строк целиком).
        let mut packed_x = unsafe { stream.ws_alloc::<u8>(mk_run / 2) }
            .map_err(|e| SynaptixError::Cuda(format!("nvfp4_linear: alloc packed_x: {e:?}")))?;
        if padded {
            let mut tail = packed_x.slice_mut((m as usize) * (k as usize) / 2..);
            stream
                .memset_zeros(&mut tail)
                .map_err(|e| SynaptixError::Cuda(format!("nvfp4_linear: zero packed tail: {e:?}")))?;
        }
        // uninit: контракт quantize_f16_to_nvfp4_view — буфер скейлов после
        // вызова полностью определён (fast-ядро зануляет scale-хвост 128-тайла
        // через outer_cov; slow-fallback зануляет буфер сам). CE-memset скейлов
        // на каждый вызов ел до 9% e2e на attn-128 (A/B 2026-06-05).
        let scales_sz = nvfp4_scale_buffer_size(m_run as usize, k as usize);
        let mut scales_x = unsafe { stream.ws_alloc::<u8>(scales_sz) }
            .map_err(|e| SynaptixError::Cuda(format!("nvfp4_linear: alloc scales_x: {e:?}")))?;
        // квантуем m РЕАЛЬНЫХ строк; хвост m..m_run остаётся нулевым.
        quantize_f16_to_nvfp4_view(quant_k, stream, &x_view, &mut packed_x, &mut scales_x, m, k)?;
        Some((packed_x, scales_x))
    } else {
        None
    };
    // prequant packed = m строк (m*k/2 байт); ядро при m_run читает m_run строк → паддим
    // копией в m_run-буфер (хвост ноль из alloc_zeros). Scales НЕ паддим: nvfp4_scale_buffer_size(m)
    // уже округляет outer до 128 → покрывает m_run строк (round128(m)==ceil(m/128)*128==coverage;
    // round16(m)<=128 для m<128). Bit-точно: позиция scale строки i не зависит от outer-M.
    let owned_pad_p: Option<CudaSlice<u8>> = match prequant {
        Some((p, _)) if padded => {
            let copy_bytes = p.len().min(mk_run / 2);
            let mut padded_p = unsafe { stream.ws_alloc::<u8>(mk_run / 2) }
                .map_err(|e| SynaptixError::Cuda(format!("nvfp4_linear: alloc padded prequant: {e:?}")))?;
            {
                let mut tail = padded_p.slice_mut(copy_bytes..);
                stream
                    .memset_zeros(&mut tail)
                    .map_err(|e| SynaptixError::Cuda(format!("nvfp4_linear: zero prequant tail: {e:?}")))?;
            }
            stream
                .memcpy_dtod(&p.slice(0..copy_bytes), &mut padded_p.slice_mut(0..copy_bytes))
                .map_err(|e| SynaptixError::Cuda(format!("nvfp4_linear: copy prequant pad: {e:?}")))?;
            Some(padded_p)
        }
        _ => None,
    };
    let (packed_x, scales_x): (&CudaSlice<u8>, &CudaSlice<u8>) = match (&owned_self, &owned_pad_p, prequant) {
        (Some((p, s)), _, _) => (p, s),
        (None, Some(pp), Some((_, s))) => (pp, s),
        (None, None, Some((p, s))) => (p, s),
        _ => return Err(SynaptixError::Cuda("nvfp4_linear: нет источника активации".into())),
    };

    let plan = pick_nvfp4(m_run, n, k);

    let scales_w = w
        .scales()
        .as_cuda()
        .ok_or(SynaptixError::Unsupported(
            "nvfp4_linear: scales W non-cuda",
        ))?
        .slice();

    // out: при паддинге — отдельный буфер m_run×n (ядро пишет m_run строк), иначе
    // пишем прямо в out_u8 (m×n). Хвостовые строки копией ниже отбрасываются.
    let mut owned_out: Option<CudaSlice<u8>> = if padded {
        // uninit: ядро пишет все m_run×n строк (alloc_zeros memset'ил 218MB).
        Some(unsafe { stream.ws_alloc::<u8>(mn_run * 2) }
            .map_err(|e| SynaptixError::Cuda(format!("nvfp4_linear: alloc out_pad: {e:?}")))?)
    } else {
        None
    };

    let shuf_storage = w.shuffled_or_try_init(|| {
        let packed_arc = w.packed_arc().ok_or_else(|| {
            SynaptixError::Cuda("nvfp4_linear: raw W освобождён, shuffled не построить".into())
        })?;
        let raw_w = packed_arc
            .as_cuda()
            .ok_or(SynaptixError::Unsupported(
                "nvfp4_linear: packed W non-cuda",
            ))?
            .slice();
        let nbytes = raw_w.len();
        let buf = stream
            .cache_alloc_zeros::<u8>(nbytes)
            .map_err(|e| SynaptixError::Cuda(format!("nvfp4_linear: alloc shuf W: {e:?}")))?;
        let mut storage = Storage::Cuda(CudaBuf::new(ctx.clone(), stream.clone(), buf, ordinal));
        {
            let shuf = storage
                .as_cuda_mut()
                .ok_or(SynaptixError::Unsupported(
                    "nvfp4_linear: shuf storage non-cuda",
                ))?
                .slice_mut();
            nvfp4_w_repack(gemv_k, stream, raw_w, shuf, n, k)?;
        }
        Ok(Arc::new(storage))
    })?;
    // Безусловно освобождаем сырой packed-NVFP4 сразу после построения shuf: shuf —
    // ЕДИНСТВЕННЫЙ формат, который читают все GEMV/GEMM-планы (CUTLASS/TmaWs2,
    // которым нужен был packed, вырезаны в decutlass). Держать обе копии = 2×
    // footprint весов → OOM при длинном контексте. (Per-weight: peak остаётся 1×.)
    w.release_packed();
    let shuf_w = shuf_storage
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("nvfp4_linear: shuf W non-cuda"))?
        .slice();
    {
        let mut out_view = if let Some(op) = owned_out.as_mut() {
            unsafe { op.transmute_mut::<f16>(mn_run) }
        } else {
            unsafe { out_u8.transmute_mut::<f16>(mn) }
        }
        .ok_or_else(|| SynaptixError::Cuda("nvfp4_linear: transmute out→f16".into()))?;
        match plan {
            Nvfp4Plan::Gemv => nvfp4_mma_gemv_shuf_f16_view(
                gemv_k, stream, shuf_w, scales_w, packed_x, scales_x, &mut out_view, n, k,
            )?,
            Nvfp4Plan::Reg => nvfp4_mma_gemm_shuf_2dr_f16_view(
                gemm_k, stream, shuf_w, scales_w, packed_x, scales_x, &mut out_view, n, k, m_run,
            )?,
            Nvfp4Plan::Coop => nvfp4_mma_gemm_shuf_2d_f16_view(
                gemm_k, stream, shuf_w, scales_w, packed_x, scales_x, &mut out_view, n, k, m_run,
            )?,
            Nvfp4Plan::N8 => nvfp4_mma_gemm_shuf_n8_f16_view(
                gemm_k, stream, shuf_w, scales_w, packed_x, scales_x, &mut out_view, n, k, m_run,
            )?,
            Nvfp4Plan::Broadcast => nvfp4_mma_gemm_shuf_f16_view(
                gemm_k, stream, shuf_w, scales_w, packed_x, scales_x, &mut out_view, n, k, m_run,
            )?,
            Nvfp4Plan::Full(cfg) => {
                let full_k = if out_bf16 {
                    GemmNvfp4FullKernels::for_context_bf16(ctx)?
                } else {
                    GemmNvfp4FullKernels::for_context(ctx)?
                };
                gemm_nvfp4_full_cfg_view(
                    &full_k, stream, shuf_w, scales_w, packed_x, scales_x, &mut out_view, n, k,
                    m_run, cfg,
                )?
            }
        }
    }

    // padded: ядро записало m_run строк в owned_out; копируем первые m строк (m×n,
    // row-major contiguous) в реальный out_u8, отбрасывая нулевой хвост.
    if let Some(op) = owned_out.as_ref() {
        let bytes = mn * 2;
        stream
            .memcpy_dtod(&op.slice(0..bytes), &mut out_u8.slice_mut(0..bytes))
            .map_err(|e| SynaptixError::Cuda(format!("nvfp4_linear: copy out_pad: {e:?}")))?;
    }

    Ok(plan)
}

/// Размеры буферов для квантованной активации NVFP4: (packed bytes, scales bytes).
pub fn nvfp4_act_buffer_sizes(m: usize, k: usize) -> (usize, usize) {
    (m * k / 2, nvfp4_scale_buffer_size(m, k))
}

/// Квантует активацию `x` (F16, m×k) в NVFP4 packed + scales — отдельно от GEMV.
/// Квантуем общий `h` ОДИН раз → переиспользуем во всех проекциях из него
/// (`nvfp4_linear_f16(..., Some((packed, scales)))`), убирая дублирующие quantize.
pub fn nvfp4_quantize_act(
    quant_k: &Nvfp4QuantKernels,
    stream: &Arc<CudaStream>,
    x_u8: &CudaSlice<u8>,
    packed_out: &mut CudaSlice<u8>,
    scales_out: &mut CudaSlice<u8>,
    m: u32,
    k: u32,
) -> Result<()> {
    let mk = (m as usize) * (k as usize);
    let x_view = unsafe { x_u8.transmute::<f16>(mk) }
        .ok_or_else(|| SynaptixError::Cuda("nvfp4_quantize_act: transmute x→f16".into()))?;
    quantize_f16_to_nvfp4_view(quant_k, stream, &x_view, packed_out, scales_out, m, k)
}

/// MXFP8 prefill (m>1) через КОРРЕКТНОЕ ядро (cp.async, порт gau-nernst, cos=0.999999
/// на outlier) вместо dequant→BF16. Y[m,n] f16 = X[m,k] f16 @ Wᵀ. Оба операнда MXFP8:
///   • W: packed E4M3 [n,k] + НАТУРАЛЬНЫЕ E8M0 scales [n,K/32] — НАПРЯМУЮ из prod (без
///     permute/tiled-кэша: ядро ждёт натуральную раскладку).
///   • X: квантуем natural свежо; M паддим до 128 (ядро требует M%128==0).
/// Возвращает Ok(false) если N%128!=0 / K%128!=0 (только 128×128×128-кратные) —
/// тогда вызывающий делает фолбэк (dequant→BF16). Bit-точность: квант построчный, хвост
/// паддинга нулевой и отбрасывается копией первых m строк.
#[allow(clippy::too_many_arguments)]
struct Mxfp8Scratch {
    xq: CudaSlice<u8>,
    sa: CudaSlice<u8>,
}

static MXFP8_SCRATCH: OnceLock<Mutex<HashMap<usize, Mxfp8Scratch>>> = OnceLock::new();

fn with_mxfp8_scratch<R>(
    stream: &Arc<CudaStream>,
    ordinal: usize,
    xq_need: usize,
    sa_need: usize,
    f: impl FnOnce(&mut CudaSlice<u8>, &mut CudaSlice<u8>) -> Result<R>,
) -> Result<R> {
    let cache = MXFP8_SCRATCH.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| SynaptixError::Cuda("mxfp8 scratch: poisoned".into()))?;
    let slot = guard.entry(ordinal);
    let entry = match slot {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => {
            let xq = stream
                .cache_alloc_zeros::<u8>(xq_need.max(1))
                .map_err(|e| SynaptixError::Cuda(format!("mxfp8: alloc xq scratch: {e:?}")))?;
            let sa = stream
                .cache_alloc_zeros::<u8>(sa_need.max(1))
                .map_err(|e| SynaptixError::Cuda(format!("mxfp8: alloc sa scratch: {e:?}")))?;
            e.insert(Mxfp8Scratch { xq, sa })
        }
    };
    if entry.xq.len() < xq_need {
        entry.xq = stream
            .cache_alloc_zeros::<u8>(xq_need)
            .map_err(|e| SynaptixError::Cuda(format!("mxfp8: grow xq scratch: {e:?}")))?;
    }
    if entry.sa.len() < sa_need {
        entry.sa = stream
            .cache_alloc_zeros::<u8>(sa_need)
            .map_err(|e| SynaptixError::Cuda(format!("mxfp8: grow sa scratch: {e:?}")))?;
    }
    let Mxfp8Scratch { xq, sa } = entry;
    f(xq, sa)
}

/// Отдать драйверу скретчи MXFP8-путей (квант активации + деквант-полоса и
/// узкий выход фолбэка). Живут в статиках между моделями; вызывается при
/// выгрузке, чтобы `cuMemPoolTrimTo` смог вернуть сегменты. Возвращает,
/// сколько байт освобождено.
pub fn clear_mxfp8_scratch() -> usize {
    let mut freed = 0usize;
    if let Some(c) = MXFP8_SCRATCH.get() {
        if let Ok(mut g) = c.lock() {
            for s in g.values() {
                freed += s.xq.len() + s.sa.len();
            }
            g.clear();
        }
    }
    if let Some(c) = MXFP8_DEQ_SCRATCH.get() {
        if let Ok(mut g) = c.lock() {
            for s in g.values() {
                freed += s.w.len() + s.y.len();
            }
            g.clear();
        }
    }
    freed
}

pub fn mxfp8_linear_tiled(
    qk: &Mxfp8QuantKernels,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x_u8: &CudaSlice<u8>,
    out_u8: &mut CudaSlice<u8>,
    w: &QuantWeight,
    m: u32,
) -> Result<bool> {
    mxfp8_linear_tiled_n(qk, ctx, stream, x_u8, out_u8, w, m, w.n() as u32)
}

/// Как [`mxfp8_linear_tiled`], но считает только ПЕРВЫЕ `n` строк веса
/// (натуральная раскладка [N,K] → префикс строк = префикс байт, указатели те
/// же). `out_u8` при этом плотный `[m, n]`. Нужен для N вне 128-кратности:
/// голова идёт быстрым ядром, хвост — фолбэком.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_linear_tiled_n(
    qk: &Mxfp8QuantKernels,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x_u8: &CudaSlice<u8>,
    out_u8: &mut CudaSlice<u8>,
    w: &QuantWeight,
    m: u32,
    n: u32,
) -> Result<bool> {
    let k = w.k() as u32;
    let m_pad = (m + 127) & !127;
    if n % 128 != 0 || k % 128 != 0 {
        return Ok(false);
    }
    let k_us = k as usize;
    let gk = GemmMxFp8Kernels::for_context(ctx)?;

    // W: натуральные packed [n,k] + натуральные scales [n,K/32] НАПРЯМУЮ (без permute).
    let packed_arc = w
        .packed_arc()
        .ok_or_else(|| SynaptixError::Cuda("mxfp8: packed W освобождён".into()))?;
    let w_packed = packed_arc
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("mxfp8: packed non-cuda"))?
        .slice();
    let w_scales = w
        .scales()
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("mxfp8: scales non-cuda"))?
        .slice();

    // Активация: квант НАПРЯМУЮ из x (старый x_pad-путь делал alloc_zeros +
    // полную копию x — 217MB DRAM на 26520 на КАЖДЫЙ вызов); скретчи uninit
    // (квант пишет все свои строки), при m%128!=0 зануляется только хвост
    // xq/sa (fp8-нуль → вклад строки 0).
    let (m_us, mp_us) = (m as usize, m_pad as usize);
    let aligned = m == m_pad;
    let use_rot = k % 512 == 0 && k / 128 >= 2;
    // rot-путь: TMA зануляет OOB-чтения хвостового M-ряда, эпилог гардит сторы
    // → скретчи ровно на m строк и GEMM ПРЯМО в out при любом m (старый путь
    // m_pad-скретчей+копий стоил ~1GB DRAM/вызов на 26520).
    let scratch_rows = if use_rot { m_us } else { mp_us };
    let xq_need = scratch_rows * k_us;
    let sa_need = scratch_rows * k_us / 32;
    with_mxfp8_scratch(stream, ctx.ordinal(), xq_need, sa_need, |xq, sa| {
        {
            let x_view = unsafe { x_u8.transmute::<f16>(m_us * k_us) }
                .ok_or_else(|| SynaptixError::Cuda("mxfp8: transmute x→f16".into()))?;
            crate::elementwise::quant::mxfp8_quant_natural(qk, stream, &x_view, xq, sa, m, k)?;
        }
        if !use_rot && !aligned {
            let mut xq_tail = xq.slice_mut(m_us * k_us..xq_need);
            stream
                .memset_zeros(&mut xq_tail)
                .map_err(|e| SynaptixError::Cuda(format!("mxfp8: zero xq tail: {e:?}")))?;
            let mut sa_tail = sa.slice_mut(m_us * k_us / 32..sa_need);
            stream
                .memset_zeros(&mut sa_tail)
                .map_err(|e| SynaptixError::Cuda(format!("mxfp8: zero sa tail: {e:?}")))?;
        }

        run_mxfp8_gemm(&gk, stream, xq, w_packed, sa, w_scales, out_u8, m, m_pad, n, k, use_rot, aligned)
    })
}

/// MXFP8 linear из УЖЕ квантованной активации (`xq` packed [m,k] e4m3 + `sa`
/// natural scales [m,k/32] — от mxfp8_quantize_act / *_mod_quant_mxfp8).
/// Пропускает квант; только rot-путь (TMA-клип хвоста M, без паддинга).
pub fn mxfp8_linear_prequant(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    xq: &CudaSlice<u8>,
    sa: &CudaSlice<u8>,
    out_u8: &mut CudaSlice<u8>,
    w: &QuantWeight,
    m: u32,
) -> Result<()> {
    let n = w.n() as u32;
    let k = w.k() as u32;
    let use_rot = n % 128 == 0 && k % 128 == 0 && k % 512 == 0 && k / 128 >= 2;
    if !use_rot {
        return Err(SynaptixError::Unsupported("mxfp8 prequant: форма вне rot-пути"));
    }
    let gk = GemmMxFp8Kernels::for_context(ctx)?;
    let packed_arc = w
        .packed_arc()
        .ok_or_else(|| SynaptixError::Cuda("mxfp8 prequant: packed W освобождён".into()))?;
    let w_packed = packed_arc
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("mxfp8 prequant: packed non-cuda"))?
        .slice();
    let w_scales = w
        .scales()
        .as_cuda()
        .ok_or(SynaptixError::Unsupported("mxfp8 prequant: scales non-cuda"))?
        .slice();
    let (m_us, n_us) = (m as usize, n as usize);
    let mut out_view = unsafe { out_u8.transmute_mut::<f16>(m_us * n_us) }
        .ok_or_else(|| SynaptixError::Cuda("mxfp8 prequant: transmute out→f16".into()))?;
    crate::best_cu::gemm::gemm_mxfp8::gemm_mxfp8_rot(
        &gk, stream, xq, w_packed, sa, w_scales, &mut out_view, m, n, k, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_mxfp8_gemm(
    gk: &GemmMxFp8Kernels,
    stream: &Arc<CudaStream>,
    xq: &CudaSlice<u8>,
    w_packed: &CudaSlice<u8>,
    sa: &CudaSlice<u8>,
    w_scales: &CudaSlice<u8>,
    out_u8: &mut CudaSlice<u8>,
    m: u32,
    m_pad: u32,
    n: u32,
    k: u32,
    use_rot: bool,
    aligned: bool,
) -> Result<bool> {
    let (m_us, n_us, mp_us) = (m as usize, n as usize, m_pad as usize);
    // GEMM: при m кратном 128 — НАПРЯМУЮ в out (без y-скретча и копии);
    // иначе y-скретч uninit (ядро пишет все m_pad×n) + копия первых m строк.
    // ROT (рецепт CUTLASS-порта nvfp4: TMA ::cta + ротация пар k32 + L2-растр +
    // staged-эпилог, выделенный producer-WG) — бьёт qutlass mxf8 на всех LTX-формах,
    // bit-exact к gn_mxfp8_128x128.
    if use_rot {
        let mut out_view = unsafe { out_u8.transmute_mut::<f16>(m_us * n_us) }
            .ok_or_else(|| SynaptixError::Cuda("mxfp8: transmute out→f16".into()))?;
        crate::best_cu::gemm::gemm_mxfp8::gemm_mxfp8_rot(
            &gk, stream, &xq, w_packed, &sa, w_scales, &mut out_view, m, n, k, true,
        )?;
    } else if aligned {
        let mut out_view = unsafe { out_u8.transmute_mut::<f16>(m_us * n_us) }
            .ok_or_else(|| SynaptixError::Cuda("mxfp8: transmute out→f16".into()))?;
        gemm_mxfp8(&gk, stream, &xq, w_packed, &sa, w_scales, &mut out_view, m_pad, n, k)?;
    } else {
        let mut y = unsafe { stream.ws_alloc::<f16>(mp_us * n_us) }
            .map_err(|e| SynaptixError::Cuda(format!("mxfp8: alloc y: {e:?}")))?;
        gemm_mxfp8(&gk, stream, &xq, w_packed, &sa, w_scales, &mut y.slice_mut(0..mp_us * n_us), m_pad, n, k)?;
        let mut out_view = unsafe { out_u8.transmute_mut::<f16>(m_us * n_us) }
            .ok_or_else(|| SynaptixError::Cuda("mxfp8: transmute out→f16".into()))?;
        let y_src = y.slice(0..m_us * n_us);
        stream
            .memcpy_dtod(&y_src, &mut out_view)
            .map_err(|e| SynaptixError::Cuda(format!("mxfp8: copy out: {e:?}")))?;
    }
    Ok(true)
}


// ─────────────────────────────────────────────────────────────────────────────
// Фолбэк prefill'а MXFP8 для форм вне tiled-ядра (N%128 != 0)
// ─────────────────────────────────────────────────────────────────────────────

/// Скретчи фолбэка: `w` — деквант-полоса весов в f16, `y` — узкий выход GEMM.
/// Держим по устройству и растим на месте (как [`Mxfp8Scratch`]): фолбэк
/// дёргается на каждом спекулятивном шаге, разовые аллокации там дороги.
struct Mxfp8DeqScratch {
    w: CudaSlice<u8>,
    y: CudaSlice<u8>,
}

static MXFP8_DEQ_SCRATCH: OnceLock<Mutex<HashMap<usize, Mxfp8DeqScratch>>> = OnceLock::new();

fn with_mxfp8_deq_scratch<R>(
    stream: &Arc<CudaStream>,
    ordinal: usize,
    w_need: usize,
    y_need: usize,
    f: impl FnOnce(&mut CudaSlice<u8>, &mut CudaSlice<u8>) -> Result<R>,
) -> Result<R> {
    let cache = MXFP8_DEQ_SCRATCH.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| SynaptixError::Cuda("mxfp8 deq scratch: poisoned".into()))?;
    let entry = match guard.entry(ordinal) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => {
            // uninit: деквант пишет всю полосу, GEMM — весь узкий выход.
            let w = unsafe { stream.cache_alloc_uninit::<u8>(w_need.max(1)) }
                .map_err(|e| SynaptixError::Cuda(format!("mxfp8 deq: alloc W: {e:?}")))?;
            let y = unsafe { stream.cache_alloc_uninit::<u8>(y_need.max(1)) }
                .map_err(|e| SynaptixError::Cuda(format!("mxfp8 deq: alloc Y: {e:?}")))?;
            e.insert(Mxfp8DeqScratch { w, y })
        }
    };
    if entry.w.len() < w_need {
        entry.w = unsafe { stream.cache_alloc_uninit::<u8>(w_need) }
            .map_err(|e| SynaptixError::Cuda(format!("mxfp8 deq: grow W: {e:?}")))?;
    }
    if entry.y.len() < y_need {
        entry.y = unsafe { stream.cache_alloc_uninit::<u8>(y_need) }
            .map_err(|e| SynaptixError::Cuda(format!("mxfp8 deq: grow Y: {e:?}")))?;
    }
    let Mxfp8DeqScratch { w, y } = entry;
    f(w, y)
}

/// Бюджет одной деквант-полосы в байтах — `SYN_MXFP8_DEQ_MB` (дефолт 64 МиБ).
fn mxfp8_deq_budget_bytes() -> usize {
    static B: OnceLock<usize> = OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("SYN_MXFP8_DEQ_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(64)
            * 1024
            * 1024
    })
}

/// Prefill MXFP8 (m>1) для форм, которые не берёт [`mxfp8_linear_tiled`]
/// (N%128 != 0 — например lm_head 202048×6656 у muse-glimmer).
///
/// Прямолинейный путь «деквант ВСЕГО W в f16 → f16-GEMM» стоит `N*K*2` байт
/// (2.5 ГиБ на этом lm_head) и вчетверо больше трафика, чем сам вес: на
/// заполненной карте он падал в `CUDA_ERROR_OUT_OF_MEMORY` на каждом
/// спекулятивном шаге (DFlash/lookup зовут голову с m>1). Здесь:
///   • голова `n_head = N - N%128` считается tiled-ядром по тем же указателям
///     веса (строки натуральные, префикс строк = префикс байт) в узкий
///     скретч `[m, n_head]`;
///   • остаток (<128 строк; при K%128 != 0 — весь N) дековантуется полосами
///     по [`mxfp8_deq_budget_bytes`] и умножается f16-GEMM'ом;
///   • куски сшиваются в `out` pitched-копиями (один cuMemcpy2D на кусок).
///
/// Пик памяти — `max(m*n_head, m*chunk)*2 + chunk*K*2` вместо `N*K*2`.
pub fn mxfp8_linear_dequant_fallback(
    qk: &Mxfp8QuantKernels,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x_u8: &CudaSlice<u8>,
    out_u8: &mut CudaSlice<u8>,
    w: &QuantWeight,
    m: u32,
) -> Result<()> {
    use crate::best_cu::gemm::gemm_bf16::{
        best_gemm_f16tn_linear_u8, copy_2d_bytes, BestGemmBf16Kernels,
    };
    use crate::elementwise::quant::mxfp8_dequant_f16;

    let n = w.n() as u32;
    let k = w.k() as u32;
    let (m_us, n_us, k_us) = (m as usize, n as usize, k as usize);
    let row_bytes = k_us * 2;
    if k % 32 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "mxfp8 fallback: K={k} должно быть кратно 32"
        )));
    }

    // Голова доступна, только если tiled-ядро вообще применимо по K.
    let n_head = if k % 128 == 0 { n - n % 128 } else { 0 };
    let rest = n - n_head;
    let chunk = if rest == 0 {
        0
    } else {
        ((mxfp8_deq_budget_bytes() / row_bytes.max(1)).max(1) as u32).min(rest)
    };
    let y_need = m_us * (n_head.max(chunk) as usize) * 2;
    let w_need = (chunk as usize) * row_bytes;

    with_mxfp8_deq_scratch(stream, ctx.ordinal(), w_need, y_need, |w_f16, y| {
        if n_head > 0 {
            if !mxfp8_linear_tiled_n(qk, ctx, stream, x_u8, y, w, m, n_head)? {
                return Err(SynaptixError::Cuda(format!(
                    "mxfp8 fallback: голова {n_head}×{k} не взялась tiled-ядром"
                )));
            }
            let head_bytes = (n_head as usize) * 2;
            copy_2d_bytes(stream, y, out_u8, m, head_bytes, head_bytes, n_us * 2, 0)?;
        }
        if rest > 0 {
            let bk = BestGemmBf16Kernels::for_context(ctx)?;
            let packed_arc = w.packed_arc().ok_or_else(|| {
                SynaptixError::Cuda("mxfp8 fallback: packed W освобождён".into())
            })?;
            let w_packed = packed_arc
                .as_cuda()
                .ok_or(SynaptixError::Unsupported("mxfp8 fallback: packed non-cuda"))?
                .slice();
            let w_scales = w
                .scales()
                .as_cuda()
                .ok_or(SynaptixError::Unsupported("mxfp8 fallback: scales non-cuda"))?
                .slice();
            let sc_row = k_us / 32;
            let mut n0 = n_head;
            while n0 < n {
                let cn = chunk.min(n - n0);
                let (o, cn_us) = (n0 as usize, cn as usize);
                {
                    let p = w_packed.slice(o * k_us..(o + cn_us) * k_us);
                    let sc = w_scales.slice(o * sc_row..(o + cn_us) * sc_row);
                    let mut wv = unsafe { w_f16.transmute_mut::<f16>(cn_us * k_us) }
                        .ok_or_else(|| {
                            SynaptixError::Cuda("mxfp8 fallback: transmute W→f16".into())
                        })?;
                    mxfp8_dequant_f16(qk, stream, &p, &sc, &mut wv, cn, k)?;
                }
                best_gemm_f16tn_linear_u8(&bk, stream, w_f16, x_u8, y, cn, k, m, None, None)?;
                copy_2d_bytes(stream, y, out_u8, m, cn_us * 2, cn_us * 2, n_us * 2, o * 2)?;
                n0 += cn;
            }
        }
        Ok(())
    })
}
