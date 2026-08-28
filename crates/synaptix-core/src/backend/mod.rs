pub mod attn;
pub mod registry;

use crate::device::Device;
use crate::error::{Result, SynaptixError};
use crate::stream::Stream;
use crate::tensor::layout::Layout;
use crate::tensor::quant::QuantWeight;
use crate::tensor::storage::Storage;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    /// Копия значения без арифметики. Нужна там, где важны сами байты:
    /// `x * 1 + 0` превращает `-0.0` в `+0.0`, поэтому strided-копия через
    /// affine молча меняла веса.
    Identity,
    Neg,
    Abs,
    Sqrt,
    Sqr,
    Recip,
    Exp,
    Log,
    Sin,
    Cos,
    Silu,
    GeluTanh,
    GeluExact,
    Tanh,
    Clamp(f32, f32),
    Powf(f32),
    Affine(f32, f32),
    Erf,
    Sigmoid,
    Relu,
    Relu2,
    LeakyRelu(f32),
    Sign,
    StepGtZero,
    Round,
    Floor,
    Ceil,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    ArgMax,
}

pub trait Backend: Send + Sync + 'static {
    fn device_kind(&self) -> Device;

    fn alloc_zeros(&self, n_bytes: usize, device: Device) -> Result<Storage>;

    fn alloc_uninit(&self, n_bytes: usize, device: Device) -> Result<Storage> {
        self.alloc_zeros(n_bytes, device)
    }

    fn copy(
        &self,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        stream: &Stream,
    ) -> Result<()>;

    fn cast(
        &self,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        stream: &Stream,
    ) -> Result<()>;

    fn unary(
        &self,
        op: UnaryOp,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        stream: &Stream,
    ) -> Result<()>;

    fn binary(
        &self,
        op: BinaryOp,
        a: (&Storage, &Layout),
        b: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        stream: &Stream,
    ) -> Result<()>;

    fn matmul(
        &self,
        lhs: (&Storage, &Layout),
        rhs: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        stream: &Stream,
    ) -> Result<()>;

    /// Квантованный Linear (hardware tensor-core): `out[M,N] = x[M,K] @ w[N,K]ᵀ`,
    /// где `w` — `QuantWeight` (NVFP4/MXFP8) со своими scale-тензорами. Активация
    /// `x` квантуется на лету. Default — `Unsupported` (CPU и прочие бэкенды),
    /// переопределяет CUDA.
    fn linear_quant(
        &self,
        _x: (&Storage, &Layout),
        _w: &QuantWeight,
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported(
            "linear_quant не поддержан этим backend",
        ))
    }

    /// Квантование плотного веса `w[N,K]` (F16) в NVFP4: возвращает `(packed,
    /// scales)` storages (E2M1 4-бит + E4M3 block scale, tile-major layout) для
    /// последующего `linear_quant`. One-time на загрузке. Default `Unsupported`.
    fn quantize_nvfp4(
        &self,
        _w: (&Storage, &Layout),
        _n: usize,
        _k: usize,
        _stream: &Stream,
    ) -> Result<(Storage, Storage)> {
        Err(SynaptixError::Unsupported(
            "quantize_nvfp4 не поддержан этим backend",
        ))
    }

    /// MXFP8 (Blackwell block-scale FP8): F16-вес `[n,k]` → e4m3 packed `[n,k]` +
    /// natural E8M0 scales `[n,k/32]`. Default — `Unsupported`.
    fn quantize_mxfp8(
        &self,
        _w: (&Storage, &Layout),
        _n: usize,
        _k: usize,
        _stream: &Stream,
    ) -> Result<(Storage, Storage)> {
        Err(SynaptixError::Unsupported(
            "quantize_mxfp8 не поддержан этим backend",
        ))
    }

    /// Плотный Linear: `out[M,N] = x[M,K] @ w[N,K]ᵀ`, где `w` — обычный тензор в
    /// натуральном [out, in] layout (как веса `nn::Linear`). Backend МОЖЕТ
    /// реализовать быстрый специализированный путь (например CUDA GEMV для M=1
    /// decode — без транспонирования веса и без dense-GEMM). Default —
    /// `Unsupported`, тогда `Tensor::linear` падает в `matmul(wᵀ)`.
    fn linear(
        &self,
        _x: (&Storage, &Layout),
        _w: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("linear не поддержан этим backend"))
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_epilogue(
        &self,
        _x: (&Storage, &Layout),
        _w: (&Storage, &Layout),
        _bias: Option<(&Storage, &Layout)>,
        _residual: Option<(&Storage, &Layout)>,
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported(
            "linear_epilogue не поддержан этим backend",
        ))
    }

    /// Fused RMSNorm по last dim: `out = x / sqrt(mean(x²)+eps) * w` (если
    /// `qwen_gain` — gain = `w + 1`). `x[.., H]`, `w[H]`. Один kernel-launch на
    /// строку с F32-аккумулятором. Default `Unsupported` → `Tensor::rms_norm_fused`
    /// падает в decomposed путь (`synaptix-ops`).
    fn rms_norm(
        &self,
        _x: (&Storage, &Layout),
        _w: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _eps: f32,
        _qwen_gain: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("rms_norm не поддержан этим backend"))
    }

    /// Квантует активацию `x` (F16 [m,k]) в NVFP4 packed + scales — отдельно от
    /// GEMV. Позволяет квантовать общий `h` 1× и переиспользовать во всех проекциях
    /// (`linear_quant_prequant`). `packed_out` [m,k/2] u8, `scales_out` u8.
    #[allow(clippy::too_many_arguments)]
    fn nvfp4_quantize_act(
        &self,
        _x: (&Storage, &Layout),
        _packed_out: (&mut Storage, &Layout),
        _scales_out: (&mut Storage, &Layout),
        _m: usize,
        _k: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("nvfp4_quantize_act не поддержан этим backend"))
    }

    fn silu_mul_quant_nvfp4(
        &self,
        _x: (&Storage, &Layout),
        _packed_out: (&mut Storage, &Layout),
        _scales_out: (&mut Storage, &Layout),
        _m: usize,
        _k: usize,
        _inv_pre: f32,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("silu_mul_quant_nvfp4 не поддержан этим backend"))
    }

    /// Fused ternary elementwise: kind 0 = gated-residual `out=x+b*c` (формы
    /// равны), 1 = то же с `c`-строкой `[D]`, 2 = adaLN-мод `out=x*(1+b)+c`
    /// (`b`/`c` строки `[D]`). Раунды повторяют decomposed → бит-в-бит.
    /// Default `Unsupported` (вызывающий падает на decomposed-цепочку).
    fn ternary_fused(
        &self,
        _kind: u8,
        _x: (&Storage, &Layout),
        _b: (&Storage, &Layout),
        _c: (&Storage, &Layout),
        _dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("ternary_fused не поддержан этим backend"))
    }

    /// Fused «adaLN-модуляция + NVFP4-квант» (эпилог нормы):
    /// `y = rms(x)·(1+scale)+shift` (бит-в-бит с decomposed-цепочкой) и
    /// `(packed, scales) = nvfp4_quant(f16(y))` одним launch. `x/scale/shift/y`
    /// `[m,k]` F16|BF16 (scale/shift по-токенные). Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn rms_mod_quant_nvfp4(
        &self,
        _x: (&Storage, &Layout),
        _scale: (&Storage, &Layout),
        _shift: (&Storage, &Layout),
        _y: (&mut Storage, &Layout),
        _packed_out: &mut Storage,
        _scales_out: &mut Storage,
        _m: usize,
        _k: usize,
        _eps: f32,
        // 0 = rms+модуляция (LTX), 1 = LN+модуляция (FLUX), 2 = rms·w (LLM),
        // 3 = rms·(1+w) (gemma-стиль).
        _kind: u8,
        _mod_div: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("rms_mod_quant_nvfp4 не поддержан этим backend"))
    }

    /// MXFP8-вариант [`Self::rms_mod_quant_nvfp4`]: тот же контракт нормы
    /// (kind 0..3), эпилог = mxfp8-квант (бит-в-бит с mxfp8-квантом активации).
    /// `packed_out` u8 `[m·k]`, `scales_out` natural u8 `[m·k/32]`.
    #[allow(clippy::too_many_arguments)]
    fn rms_mod_quant_mxfp8(
        &self,
        _x: (&Storage, &Layout),
        _scale: (&Storage, &Layout),
        _shift: (&Storage, &Layout),
        _y: (&mut Storage, &Layout),
        _packed_out: &mut Storage,
        _scales_out: &mut Storage,
        _m: usize,
        _k: usize,
        _eps: f32,
        _kind: u8,
        _mod_div: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("rms_mod_quant_mxfp8 не поддержан этим backend"))
    }

    /// Квантует активацию `x` (F16 [m,k]) в MXFP8 packed + natural scales —
    /// для шаринга между проекциями (`linear_quant_prequant` c MXFP8-весом).
    /// `packed_out` u8 [m,k], `scales_out` u8 [m,k/32].
    #[allow(clippy::too_many_arguments)]
    fn mxfp8_quantize_act(
        &self,
        _x: (&Storage, &Layout),
        _packed_out: (&mut Storage, &Layout),
        _scales_out: (&mut Storage, &Layout),
        _m: usize,
        _k: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("mxfp8_quantize_act не поддержан этим backend"))
    }

    /// GEMM/GEMV из УЖЕ квантованной активации (`packed_x`/`scales_x` от
    /// [`Self::nvfp4_quantize_act`] | [`Self::mxfp8_quantize_act`] — формат по
    /// `w.dtype()`). Пропускает повторное квантование. `out` [m, w.n()] f16.
    #[allow(clippy::too_many_arguments)]
    fn linear_quant_prequant(
        &self,
        _packed_x: &Storage,
        _scales_x: &Storage,
        _w: &QuantWeight,
        _out: (&mut Storage, &Layout),
        _m: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("linear_quant_prequant не поддержан этим backend"))
    }

    /// Fused residual+RMSNorm (out-of-place): `hidden_out = x + residual;
    /// y = RMSNorm(hidden_out) * weight`. Заменяет 2 launch'а (add + rms_norm) на
    /// 1 и один memory-pass по hidden. `x`, `residual` `[batch, hidden]`, `weight`
    /// `[hidden]`; `hidden_out`, `y` `[batch, hidden]`. `qwen_gain` → weight=(1+w).
    /// Default `Unsupported` → `Tensor::rms_norm_residual_fused` падает в decomposed.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_residual(
        &self,
        _x: (&Storage, &Layout),
        _residual: (&Storage, &Layout),
        _w: (&Storage, &Layout),
        _hidden_out: (&mut Storage, &Layout),
        _y: (&mut Storage, &Layout),
        _eps: f32,
        _qwen_gain: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("rms_norm_residual не поддержан этим backend"))
    }

    /// Fused pointwise `out = silu(gate) * up`. Один kernel-launch вместо двух
    /// (silu unary → mul binary), 3 trip'а памяти вместо 4. Все три тензора
    /// одинакового dtype/shape, contiguous. Default `Unsupported` →
    /// `Tensor::silu_and_mul` падает в decomposed путь.
    fn silu_and_mul(
        &self,
        _gate: (&Storage, &Layout),
        _up: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("silu_and_mul не поддержан этим backend"))
    }

    /// Fused LayerNorm по last dim: `out = ((x - mean)/sqrt(var+eps)) * w [+ bias]`.
    /// `x[.., H]`, `w[H]`, `bias` опциональный `[H]`. Один kernel-launch на строку с
    /// F32-аккумулятором (вместо ~12 decomposed-ops). Default `Unsupported` →
    /// `Tensor::layer_norm_fused` падает в decomposed путь (`synaptix-ops`).
    fn layer_norm(
        &self,
        _x: (&Storage, &Layout),
        _w: (&Storage, &Layout),
        _bias: Option<(&Storage, &Layout)>,
        _out: (&mut Storage, &Layout),
        _eps: f32,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("layer_norm не поддержан этим backend"))
    }

    /// Fused Split (GPT-NeoX) RoPE: `x[.., S, D]`, `cos`/`sin` — F32 `[S, D/2]`
    /// (как из `RopeCache::select_*`). Ротация в F32, выход типа `x`. Один
    /// kernel-launch вместо ~12 decomposed-ops. Default `Unsupported`.
    fn rope_split(
        &self,
        _x: (&Storage, &Layout),
        _cos: (&Storage, &Layout),
        _sin: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("rope_split не поддержан этим backend"))
    }

    /// Partial Split RoPE: вращает первые `rot_dim` из `D`, остальные измерения
    /// проходят без изменений. `x[.., S, D]`, `cos`/`sin` — F32 `[S, rot_dim/2]`.
    /// Позиция строки = `row % S`, что даёт broadcast по головам при layout
    /// `[H, S, D]`. Default `Unsupported`.
    fn rope_split_partial(
        &self,
        _x: (&Storage, &Layout),
        _cos: (&Storage, &Layout),
        _sin: (&Storage, &Layout),
        _rot_dim: usize,
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("rope_split_partial не поддержан этим backend"))
    }

    /// Interleaved (adjacent-pair / FLUX use_real_unbind_dim=-1) RoPE одним ядром.
    /// `x`/`out` [B,S,H,D]; `cos`/`sin` — F32 ПОЛНАЯ таблица [S,D]. `h` = число
    /// голов (позиция = (row/h)%S). Заменяет ~10 decomposed-ops. Default Unsupported.
    fn rope_interleaved(
        &self,
        _x: (&Storage, &Layout),
        _cos: (&Storage, &Layout),
        _sin: (&Storage, &Layout),
        _h: usize,
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("rope_interleaved не поддержан этим backend"))
    }

    /// Fused scaled-dot-product attention: `out = softmax(scale·Q·Kᵀ [+causal])·V`.
    /// `q` [B,NH,Tq,D], `k`/`v` [B,NKV,Tkv,D] (GQA — `nh % nkv == 0`, ядро само
    /// расширяет KV), `out` [B,NH,Tq,D]. F32-аккумулятор, online-softmax. Подходит
    /// и для decode (Tq=1) и для causal prefill. Default `Unsupported`.
    fn flash_attention(
        &self,
        _q: (&Storage, &Layout),
        _k: (&Storage, &Layout),
        _v: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _scale: f32,
        _causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("flash_attention не поддержан этим backend"))
    }

    /// Двунаправленный sliding-window flash (band ±window). Default Unsupported
    /// → caller fallback (наивная маска).
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_window(
        &self,
        _q: (&Storage, &Layout),
        _k: (&Storage, &Layout),
        _v: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _scale: f32,
        _window: i32,
        _causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("flash_attention_window не поддержан этим backend"))
    }

    #[allow(clippy::too_many_arguments)]
    fn flash_attention_window_dev(
        &self,
        _q: (&Storage, &Layout),
        _k: (&Storage, &Layout),
        _v: (&Storage, &Layout),
        _t_cache: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _scale: f32,
        _window: i32,
        _causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("flash_attention_window_dev не поддержан этим backend"))
    }

    /// Flash-attention в layout [B,S,H,D] (head-минорный) — image-attn без
    /// permute+contiguous транспоза. Default Unsupported → caller fallback.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_bshd(
        &self,
        _q: (&Storage, &Layout),
        _k: (&Storage, &Layout),
        _v: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _scale: f32,
        _causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("flash_attention_bshd не поддержан этим backend"))
    }

    /// In-place scatter-write нового `src` `[B, nkv, T_new, hd]` (contiguous) в
    /// preallocated `dst` `[B, nkv, max_seq, hd]` (contiguous) на позицию
    /// `seq_pos` по dim-T. Заменяет `Tensor::cat` для KV-кеша (O(1) запись вместо
    /// O(S) реаллокации+копии всего буфера). `max_seq` берётся из `dst.dims()[2]`.
    /// Default `Unsupported`.
    fn kv_append(
        &self,
        _dst: (&mut Storage, &Layout),
        _src: (&Storage, &Layout),
        _seq_pos: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("kv_append не поддержан этим backend"))
    }

    /// Device-resident-position Split RoPE (для CUDA-graph decode). Как
    /// [`Backend::rope_split`], но: `cos`/`sin` — в dtype `x` (не F32) и в
    /// *дублированном* layout `[max_seq, rotary_dim]` (ядро индексирует
    /// `cos[pos*rotary_dim + d]`); активная позиция приходит device-резидентным
    /// `start_pos` `(&Storage,&Layout)` U32[1] — launch config от значения не
    /// зависит → один граф валиден для всех decode-позиций. `x`/`out`
    /// `[b,h,t,head_dim]` (contiguous). Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn rope_apply_dev(
        &self,
        _x: (&Storage, &Layout),
        _cos: (&Storage, &Layout),
        _sin: (&Storage, &Layout),
        _start_pos: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _rotary_dim: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("rope_apply_dev не поддержан этим backend"))
    }

    /// Device-resident-position in-place KV append (для CUDA-graph decode). Как
    /// [`Backend::kv_append`], но слот пишется по device-резидентной позиции
    /// `seq_pos` `(&Storage,&Layout)` U32[1] (вместо immediate `usize`) — один
    /// граф валиден для всех decode-позиций. Default `Unsupported`.
    fn kv_append_dev(
        &self,
        _dst: (&mut Storage, &Layout),
        _src: (&Storage, &Layout),
        _seq_pos: (&Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("kv_append_dev не поддержан этим backend"))
    }

    /// Device-resident-length flash-decode (для CUDA-graph decode). Как
    /// [`Backend::flash_attention`], но активная длина KV приходит
    /// device-резидентным `t_cache` `(&Storage,&Layout)` U32[1]; `k`/`v` —
    /// strided-view preallocated буфера `[B,nkv,max_seq,hd]` (физический
    /// `t_stride` выводится из layout). Tq обычно = 1. Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_dev(
        &self,
        _q: (&Storage, &Layout),
        _k: (&Storage, &Layout),
        _v: (&Storage, &Layout),
        _t_cache: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _scale: f32,
        _causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("flash_attention_dev не поддержан этим backend"))
    }

    /// Device-resident-length FA-prefill (для CUDA-graph prefill chunk'а). Как
    /// [`Backend::flash_attention_dev`], но оптимизирован под Tq>1: tensor-core
    /// FA-4 ядро с Q-тайлингом по `BM=16` (WMMA m16n8k16), нет split-K (split-K
    /// — анти-паттерн для prefill, теряется reuse Q-тайла). `t_cache` `U32[1]` —
    /// активная длина KV (для chunk'а на абсолютной позиции `pos_start`:
    /// `t_cache = pos_start + Tq`); `q_pos[ti] = t_cache - Tq + ti = pos_start + ti`.
    /// Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_prefill_dev(
        &self,
        _q: (&Storage, &Layout),
        _k: (&Storage, &Layout),
        _v: (&Storage, &Layout),
        _t_cache: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _scale: f32,
        _causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported(
            "flash_attention_prefill_dev не поддержан этим backend",
        ))
    }

    /// Device-pos MXFP8-KV квантизующий append (CUDA-graph): как
    /// [`Backend::kv_append_quant_mxfp8`], но `seq_pos` device-резидентный U32[1].
    fn kv_append_quant_mxfp8_dev(
        &self,
        _dst: (&mut Storage, &Layout),
        _scale_dst: (&mut Storage, &Layout),
        _src: (&Storage, &Layout),
        _seq_pos: (&Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("kv_append_quant_mxfp8_dev не поддержан этим backend"))
    }

    /// Device-Tkv MXFP8-KV flash-decode (CUDA-graph): как
    /// [`Backend::flash_attention_mxfp8kv`], но активная длина KV `t_cache` U32[1]
    /// device-резидентна. `k`/`v` MXFP8 strided-view; `k_scale`/`v_scale` U8
    /// `[B,nkv,max_seq,hd/32]`. Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_mxfp8kv_dev(
        &self,
        _q: (&Storage, &Layout),
        _k: (&Storage, &Layout),
        _v: (&Storage, &Layout),
        _k_scale: (&Storage, &Layout),
        _v_scale: (&Storage, &Layout),
        _t_cache: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _scale: f32,
        _causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("flash_attention_mxfp8kv_dev не поддержан этим backend"))
    }

    /// Token-embedding gather: `out[t,:] = table[ids[t],:]`. `table` `[vocab,dim]`,
    /// `ids` `[n]` U32 (читаются с **device** — без host round-trip, в отличие от
    /// `index_select`, который `clone_dtoh`'ит индексы и ломает CUDA-graph capture),
    /// `out` `[n,dim]`. Default `Unsupported` → `Tensor::embed_gather` падает в
    /// `index_select`-путь. OOB id → строка нулей.
    fn embed_gather(
        &self,
        _table: (&Storage, &Layout),
        _ids: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("embed_gather не поддержан этим backend"))
    }

    /// Перемешать NVFP4-вес в раскладку, которую читают GEMV/GEMM-ядра.
    /// `packed` `[n, k]` → `out` того же размера. Обычно это делает первое
    /// умножение; отдельный вызов нужен, чтобы подготовить веса заранее.
    fn nvfp4_repack(
        &self,
        _packed: &Storage,
        _out: &mut Storage,
        _n: usize,
        _k: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("nvfp4_repack не поддержан этим backend"))
    }

    /// Батч NVFP4-GEMV: `out[e, :] = W_e · x_e`, все веса формы `[n, k]` в
    /// перемешанной раскладке. `x_rows` — номер строки активации в её кванте:
    /// батч умеет читать разные строки одного общего буфера, как выходит после
    /// фьюза swiglu, посчитанного на всех экспертах разом. Нужен MoE-декоду: десяток матриц по одной
    /// строке каждая отдельными запусками упирается в launch overhead.
    /// Default `Unsupported` → вызывающий считает эксперты по одному.
    #[allow(clippy::too_many_arguments)]
    fn nvfp4_gemv_batched(
        &self,
        _w_shuf: &[&Storage],
        _w_scales: &[&Storage],
        _x_packed: &[&Storage],
        _x_scales: &[&Storage],
        _x_rows: &[usize],
        _out: (&mut Storage, &Layout),
        _n: usize,
        _k: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("nvfp4_gemv_batched не поддержан этим backend"))
    }

    fn embed_gather_mxfp8(
        &self,
        _table: &Storage,
        _scales: &Storage,
        _ids: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _vocab: usize,
        _dim: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("embed_gather_mxfp8 не поддержан этим backend"))
    }

    /// MXFP8-KV квантизующий append: `dst` — MXFP8 `[B,nkv,max_seq,hd]` (E4M3
    /// mantissa, 1 байт/элем), `scale_dst` — U8 `[B,nkv,max_seq,hd/32]` (E8M0
    /// per-32-block), `src` — BF16 `[B,nkv,T_new,hd]`. Квантизует `src` (per-32-
    /// block amax→E8M0) в slot `seq_pos`. Default `Unsupported`.
    fn kv_append_quant_mxfp8(
        &self,
        _dst: (&mut Storage, &Layout),
        _scale_dst: (&mut Storage, &Layout),
        _src: (&Storage, &Layout),
        _seq_pos: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("kv_append_quant_mxfp8 не поддержан этим backend"))
    }

    /// MXFP8-KV flash-attention: `out = softmax(scale·Q·Kᵀ [+causal])·V` с MXFP8
    /// K/V (E4M3) и per-32-block E8M0 scale (деквант inline). `q`/`out` — float
    /// `[B,NH,Tq,D]`; `k`/`v` — MXFP8 `[B,NKV,Tkv,D]`; `k_scale`/`v_scale` — U8
    /// `[B,NKV,Tkv,D/32]`. Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn flash_attention_mxfp8kv(
        &self,
        _q: (&Storage, &Layout),
        _k: (&Storage, &Layout),
        _v: (&Storage, &Layout),
        _k_scale: (&Storage, &Layout),
        _v_scale: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _scale: f32,
        _causal: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("flash_attention_mxfp8kv не поддержан этим backend"))
    }

    /// Device-резидентный decode-шаг (T=1) GatedDeltaNet linear-attn слоя (для
    /// CUDA-graph). Связывает stateful conv1d-update + prep + gated-delta-rule +
    /// RmsNormGated в одну capture-safe последовательность. `qkv`/`conv_w`/`a`/`b`/
    /// `z`/`norm_w` — F16; `dt_bias`/`a_log`/`ssm_state` — F32; `conv_state` — F16;
    /// `out` — F16 `[1,value_dim]`. `conv_state` `[(K-1),conv_dim]` и `ssm_state`
    /// `[num_v,dk,dv]` обновляются in-place (стабильные указатели persistent-буфера
    /// → один граф валиден для всех decode-шагов). Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn linear_attn_decode_step(
        &self,
        _qkv: (&Storage, &Layout),
        _conv_w: (&Storage, &Layout),
        _a: (&Storage, &Layout),
        _b: (&Storage, &Layout),
        _dt_bias: (&Storage, &Layout),
        _a_log: (&Storage, &Layout),
        _z: (&Storage, &Layout),
        _norm_w: (&Storage, &Layout),
        _conv_state: (&mut Storage, &Layout),
        _ssm_state: (&mut Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _num_k: usize,
        _num_v: usize,
        _dk: usize,
        _dv: usize,
        _conv_kernel: usize,
        _q_scale: f32,
        _eps: f32,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("linear_attn_decode_step не поддержан этим backend"))
    }

    /// Full chunked linear-attn prefill (T≥1) одним device-резидентным вызовом:
    /// `causal_conv1d_chunk` (+ silu) → `linear_attn_prep_scatter` (qe/ke/vv +
    /// g + β) → `chunk_gated_delta_rule`. Замена host-mix-блока
    /// `LinearAttn::forward` (model.rs:879-915) для CUDA-пути.
    ///
    /// Dtypes:
    /// - `qkv` `[1,T,conv_dim]`, `conv_w` `[conv_dim,K]`, `conv_state`
    ///   `[(K-1),conv_dim]` — общий compute-dtype (F16/BF16/F32);
    /// - `a`/`b` `[1,T,num_v]` — F16 (как в decode-пути);
    /// - `dt_bias`/`a_log` `[num_v]` — F32;
    /// - `ssm_state` `[num_v,hk,hv]` (in/out), `out` `[num_v,T,hv]` — F32.
    ///
    /// `conv_state` обновляется in-place; `out` должен быть `[num_v, t_pad, hv]`
    /// (caller аллоцирует с padding до `t_pad % chunk_size == 0` и сам narrow'ит
    /// до `t_in`). Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn linear_attn_chunk_prefill(
        &self,
        _qkv: (&Storage, &Layout),
        _conv_w: (&Storage, &Layout),
        _a: (&Storage, &Layout),
        _b: (&Storage, &Layout),
        _dt_bias: (&Storage, &Layout),
        _a_log: (&Storage, &Layout),
        _conv_state: (&mut Storage, &Layout),
        _ssm_state: (&mut Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _num_k: usize,
        _num_v: usize,
        _hk: usize,
        _hv: usize,
        _conv_kernel: usize,
        _t_in: usize,
        _t_pad: usize,
        _chunk_size: usize,
        _q_scale: f32,
        _silu: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported(
            "linear_attn_chunk_prefill не поддержан этим backend",
        ))
    }

    /// Chunked Gated-DeltaNet linear-attn prefill (T>1) на device. Заменяет
    /// рекуррентный host-скан: `q`/`k` `[BH,T,HK]`, `v` `[BH,T,HV]`, `g`/`beta`
    /// `[BH,T]` (g = log-decay, beta = post-sigmoid), `ssm_state` `[BH,HK,HV]`
    /// (F32, in/out), `out` `[BH,T,HV]` — все F32, contiguous, offset 0. q/k
    /// L2-нормализуются внутри, q·=`q_scale`, decay копится пер-чанково.
    /// Требует `T % cs == 0` (caller паддит). Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_rule_prefill(
        &self,
        _q: (&Storage, &Layout),
        _k: (&Storage, &Layout),
        _v: (&Storage, &Layout),
        _g: (&Storage, &Layout),
        _beta: (&Storage, &Layout),
        _ssm_state: (&mut Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _q_scale: f32,
        _bh: usize,
        _t: usize,
        _hk: usize,
        _hv: usize,
        _cs: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("gated_delta_rule_prefill не поддержан этим backend"))
    }

    /// Direct conv2d (один launch вместо im2col-через-cat decomposed-пути):
    /// `out[B,C_out,H_out,W_out] = conv(input[B,C_in,H,W], weight[C_out,C_in,Kh,Kw])
    /// [+ bias[C_out]]`. F32-аккумулятор. Padding/stride применяются внутри ядра
    /// (без материализации padded-входа). Только dilation=1 и contiguous
    /// input/weight; иначе `Unsupported` → `synaptix-ops::conv2d` падает в generic.
    /// Default `Unsupported` (CPU остаётся на generic-пути). `out_h`/`out_w`
    /// рассчитываются caller'ом.
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        _input: (&Storage, &Layout),
        _weight: (&Storage, &Layout),
        _bias: Option<(&Storage, &Layout)>,
        _out: (&mut Storage, &Layout),
        _stride: (usize, usize),
        _padding: (usize, usize),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("conv2d не поддержан этим backend"))
    }

    /// Direct conv3d (один thread = один output voxel, F32-аккумулятор). `input`
    /// `[B,C_in,D,H,W]`, `weight` `[C_out,C_in,Kd,Kh,Kw]`, `out`
    /// `[B,C_out,D_out,H_out,W_out]` — все contiguous, dilation=1. Default
    /// `Unsupported` → `synaptix-ops::conv3d` падает в decomposed путь (CPU).
    #[allow(clippy::too_many_arguments)]
    fn conv3d(
        &self,
        _input: (&Storage, &Layout),
        _weight: (&Storage, &Layout),
        _bias: Option<(&Storage, &Layout)>,
        _out: (&mut Storage, &Layout),
        _stride: (usize, usize, usize),
        _padding: (usize, usize, usize),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("conv3d не поддержан этим backend"))
    }

    /// Depthwise conv1d (groups == C, weight `[C,1,K]`): `transpose=false` —
    /// stride-свёртка с zero-pad; `true` — conv_transpose полной длины
    /// `(L−1)·s+K` (кроп по padding у вызывающего). Один поток = один выходной
    /// элемент (раньше каналный Rust-цикл = C×K микро-launch'ей — вокодер LTX).
    /// Default `Unsupported` → `synaptix-ops` остаётся на decompose-пути (CPU).
    #[allow(clippy::too_many_arguments)]
    fn dwconv1d(
        &self,
        _input: (&Storage, &Layout),
        _weight: (&Storage, &Layout),
        _bias: Option<(&Storage, &Layout)>,
        _out: (&mut Storage, &Layout),
        _stride: usize,
        _padding: usize,
        _transpose: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("dwconv1d не поддержан этим backend"))
    }

    /// im2col (row-tiled): `input[B,C_in,H,W]` (contiguous) → `col[m_count,K]`,
    /// `K = C_in*Kh*Kw`; логическая строка `r` ↔ глобальная `m = m_offset + r`
    /// (`m → (b,ho,wo)`). Питает conv2d-через-GEMM (`out[m,C_out] = col @ Wᵀ`).
    /// Тайлинг по `m` ограничивает память col на больших spatial. `B/C_in/H/W` из
    /// `input` layout; `col` `[m_count,K]` аллоцирует caller. Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn im2col(
        &self,
        _input: (&Storage, &Layout),
        _col: (&mut Storage, &Layout),
        _kh: usize,
        _kw: usize,
        _h_out: usize,
        _w_out: usize,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _m_offset: u64,
        _m_count: u64,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("im2col не поддержан этим backend"))
    }

    /// Fused GroupNorm: `out = ((x-mean)/sqrt(var+eps))*weight [+bias]`, mean/var
    /// по `(C/num_groups каналов × spatial)` на (batch, group). `x`/`out`
    /// `[B,C,*spatial]` (contiguous), `weight`/`bias` опц. `[C]`. Один launch с
    /// F32-аккумулятором (вместо ~12 decomposed-ops + multi-dim reduce). `bias`
    /// без `weight` не поддержан. Default `Unsupported` → `synaptix-ops::group_norm`
    /// падает в decomposed путь.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn group_norm(
        &self,
        _x: (&Storage, &Layout),
        _weight: Option<(&Storage, &Layout)>,
        _bias: Option<(&Storage, &Layout)>,
        _out: (&mut Storage, &Layout),
        _num_groups: usize,
        _eps: f32,
        _silu: bool,
        _nhwc: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("group_norm не поддержан этим backend"))
    }

    /// Fused PixelNorm (+опц. silu): `out = x / sqrt(mean_c(x²)+eps)` per-location
    /// по канальной оси NCHW. `x`/`out` — `[B,C,S]` логически (S = прод. spatial,
    /// contiguous). Заменяет decomposed cast(f32)→sqr→mean→sqrt→div→cast(+silu).
    /// Default `Unsupported` → caller падает в decomposed путь.
    fn pixel_norm(
        &self,
        _x: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _c: usize,
        _eps: f32,
        _silu: bool,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("pixel_norm не поддержан этим backend"))
    }

    /// Fast 4D permute NCHW [B,C,H,W] → NHWC [B,H,W,C] через shmem-tile.
    /// Заменяет generic permute()+contiguous() для входов implicit-GEMM.
    /// Default `Unsupported`.
    fn nchw_to_nhwc(
        &self,
        _src: (&Storage, &Layout),
        _dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("nchw_to_nhwc не поддержан этим backend"))
    }

    /// Обратный fast-permute NHWC [B,H,W,C] → NCHW [B,C,H,W] (shmem-tile).
    /// Default `Unsupported`.
    fn nhwc_to_nchw(
        &self,
        _src: (&Storage, &Layout),
        _dst: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("nhwc_to_nchw не поддержан этим backend"))
    }

    /// CUTLASS Implicit-GEMM conv2d (NHWC, cuDNN-стиль). `input_nhwc[N,H,W,C]`,
    /// `filter_krsc[K,R,S,C]`, `out_nhwc[N,P,Q,K]` — все contiguous. F16/BF16
    /// только. Устраняет im2col K-кратное раздутие памяти (361мс ncu для SDXL).
    /// Default `Unsupported` → caller падает в im2col-путь.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_implicit_nhwc(
        &self,
        _input_nhwc: (&Storage, &Layout),
        _filter_krsc: (&Storage, &Layout),
        _bias: Option<(&Storage, &Layout)>,
        _residual: Option<(&Storage, &Layout)>,
        _temb: Option<(&Storage, &Layout)>,
        _out: (&mut Storage, &Layout),
        _out_nhwc: bool,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("conv2d_implicit_nhwc не поддержан этим backend"))
    }

    /// Fused conv2d-эпилог: `out2d[B*H*W, C]` (NHWC-flat) + опц. `bias[C]` +
    /// опц. `residual[B,C,H,W]` + опц. `temb_bc[B,C]` (per-(b,c) broadcast,
    /// для resnet time-embedding) → `out[B, C, H, W]` (NCHW). Заменяет
    /// `broadcast_add(bias) + broadcast_add(temb[:,:,None,None]) +
    /// permute.contiguous + add(residual)` (до 4 проходов → один).
    /// Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn conv_epilogue(
        &self,
        _out2d: (&Storage, &Layout),
        _bias: Option<(&Storage, &Layout)>,
        _residual: Option<(&Storage, &Layout)>,
        _temb_bc: Option<(&Storage, &Layout)>,
        _out: (&mut Storage, &Layout),
        _b: usize,
        _c: usize,
        _h: usize,
        _w: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("conv_epilogue не поддержан этим backend"))
    }

    /// Fused GEGLU split-activation: `inp[.., 2*I]` → `out[.., I]`,
    /// `out[t,i] = inp[t,i] * gelu_exact(inp[t, I+i])`. Один проход вместо
    /// narrow×2 + contiguous×2 + gelu + mul. Default `Unsupported`.
    fn geglu_split(
        &self,
        _inp: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("geglu_split не поддержан этим backend"))
    }

    /// Fused Snake-активация (Oobleck VAE): `out = x + sin(a[c]*x)^2 * binv[c]`
    /// по канальной оси. `a`/`binv` — предвычисленные per-channel `[C]` f32
    /// (`a=exp(alpha)`, `binv=1/(exp(beta)+eps)`). `x`/`out` — `[..,C,t_len]`
    /// contiguous (channel = `(i / t_len) % c`). Заменяет decomposed
    /// exp/mul/sin/sqr/recip/add (≈5 крупных проходов → 1). Default `Unsupported`.
    #[allow(clippy::too_many_arguments)]
    fn snake(
        &self,
        _x: (&Storage, &Layout),
        _a: (&Storage, &Layout),
        _binv: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _c: usize,
        _t_len: usize,
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("snake не поддержан этим backend"))
    }

    /// nearest-2x upsample: `input[B,C,H,W]` (contiguous) → `out[B,C,2H,2W]`.
    /// `out[b,c,ho,wo] = input[b,c,ho/2,wo/2]`. Один launch (заменяет медленный
    /// cat-based upsample). `B/C/H/W` из `input` layout; `out` аллоцирует caller.
    /// Default `Unsupported`.
    fn upsample_nearest2x(
        &self,
        _input: (&Storage, &Layout),
        _out: (&mut Storage, &Layout),
        _stream: &Stream,
    ) -> Result<()> {
        Err(SynaptixError::Unsupported("upsample_nearest2x не поддержан этим backend"))
    }

    fn reduce(
        &self,
        op: ReduceOp,
        src: (&Storage, &Layout),
        dst: (&mut Storage, &Layout),
        dims: &[usize],
        stream: &Stream,
    ) -> Result<()>;
}
