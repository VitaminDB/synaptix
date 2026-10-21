//! Операции обучения (обратные проходы, слитый лосс, оптимизатор) — одной
//! точкой входа [`crate::backend::Backend::train_op`]: входы и выходы идут
//! списками, а смысл задаёт [`TrainOp`]. Так новые ядра обучения не плодят
//! по методу backend'а на каждое.
//!
//! Обратные проходы считают в F32: градиент по остаточному потоку копится в
//! F32, в compute-типе остаются только сохранённые прямым проходом входы.

/// Что посчитать. Описание входов и выходов — у каждого варианта; все
/// тензоры contiguous.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainOp {
    /// RMSNorm с замороженным весом: входы `[x [R,H], w [H], dy [R,H] F32]`
    /// (`x` и `w` одного типа F32/BF16/F16), выход `[dx [R,H] F32]`.
    RmsNormBwd { eps: f32 },
    /// `silu(gate)·up`: входы `[gate, up, dy F32]` (gate/up одного типа),
    /// выходы `[dgate F32, dup F32]`.
    SiluMulBwd,
    /// Cross-entropy прямым и обратным проходом разом: входы
    /// `[logits [R,V], labels I32 [R]]` (метка < 0 — без лосса), выходы
    /// `[loss F32 [R], dlogits [R,V]]` (F32 или BF16); градиент умножен на
    /// `scale` (обычно 1/число токенов с лоссом).
    CrossEntropy { scale: f32 },
    /// Каузальный softmax скоров внимания: входы `[scores F32 [B,T,T]]` и,
    /// если `has_seg`, `seg I32 [T]` — номер примера в упаковке (ключи чужого
    /// примера скрыты); выход `[probs F32 [B,T,T]]`. Скоры умножаются на
    /// `scale` до softmax.
    CausalSoftmax { scale: f32, has_seg: bool },
    /// Обратный softmax по строкам: входы `[p, dp]` F32, выход
    /// `[ds = scale·p·(dp − Σdp·p)]` F32.
    SoftmaxBwd { scale: f32 },
    /// Деквант NVFP4-веса из перемешанной раскладки (`nvfp4_w_repack`):
    /// входы `[shuffled U8, scales U8]`, выход `[w BF16 [n,k]]`.
    Nvfp4DequantShuffled { n: usize, k: usize },
    /// Шаг AdamW: входы `[p, g, m, v]` F32 одной длины, выходы `[p', m', v']`.
    /// `bc1`/`bc2` — поправки смещения `1 − βᵗ`.
    AdamW { lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32, bc1: f32, bc2: f32 },
    /// LoRA, сжатие: входы `[x [T,K] (F32/BF16), w [r,K] F32]`, выход
    /// `[x·wᵀ [T,r] F32]`. `r ≤ 64`.
    LoraDown,
    /// LoRA, разжатие с прибавлением: входы `[base [T,N], z [T,r] F32,
    /// m [N,r] F32]`, выход `[base + scale·z·mᵀ]` того же типа, что `base`
    /// (F32/BF16).
    LoraUpAdd { scale: f32 },
    /// LoRA, градиент: входы `[a [T,r] F32, b [T,K] (F32/BF16)]`, выход —
    /// частичные суммы `[P, r, K]` F32 по кускам `T` (`P = ⌈T/128⌉`), их
    /// сумма по первой оси — `aᵀ·b`.
    LoraOuter,
    /// Flash-attention (обучение), прямой проход: входы `[q [H,T,D],
    /// k [Hkv,T,D], v [Hkv,T,D]]` BF16 и, если `has_seg`, `seg I32 [T]`;
    /// выходы `[o BF16 [H,T,D], lse F32 [H,T]]`. Каузально, внутри примера.
    /// `D` — 64, 128 или 256.
    FlashFwd { scale: f32, has_seg: bool },
    /// Flash-attention, обратный проход: входы `[q, k, v, dout BF16 [H,T,D],
    /// lse F32 [H,T], delta F32 [H,T] (= Σ dout∘o по D)]` и `seg`, если
    /// `has_seg`; выходы `[dq F32 [H,T,D], dk F32 [Hkv,Tp,D], dv F32
    /// [Hkv,Tp,D]]`, `Tp = ⌈T/32⌉·32` (хвост за `T` — мусор, отрезается).
    FlashBwd { scale: f32, has_seg: bool },
    /// GDN, каузальная depthwise-свёртка: входы `[x [T,C] (F32/BF16),
    /// w [C,K] F32]` и `seg`, если `has_seg` (свёртка не заходит в соседний
    /// пример); выход `[y [T,C] F32]`.
    GdnConvFwd { has_seg: bool },
    /// Обратная свёртка по входу: входы `[dy [T,C] F32, w [C,K] F32]` и
    /// `seg`; выход `[dx [T,C] F32]`.
    GdnConvBwd { has_seg: bool },
    /// GDN-рекуррентность, прямой проход: входы `[q, k [T,Hk,128], v [T,Hv,dv],
    /// g, beta [T,Hv]]` F32 и `seg`; выходы `[o [T,Hv,dv], ckpt [Hv, ⌈T/64⌉,
    /// 128, dv]]` F32. `dv` кратно 32.
    GdnFwd { has_seg: bool },
    /// GDN, обратный проход: входы `[q, k, v, g, beta, ckpt, dout [T,Hv,dv]]` и
    /// `seg`; выходы `[dq_p, dk_p [P,T,Hv,128], dg_p, db_p [P,T,Hv], dv
    /// [T,Hv,dv], scratch [P·Hv·64·128·32]]` F32, `P = dv/32` — частичные суммы
    /// по блокам столбцов (их сумма по первой оси — градиент; dq/dk ещё
    /// суммируются по v-головам группы k-головы).
    GdnBwd { has_seg: bool },
}
