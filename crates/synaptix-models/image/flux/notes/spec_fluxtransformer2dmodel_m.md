# FluxTransformer2DModel (MMDiT ядро FLUX.1-dev) — bit-exact spec для нативного фреймворка synaptix

# FluxTransformer2DModel (MMDiT) — BIT-EXACT спецификация (FLUX.1-dev)

Источник истины: `diffusers/models/transformers/transformer_flux.py`, `embeddings.py`, `normalization.py`, `attention.py` (FeedForward), `attention_dispatch.py` (SDPA), `pipeline_flux.py` (id-сетки). Все формулы и формы выверены по коду.

## 0. КОНФИГ (из transformer/config.json)
```
patch_size = 1
in_channels = 64           (out_channels = in_channels = 64)
num_layers = 19            (double-stream FluxTransformerBlock)
num_single_layers = 38     (single-stream FluxSingleTransformerBlock)
attention_head_dim = 128   (= head_dim = dim_head)
num_attention_heads = 24
inner_dim = 24*128 = 3072
joint_attention_dim = 4096 (T5 контекст)
pooled_projection_dim = 768 (CLIP pooled)
guidance_embeds = true     → CombinedTimestepGuidanceTextProjEmbeddings
axes_dims_rope = (16, 56, 56)  (сумма = 128 = head_dim)
theta_rope = 10000
mlp_ratio (single) = 4.0 → mlp_hidden_dim = 12288
FF (double) inner = 4*3072 = 12288
```
Дефолтный dtype весов FLUX.1-dev = bfloat16. Все Linear имеют bias=True, КРОМЕ attention Q/K/V в double-блоках — там тоже bias=True (см. §6). См. §8 «bias/no-bias таблица».

## 1. ВХОД forward(...)
- `hidden_states`: `[B, img_seq, 64]` — УЖЕ packed латент (patchify 2×2 сделан в пайплайне).
- `encoder_hidden_states`: `[B, txt_seq, 4096]` — выход T5 (txt_seq=512 для FLUX-dev по умолчанию; spec не зависит от значения).
- `pooled_projections`: `[B, 768]` — CLIP pooled.
- `timestep`: `[B]` — доли времени в диапазоне [0,1] (FlowMatch).
- `guidance`: `[B]` — guidance scale как float (например 3.5), НЕ нормирована.
- `img_ids`: `[img_seq, 3]` (если пришёл `[B,img_seq,3]` → берётся `img_ids[0]`).
- `txt_ids`: `[txt_seq, 3]` (аналогично 3d→`txt_ids[0]`).

ВАЖНО про id-сетки (строятся в пайплайне, но нужны для self-contained):
```
txt_ids = zeros(txt_seq, 3)                     # все нули
img_ids[h,w] = [0, h, w], h∈[0,H/2), w∈[0,W/2)  # H,W — пиксельные/8; reshape→[H/2*W/2, 3]
  img_ids[...,0]=0 (axis-0 константа), img_ids[...,1]=row, img_ids[...,2]=col
```

## 2. x_embedder
```
hidden_states = Linear_64→3072(hidden_states)   # bias=True
# форма [B, img_seq, 3072]
```
Ключи: `x_embedder.weight [3072,64]`, `x_embedder.bias [3072]`.

## 3. time_text_embed (CombinedTimestepGuidanceTextProjEmbeddings) → temb [B,3072]
ПОРЯДОК (КРИТИЧНО — масштаб времени):
```
dt = hidden_states.dtype                         # bf16
timestep = timestep.to(dt) * 1000.0              # ← домножение на 1000 ДО эмбеддинга
guidance = guidance.to(dt) * 1000.0              # тоже *1000
```
### 3.1 time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
`get_timestep_embedding(t, dim=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1, max_period=10000)`:
```
half = 128
exponent = -ln(10000) * arange(0,128, dtype=f32)      # [128]
exponent = exponent / (128 - 0)                        # downscale_freq_shift=0 → делим на 128
emb_freq = exp(exponent)                               # [128], f32
emb = t[:,None].float() * emb_freq[None,:]             # [B,128]  (t уже *1000)
emb = 1 * emb                                          # scale=1
emb = cat([sin(emb), cos(emb)], dim=-1)                # [B,256]
# flip_sin_to_cos=True → переставить половины: cat([emb[:,128:], emb[:,:128]], -1)
#   итог = cat([cos(emb), sin(emb)], -1)               # [B,256]
```
ВАЖНО: внутренние вычисления sin/cos в f32; t передаётся как bf16-значение, но `.float()` приводит в f32 (т.е. сначала t округлено до bf16 при *1000, затем апкаст).
Результат timesteps_proj `[B,256]`, аналогично guidance_proj `[B,256]`.

### 3.2 timestep_embedder / guidance_embedder = TimestepEmbedding(256→3072), act=SiLU
Перед MLP: `.to(dtype=pooled_projection.dtype)` (bf16). MLP:
```
x = Linear_256→3072(x)        # linear_1, bias=True
x = SiLU(x)                   # silu(z)=z*sigmoid(z)
x = Linear_3072→3072(x)       # linear_2, bias=True
```
ОТДЕЛЬНЫЕ веса для timestep и guidance (общий time_proj, разные embedder).
```
timesteps_emb = timestep_embedder(timesteps_proj→bf16)   # [B,3072]
guidance_emb  = guidance_embedder(guidance_proj→bf16)     # [B,3072]
time_guidance_emb = timesteps_emb + guidance_emb          # [B,3072]
```
### 3.3 text_embedder = PixArtAlphaTextProjection(768→3072, act_fn="silu")
```
x = Linear_768→3072(pooled_projections)   # linear_1, bias=True
x = SiLU(x)
x = Linear_3072→3072(x)                    # linear_2, bias=True
pooled_emb = x                              # [B,3072]
```
### 3.4 Сумма
```
temb = time_guidance_emb + pooled_emb      # [B,3072]
```
Ключи: `time_text_embed.timestep_embedder.linear_1/2.{weight,bias}`, `...guidance_embedder.linear_1/2.*`, `...text_embedder.linear_1/2.*`.

## 4. context_embedder
```
encoder_hidden_states = Linear_4096→3072(encoder_hidden_states)   # bias=True
# [B, txt_seq, 3072]
```
Ключи: `context_embedder.weight [3072,4096]`, `context_embedder.bias`.

## 5. RoPE pos_embed (FluxPosEmbed, theta=10000, axes_dim=[16,56,56])
```
ids = cat([txt_ids, img_ids], dim=0)       # [txt_seq+img_seq, 3]  ← txt ПЕРВЫМ
pos = ids.float()                          # f32 (на CUDA freqs_dtype=float64! см. ниже)
```
ГОЧА dtype: на CPU/CUDA `freqs_dtype = torch.float64` (НЕ mps/npu). Т.е. частоты и outer считаются в f64, потом `.float()`. Для bit-exact в synaptix: считать freqs в f64 (или максимально точной), затем cos/sin, затем привести в f32.

Для каждой оси i∈{0,1,2} с dim_i ∈ {16,56,56}, по `get_1d_rotary_pos_embed(dim_i, pos[:,i], theta=10000, repeat_interleave_real=True, use_real=True, freqs_dtype=f64)`:
```
freqs = 1.0 / (theta ** (arange(0, dim_i, 2, f64) / dim_i))   # [dim_i/2]  (linear_factor=ntk=1)
freqs = outer(pos_i, freqs)                                   # [seq, dim_i/2], f64
cos_i = repeat_interleave(cos(freqs), 2, dim=1).float()       # [seq, dim_i]
sin_i = repeat_interleave(sin(freqs), 2, dim=1).float()       # [seq, dim_i]
```
`repeat_interleave(x,2,dim=1)`: `[a,b,c]→[a,a,b,b,c,c]` (НЕ tile). Конкатенация по осям:
```
freqs_cos = cat([cos_0, cos_1, cos_2], dim=-1)   # [seq, 16+56+56=128]
freqs_sin = cat([sin_0, sin_1, sin_2], dim=-1)   # [seq, 128]
image_rotary_emb = (freqs_cos, freqs_sin)        # каждый [txt_seq+img_seq, 128]
```
Так как txt_ids=0 → для txt-токенов freqs=0 → cos=1, sin=0 (RoPE = identity для текста). Но вычислять для всей конкат-последовательности (txt+img) ЕДИНОЖДЫ и применять в double и single блоках к КОНКАТУ [txt; img] (порядок совпадает: txt первым).

### apply_rotary_emb(x, (cos,sin), sequence_dim=1)  [layout B,S,H,D]
x форма `[B, S, H, D=128]`. sequence_dim=1 → cos/sin броадкастятся как `cos[None,:,None,:]`, `sin[None,:,None,:]` (форма [1,S,1,128]).
use_real_unbind_dim = -1 (flux):
```
x_r, x_i = x.reshape(*x.shape[:-1], 64, 2).unbind(-1)   # каждый [B,S,H,64]
x_rotated = stack([-x_i, x_r], dim=-1).flatten(start=3) # [B,S,H,128]
                                                        # чередование: (-x1,x0,-x3,x2,...)
out = (x.float()*cos + x_rotated.float()*sin).to(x.dtype)  # ВЫЧИСЛЕНИЕ В f32, потом ←dtype
```
Т.е. для пары (d0,d1): out0 = x0*cos0 - x1*sin0; out1 = x1*cos1 + x0*sin1. Поскольку cos/sin репит-интерливятся, cos0==cos1, sin0==sin1 в паре. ГОЧА: x апкастится в f32 ПЕРЕД умножением, результат приводится назад в bf16.

## 6. Double-stream FluxTransformerBlock ×19
Вход: `hidden_states (img) [B,Si,3072]`, `encoder_hidden_states (txt) [B,St,3072]`, `temb [B,3072]`, `image_rotary_emb`.

### 6.1 norm1 = AdaLayerNormZero(img), norm1_context = AdaLayerNormZero(txt)
AdaLayerNormZero.forward(x, emb=temb):
```
emb6 = Linear_3072→18432( SiLU(temb) )        # bias=True; 18432=6*3072
shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp = emb6.chunk(6, dim=1)  # каждый [B,3072]
x = LayerNorm(x) * (1 + scale_msa[:,None]) + shift_msa[:,None]   # [B,S,3072]
return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
```
LayerNorm: elementwise_affine=False, eps=1e-6, по последней оси 3072. Формула LN: `(x-mean)/sqrt(var+eps)` где var = biased (mean of squares of (x-mean), деление на N=3072, НЕ N-1), БЕЗ gamma/beta. Апкаст в f32 как у torch.nn.LayerNorm (стандартно torch считает LN в f32 для bf16 входа). chunk порядок СТРОГО: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
```
norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm1(img, temb)
norm_enc, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = norm1_context(txt, temb)
```
Ключи: `transformer_blocks.N.norm1.linear.{weight[18432,3072],bias}`, `...norm1_context.linear.*`.

### 6.2 Joint attention (FluxAttnProcessor, added_kv_proj_dim=3072)
Проекции (bias=True для всех; out_bias=True):
```
q   = to_q(norm_hidden);  k   = to_k(norm_hidden);  v   = to_v(norm_hidden)     # img, [B,Si,3072]
eq  = add_q_proj(norm_enc); ek = add_k_proj(norm_enc); ev = add_v_proj(norm_enc) # txt, [B,St,3072]
```
unflatten последней оси в (heads=24, -1=128):
```
q = q.unflatten(-1,(24,128))     # [B,Si,24,128]   (layout B,S,H,D)
k,v аналогично; eq,ek,ev → [B,St,24,128]
```
QK-norm (RMSNorm per-head, dim=128, eps=1e-6, elementwise_affine=True):
```
q = norm_q(q)      # RMSNorm по последней оси (128)
k = norm_k(k)
eq = norm_added_q(eq)
ek = norm_added_k(ek)
```
RMSNorm: `var = mean(x.float()^2, -1, keepdim); x = x * rsqrt(var+1e-6)`; вычисление var В f32, затем `x_normed * weight`, weight в bf16 → x приводится в bf16 ПЕРЕД умножением на weight (см. RMSNorm.forward: если weight bf16, hidden→bf16, потом *weight). eps=1e-6 (FluxAttention eps в double-блоке = 1e-6; в коде FluxTransformerBlock передаёт eps=1e-6). norm_q/norm_k имеют weight [128]; norm_added_q/k тоже weight[128] (eps=1e-6 фикс в коде для added).
**ГОЧА bit-exact RMSNorm**: variance в f32, rsqrt в f32, ПОТОМ x*=rsqrt (ещё f32 или приведённый?). По коду: hidden_states (исходный dtype bf16) * rsqrt(f32) → broadcasting повышает в f32; затем `hidden = hidden.to(weight.dtype=bf16)`; затем `* weight`. Значит: (1) нормировка в f32, (2) round→bf16, (3) умножение на weight в bf16.

Конкатенация txt ПЕРЕД img по оси heads-seq (dim=1 = seq):
```
q = cat([eq, q], dim=1)    # [B, St+Si, 24, 128]   txt первым
k = cat([ek, k], dim=1)
v = cat([ev, v], dim=1)
```
RoPE на ВЕСЬ конкат (q и k, не v), sequence_dim=1 (см. §5):
```
q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
```
Attention = SDPA (scale=None → 1/sqrt(128)):
```
# dispatch permute [B,S,H,D]→[B,H,S,D], SDPA, permute назад
attn = softmax( (q@k^T)/sqrt(128) , dim=-1) @ v    # без маски, не causal, dropout=0
# результат [B, St+Si, 24, 128]
hidden = attn.flatten(2,3)    # [B, St+Si, 3072]
hidden = hidden.to(q.dtype)   # bf16
```
ГОЧА softmax: torch SDPA считает softmax в f32 внутри (для bf16 входа), масштаб ровно 1/sqrt(128). Порядок: scores=q@kᵀ, scale, softmax(f32), @v.
Split назад (txt первым):
```
context_attn (txt) = hidden[:, :St]       # split_with_sizes [St, Si]
img_attn          = hidden[:, St:]
img_attn = to_out[0](img_attn.contiguous())   # Linear_3072→3072, bias=True
                                              # to_out[1]=Dropout(0) — no-op
context_attn = to_add_out(context_attn.contiguous())  # Linear_3072→3072, bias=True
```
Ключи attn: `transformer_blocks.N.attn.{to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj}.{weight[3072,3072],bias[3072]}`, `...norm_q/norm_k/norm_added_q/norm_added_k.weight[128]`, `...to_out.0.{weight,bias}`, `...to_add_out.{weight,bias}`.

### 6.3 Применение attn-выходов + FF (img)
```
attn_output = gate_msa.unsqueeze(1) * img_attn        # [B,1,3072]*[B,Si,3072]
hidden_states = hidden_states + attn_output            # residual (исходный img до norm1)
norm2 = LayerNorm(elementwise_affine=False, eps=1e-6)(hidden_states)
norm2 = norm2 * (1 + scale_mlp[:,None]) + shift_mlp[:,None]
ff_output = ff(norm2)                                  # см. FF ниже
ff_output = gate_mlp.unsqueeze(1) * ff_output
hidden_states = hidden_states + ff_output              # [B,Si,3072]
```
### 6.4 Применение attn-выходов + FF (txt / context)
```
context_attn_output = c_gate_msa.unsqueeze(1) * context_attn
encoder_hidden_states = encoder_hidden_states + context_attn_output
norm2_context = LayerNorm(False,eps=1e-6)(encoder_hidden_states)
norm2_context = norm2_context * (1 + c_scale_mlp[:,None]) + c_shift_mlp[:,None]
context_ff = ff_context(norm2_context)
encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff
# ГОЧА: clip только если fp16: if dtype==float16: clip(-65504,65504). Для bf16 НЕ применяется.
```
ПОРЯДОК: сначала полностью img (attn-residual→ff→residual), потом txt. norm2/norm2_context БЕЗ обучаемых параметров (нет ключей в state_dict).

### 6.5 FeedForward (double): ff и ff_context, activation_fn="gelu-approximate"
FeedForward.net = [GELU(approximate=tanh, 3072→12288), Dropout(0), Linear(12288→3072)]. Индексы: net.0 = GELU(содержит .proj Linear), net.2 = выходной Linear.
```
h = Linear_3072→12288(x)          # net.0.proj, bias=True
h = gelu_tanh(h)                  # F.gelu(h, approximate='tanh')
# net.1 Dropout(0) — no-op
out = Linear_12288→3072(h)        # net.2, bias=True
```
**gelu_tanh формула (approximate="tanh"):**
```
gelu_tanh(x) = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
  sqrt(2/pi) = 0.7978845608028654
```
Ключи: `transformer_blocks.N.ff.net.0.proj.{weight[12288,3072],bias}`, `ff.net.2.{weight[3072,12288],bias}`; аналогично `ff_context.net.0.proj.*`, `ff_context.net.2.*`.

Выход блока: `(encoder_hidden_states, hidden_states)` (txt, img). Порядок присваивания в forward модели: `encoder_hidden_states, hidden_states = block(...)`.

(controlnet_block_samples — None в обычном инференсе, игнорировать.)

## 7. Single-stream FluxSingleTransformerBlock ×38
ВНИМАНИЕ: перед циклом single-блоков hidden_states (img) и encoder_hidden_states (txt) НЕ конкатятся в forward модели — конкат происходит ВНУТРИ каждого single-блока, и в конце блока сплитятся обратно. Сигнатура та же: вход (hidden_states=img, encoder_hidden_states=txt, temb, image_rotary_emb).

### 7.1 Тело блока
```
text_seq_len = St
hidden = cat([encoder_hidden_states, hidden_states], dim=1)   # [B, St+Si, 3072]  txt первым
residual = hidden
```
### 7.2 norm = AdaLayerNormZeroSingle
```
emb3 = Linear_3072→9216( SiLU(temb) )           # bias=True; 9216=3*3072
shift_msa, scale_msa, gate_msa = emb3.chunk(3, dim=1)    # [B,3072] каждый
norm_hidden = LayerNorm(False,eps=1e-6)(hidden) * (1+scale_msa[:,None]) + shift_msa[:,None]
gate = gate_msa                                  # [B,3072]
```
chunk порядок: shift_msa, scale_msa, gate_msa.
### 7.3 MLP-ветвь (параллельно attention)
```
mlp_hidden = Linear_3072→12288(norm_hidden)      # proj_mlp, bias=True
mlp_hidden = gelu_tanh(mlp_hidden)               # act_mlp = nn.GELU(approximate="tanh")
```
### 7.4 Attention-ветвь (FluxAttnProcessor, pre_only=True, БЕЗ added_kv, БЕЗ to_out)
Вход attention = norm_hidden (вся конкат [B,St+Si,3072]). encoder_hidden_states=None внутри attn → ветка без added-проекций и без split.
```
q = to_q(norm_hidden); k = to_k(norm_hidden); v = to_v(norm_hidden)   # bias=True
q = q.unflatten(-1,(24,128)); k,v аналогично                          # [B,St+Si,24,128]
q = norm_q(q); k = norm_k(k)    # RMSNorm dim=128, eps=1e-6, elementwise_affine=True
# added_kv_proj_dim is None → НЕТ конкатенации (eq/ek/ev отсутствуют)
q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
attn = SDPA(q,k,v, scale=1/sqrt(128))             # [B,St+Si,24,128]
attn_output = attn.flatten(2,3).to(q.dtype)       # [B,St+Si,3072]
# pre_only=True: НЕТ to_out, НЕТ to_add_out → возвращается attn_output как есть
```
ГОЧА: в single-блоке eps RMSNorm = 1e-6 (FluxAttention(eps=1e-6, pre_only=True)). norm_q/norm_k weight[128].
### 7.5 proj_out и residual
```
cat_out = cat([attn_output, mlp_hidden], dim=2)   # [B,St+Si, 3072+12288=15360]
gate = gate.unsqueeze(1)                           # [B,1,3072]
hidden = gate * Linear_15360→3072(cat_out)         # proj_out, bias=True
hidden = residual + hidden
# ГОЧА: if dtype==float16: clip(-65504,65504). bf16 → пропустить.
encoder_hidden_states = hidden[:, :St]             # сплит txt
hidden_states         = hidden[:, St:]             # сплит img
return encoder_hidden_states, hidden_states
```
Ключи: `single_transformer_blocks.N.norm.linear.{weight[9216,3072],bias}`, `...attn.{to_q,to_k,to_v}.{weight[3072,3072],bias}`, `...attn.norm_q/norm_k.weight[128]`, `...proj_mlp.{weight[12288,3072],bias}`, `...proj_out.{weight[3072,15360],bias}`. (Single attn НЕ имеет to_out/to_add_out/add_*_proj/norm_added_* — их нет в state_dict.)

После цикла single-блоков: используется только hidden_states (img); encoder_hidden_states отбрасывается. (В коде последнее присваивание сохраняет обе переменные, но дальше берётся только hidden_states.)

## 8. Финал: norm_out + proj_out
### 8.1 norm_out = AdaLayerNormContinuous(3072, cond=3072, elementwise_affine=False, eps=1e-6)
```
emb2 = Linear_3072→6144( SiLU(temb).to(x.dtype) )   # bias=True; 6144=2*3072
scale, shift = emb2.chunk(2, dim=1)                  # [B,3072] каждый
hidden_states = LayerNorm(False,eps=1e-6)(hidden_states) * (1+scale)[:,None,:] + shift[:,None,:]
```
chunk порядок: scale ПЕРВЫМ, shift ВТОРЫМ (отличие от AdaLayerNormZero, где shift первым!). norm — LayerNorm без аффинных параметров.
Ключи: `norm_out.linear.{weight[6144,3072],bias}`.
### 8.2 proj_out
```
output = Linear_3072→64(hidden_states)   # proj_out, bias=True
```
Форма выхода `[B, img_seq, 64]`. Ключи: `proj_out.{weight[64,3072],bias[64]}`. Возврат Transformer2DModelOutput(sample=output) (или (output,) если return_dict=False).

## 9. СВОДКА ФОРМ (B=1, txt_seq=St=512, img_seq=Si)
| шаг | форма |
|---|---|
| вход img | [B,Si,64] → x_embedder → [B,Si,3072] |
| вход txt | [B,512,4096] → context_embedder → [B,512,3072] |
| temb | [B,3072] |
| rope cos/sin | [512+Si, 128] |
| double attn q/k/v | [B,512+Si,24,128] |
| double выход | img [B,Si,3072], txt [B,512,3072] |
| single вход (конкат) | [B,512+Si,3072] |
| single attn | [B,512+Si,24,128] |
| single proj_out вход | [B,512+Si,15360] |
| выход | [B,Si,64] |

## 10. КЛЮЧЕВЫЕ BIT-EXACT ГОЧИ (checklist)
1. timestep И guidance домножаются на **1000** ДО эмбеддинга (после `.to(bf16)`). Округление до bf16 происходит ПЕРЕД *1000? Нет — `.to(dt)*1000`: сначала bf16, потом *1000 (тоже bf16). Затем в get_timestep_embedding `.float()`.
2. time_proj: flip_sin_to_cos=True, downscale_freq_shift=0 → порядок [cos, sin] на выходе; freq exponent делится на half_dim=128 (НЕ 127).
3. RoPE freqs на CUDA/CPU считаются в **float64**, затем cos/sin→float32. repeat_interleave(2) (НЕ tile/concat): [a,a,b,b,...].
4. apply_rotary_emb: rotate-pairs через reshape(...,64,2).unbind(-1) и stack([-x_i, x_r]).flatten → чередование (-x1,x0,-x3,x2,...). Вычисление в f32, round→bf16.
5. SDPA scale = 1/sqrt(128) (НЕ 1/sqrt(3072)), softmax в f32.
6. Конкатенация в attention: **txt первым** (encoder, затем hidden) по seq-оси; split назад тем же порядком.
7. AdaLayerNormZero chunk: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp). AdaLayerNormZeroSingle: (shift, scale, gate). AdaLayerNormContinuous: (scale, shift) — **scale первым!**
8. Модуляция LN: `LN(x)*(1+scale)+shift`. scale/gate берутся как `[:, None]` (broadcast по seq).
9. Все LayerNorm здесь: elementwise_affine=False, eps=1e-6, var=biased (÷N), torch считает в f32 для bf16.
10. RMSNorm (QK-norm): eps=1e-6, dim=128 per-head, var в f32, round→bf16 ПЕРЕД *weight.
11. FF активация = gelu **tanh-approx** (двойные ff + single proj_mlp + single act_mlp). Формула в §6.5. TimestepEmbedding/AdaLN/text_embedder используют **SiLU** (не gelu).
12. text_embedder (PixArtAlpha) act = SiLU (не gelu_tanh — несмотря на дефолт класса gelu_tanh, FLUX передаёт act_fn="silu").
13. Все Linear bias=True (включая to_q/k/v/add_*/to_out.0/to_add_out/proj_out/embedders). RMSNorm weight есть, bias нет. LayerNorm-ы (norm/norm2/norm2_context/norm внутри AdaLN) — БЕЗ параметров.
14. fp16-clip(-65504,65504) только если dtype==float16; для bf16/f32 пропустить. (В FLUX-dev обычно bf16 → не срабатывает.)
15. Веса Linear хранятся как [out,in] (PyTorch). matmul = x @ W^T + b.
16. Порядок суммирования temb: (timesteps_emb+guidance_emb) затем +pooled_emb (три слагаемых, ассоциативность важна в bf16).
17. img_ids[...,0] и txt_ids все по оси-0 = 0 → RoPE axis-0 (dim 16) даёт cos=1,sin=0 для всех; txt полностью identity (все нули).

## WEIGHT KEYS
ПОЛНЫЙ список ключей HF state_dict (префикс модуля transformer.; в чистом checkpoint без transformer. — см. index.json). Формы [out,in] для Linear weight, [dim] для norm/RMSNorm weight, [out] для bias.

ТОП-УРОВЕНЬ:
- x_embedder.weight [3072,64], x_embedder.bias [3072]
- context_embedder.weight [3072,4096], context_embedder.bias [3072]
- time_text_embed.timestep_embedder.linear_1.weight [3072,256], .linear_1.bias [3072]
- time_text_embed.timestep_embedder.linear_2.weight [3072,3072], .linear_2.bias [3072]
- time_text_embed.guidance_embedder.linear_1.weight [3072,256], .linear_1.bias [3072]
- time_text_embed.guidance_embedder.linear_2.weight [3072,3072], .linear_2.bias [3072]
- time_text_embed.text_embedder.linear_1.weight [3072,768], .linear_1.bias [3072]
- time_text_embed.text_embedder.linear_2.weight [3072,3072], .linear_2.bias [3072]
- norm_out.linear.weight [6144,3072], norm_out.linear.bias [6144]
- proj_out.weight [64,3072], proj_out.bias [64]

DOUBLE-блоки N=0..18 (transformer_blocks.N.):
- norm1.linear.weight [18432,3072], norm1.linear.bias [18432]
- norm1_context.linear.weight [18432,3072], norm1_context.linear.bias [18432]
- attn.to_q.weight [3072,3072], attn.to_q.bias [3072]
- attn.to_k.weight [3072,3072], attn.to_k.bias [3072]
- attn.to_v.weight [3072,3072], attn.to_v.bias [3072]
- attn.add_q_proj.weight [3072,3072], attn.add_q_proj.bias [3072]
- attn.add_k_proj.weight [3072,3072], attn.add_k_proj.bias [3072]
- attn.add_v_proj.weight [3072,3072], attn.add_v_proj.bias [3072]
- attn.norm_q.weight [128], attn.norm_k.weight [128]
- attn.norm_added_q.weight [128], attn.norm_added_k.weight [128]
- attn.to_out.0.weight [3072,3072], attn.to_out.0.bias [3072]   (to_out.1 = Dropout, нет весов)
- attn.to_add_out.weight [3072,3072], attn.to_add_out.bias [3072]
- ff.net.0.proj.weight [12288,3072], ff.net.0.proj.bias [12288]   (net.0 = GELU, .proj — внутренний Linear)
- ff.net.2.weight [3072,12288], ff.net.2.bias [3072]              (net.1 = Dropout, нет весов)
- ff_context.net.0.proj.weight [12288,3072], ff_context.net.0.proj.bias [12288]
- ff_context.net.2.weight [3072,12288], ff_context.net.2.bias [3072]
(norm2, norm2_context — LayerNorm elementwise_affine=False, БЕЗ ключей)

SINGLE-блоки N=0..37 (single_transformer_blocks.N.):
- norm.linear.weight [9216,3072], norm.linear.bias [9216]
- attn.to_q.weight [3072,3072], attn.to_q.bias [3072]
- attn.to_k.weight [3072,3072], attn.to_k.bias [3072]
- attn.to_v.weight [3072,3072], attn.to_v.bias [3072]
- attn.norm_q.weight [128], attn.norm_k.weight [128]
- proj_mlp.weight [12288,3072], proj_mlp.bias [12288]
- proj_out.weight [3072,15360], proj_out.bias [3072]
(НЕТ: add_*_proj, norm_added_*, to_out, to_add_out — single attn pre_only без added_kv)

Чекпойнт шардирован на 3 файла (diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors) + index.json (weight_map). Dtype = bfloat16. В index.json ключи БЕЗ префикса transformer. (т.е. начинаются прямо с x_embedder./transformer_blocks. и т.д.).

## GOTCHAS
КРИТИЧЕСКИЕ подводные камни (повторно, концентрированно):

1. МАСШТАБ ВРЕМЕНИ: timestep.to(bf16)*1000 И guidance.to(bf16)*1000 ДО get_timestep_embedding. Пропуск *1000 = полностью неверный temb.

2. get_timestep_embedding(dim=256, flip_sin_to_cos=True, downscale_freq_shift=0): exponent = -ln(10000)*arange(128)/128 (делитель = half_dim=128, т.к. shift=0). Выход после flip = cat([cos, sin]). flip переставляет половины местами.

3. RoPE freqs В FLOAT64 на CPU/CUDA (freqs_dtype=torch.float64), cos/sin→float32. theta=10000, dims=[16,56,56]. repeat_interleave(2,dim=1): [a,a,b,b,...] НЕ [a,b,a,b]. ids = cat([txt_ids, img_ids]) → txt первым. txt_ids все нули (RoPE=identity для текста); img_ids[h,w]=[0,h,w].

4. apply_rotary_emb FLUX-вариант (use_real_unbind_dim=-1): x.reshape(...,D/2,2).unbind(-1) даёт (x_even, x_odd); x_rotated = stack([-x_odd, x_even]).flatten → (-x1,x0,-x3,x2,...). out = x.float()*cos + x_rotated.float()*sin, затем →bf16. sequence_dim=1 (layout [B,S,H,D]), cos/sin броадкаст [1,S,1,D]. Применяется к q и k (НЕ v), на ВЕСЬ конкат [txt;img].

5. SDPA scale = 1/sqrt(head_dim=128) (scale=None default), НЕ 1/sqrt(3072). softmax в f32. Без маски, не causal. dispatch permute [B,S,H,D]→[B,H,S,D] вокруг SDPA.

6. chunk-порядки РАЗНЫЕ:
   - AdaLayerNormZero (6): shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
   - AdaLayerNormZeroSingle (3): shift_msa, scale_msa, gate_msa
   - AdaLayerNormContinuous (2): scale, shift  ← scale ПЕРВЫМ (инверсия!)
   Модуляция везде: norm(x)*(1+scale)+shift.

7. RMSNorm (QK-norm) eps=1e-6, per-head dim=128: variance=mean(x.f32^2,-1); x*=rsqrt(var+eps) в f32; round→bf16; *weight в bf16. weight[128] есть, bias нет.

8. LayerNorm-ы (norm1/norm2/norm/norm_out внутренние) elementwise_affine=False, eps=1e-6, БЕЗ параметров, var biased (÷N=3072), torch апкаст f32.

9. Активации: gelu-TANH (approximate="tanh") в double ff/ff_context и single proj_mlp+act_mlp: 0.5x(1+tanh(0.7978845608*(x+0.044715x^3))). SiLU (x*sigmoid(x)) в timestep/guidance/text embedders и во ВСЕХ AdaLN.linear (SiLU перед Linear модуляции). text_embedder act="silu" (НЕ дефолтный gelu_tanh класса).

10. Все Linear bias=True. matmul x@Wᵀ+b (W хранится [out,in]).

11. Single-блок: конкат [txt;img] ВНУТРИ блока (не в forward модели), сплит обратно в конце. MLP-ветвь и attn-ветвь параллельны от ОДНОГО norm_hidden. proj_out вход = cat([attn(3072), mlp_act(12288)])=15360. hidden=residual+gate*proj_out. После 38 single-блоков берётся только img-часть.

12. Double-блок порядок: norm1→attn→gate_msa·attn+res(img); norm2(mod)→ff→gate_mlp·ff+res(img); затем то же для txt (c_gate). FF residual на hidden_states ПОСЛЕ attn-residual.

13. fp16-clip(-65504,65504) только при dtype==float16 (в single и в txt-ветви double). bf16/f32 — не применять. FLUX-dev обычно bf16.

14. temb сумма: (timesteps_emb+guidance_emb)+pooled_emb — порядок важен в bf16.

15. norm_out (AdaLayerNormContinuous): SiLU(temb).to(x.dtype) ПЕРЕД linear; scale,shift=chunk(2); LN(x)*(1+scale)[:,None,:]+shift[:,None,:].

16. Выход [B,img_seq,64]. Распаковка латента (_unpack 2x2) — вне трансформера (в пайплайне).

17. unflatten(-1,(heads,-1)) даёт layout [B,S,H,D] (head — внешний от dim). reshape 3072→(24,128) С-порядок: первые 128 элементов = head0. flatten(2,3) обратно по тому же порядку.

18. Single attn НЕ имеет to_out/to_add_out/add_*_proj/norm_added_* (pre_only=True, added_kv_proj_dim=None). Double attn имеет все. norm_added_q/k eps=1e-6 (захардкожен в FluxAttention для added).
