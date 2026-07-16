# T5-XXL text encoder (google/t5-v1_1-xxl, T5EncoderModel — только encoder) для FLUX.1-dev text_encoder_2

# Спецификация bit-exact порта: T5-XXL encoder (FLUX.1-dev text_encoder_2)

Источник ground-truth: `transformers/models/t5/modeling_t5.py` (классы `T5EncoderModel`, `T5Stack`, `T5Block`, `T5LayerSelfAttention`, `T5Attention`, `T5LayerFF`, `T5DenseGatedActDense`, `T5LayerNorm`), `transformers/activations.py::NewGELUActivation`, `transformers/masking_utils.py::eager_mask`, и `diffusers/pipelines/flux/pipeline_flux.py::_get_t5_prompt_embeds`.

Это ТОЛЬКО encoder. Нет декодера, нет cross-attention, нет KV-кэша, нет causal-маски, нет dropout (inference).

---

## 0. Конфиг FLUX.1-dev/text_encoder_2/config.json (зашитые значения)

```
d_model (D)                     = 4096
d_ff    (FF)                    = 10240
d_kv    (head_dim, Dh)         = 64
num_heads (H)                  = 64
num_layers (L)                 = 24
inner_dim = H*Dh               = 64*64 = 4096   (== d_model, но это совпадение)
feed_forward_proj              = "gated-gelu"  → is_gated_act = True
dense_act_fn                   = "gelu_new"
relative_attention_num_buckets = 32
relative_attention_max_distance= 128
layer_norm_epsilon (eps)       = 1e-6
vocab_size                     = 32128
is_decoder                     = False (encoder), bidirectional = True
torch_dtype                    = bfloat16   ← ВЕСА В BF16
pad_token_id                   = 0
eos_token_id                   = 1
```

Compute-dtype: bf16 (веса bf16; вычисления в bf16, кроме явных upcast в f32 — см. ниже). last_hidden_state на выходе FLUX кастится в bf16.

---

## 1. Вход и токенизация (FLUX-специфика)

FLUX (`_get_t5_prompt_embeds`) токенизирует:
- `padding="max_length"`, `max_length=512`, `truncation=True`, T5Tokenizer (SentencePiece `spiece.model`).
- Результат `input_ids` имеет форму **[B, 512]** (ВСЕГДА паддится до 512; короткие промпты добиваются pad_token_id=0).
- T5 добавляет EOS=`</s>`(id=1) в конце реального текста; pad=`<pad>`(id=0). НЕТ BOS.

**КРИТИЧНО (главная FLUX-гоча):** FLUX вызывает энкодер как
`self.text_encoder_2(text_input_ids, output_hidden_states=False)` — **БЕЗ `attention_mask`**.
Значит внутри `attention_mask=None` → mask НЕ строится → **padding-токены НЕ маскируются**: все 512 позиций (включая pad) полноценно участвуют в attention. Это намеренное поведение FLUX. Для bit-exact ПОРТ ДОЛЖЕН тоже игнорировать padding (никакого -inf на pad-колонки), если на вход не передан явный attention_mask.

Формы далее: B — батч, S=512 — длина (q_len == k_len == S, self-attention).

---

## 2. Веса: ключи HF state_dict (для loader'а)

Все веса bf16. Linear в PyTorch хранит вес как **[out_features, in_features]**; forward = `x @ W.T`. Bias НЕТ нигде.

```
shared.weight                                                  [32128, 4096]   embedding (= encoder.embed_tokens, tied)
encoder.block.{i}.layer.0.SelfAttention.q.weight               [4096, 4096]    i=0..23
encoder.block.{i}.layer.0.SelfAttention.k.weight               [4096, 4096]
encoder.block.{i}.layer.0.SelfAttention.v.weight               [4096, 4096]
encoder.block.{i}.layer.0.SelfAttention.o.weight               [4096, 4096]
encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight  [32, 64]  ТОЛЬКО в block 0
encoder.block.{i}.layer.0.layer_norm.weight                    [4096]          RMSNorm gain перед attn
encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight           [10240, 4096]   gate (через gelu_new)
encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight           [10240, 4096]   linear
encoder.block.{i}.layer.1.DenseReluDense.wo.weight             [4096, 10240]   down
encoder.block.{i}.layer.1.layer_norm.weight                    [4096]          RMSNorm gain перед FF
encoder.final_layer_norm.weight                                [4096]          RMSNorm gain в конце
```

Примечания loader:
- `_tied_weights_keys = {"encoder.embed_tokens.weight": "shared.weight"}` — embedding лежит под `shared.weight`. Использовать его как таблицу эмбеддингов.
- `_keys_to_ignore_on_load_unexpected = [r"decoder"]` — в FLUX-чекпойнте декодера нет, игнорировать при наличии.
- `relative_attention_bias.weight` ЕСТЬ только в block 0. У block 1..23 поле SelfAttention.* без него — bias переиспользуется (см. §5).
- Веса лежат в двух файлах: `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`; маппинг в `model.safetensors.index.json` (219 ключей).

---

## 3. T5LayerNorm (это RMSNorm; используется ВЕЗДЕ как норма)

Псевдокод (`x`: [..., D], `weight`: [D], eps=1e-6):
```
x32       = x.to(float32)
variance  = mean(x32 * x32, axis=-1, keepdim=True)        # mean(x^2) по последней оси (D=4096), БЕЗ вычитания среднего
x_norm    = x32 * rsqrt(variance + eps)                     # rsqrt = 1/sqrt(...)
# КАСТ обратно, если weight в half/bf16:
if weight.dtype in {float16, bfloat16}:
    x_norm = x_norm.to(weight.dtype)                        # тут bf16
out       = weight * x_norm                                 # поэлементно, broadcast по D
```
Bit-exact гочи:
- variance/нормировка считаются в **f32** (upcast обязателен). Затем результат кастится в bf16 ПЕРЕД умножением на weight (т.к. weight bf16). Итог: `out = weight_bf16 * (x32 * rsqrt(var+eps)).to(bf16)`.
- НЕТ вычитания среднего, НЕТ bias, НЕТ `1+w` (gain используется напрямую, не `(1+weight)` как в Gemma).
- eps складывается ВНУТРИ rsqrt (var+eps), не снаружи.
- `mean` по оси D — сумма 4096 значений / 4096 в f32.

---

## 4. gelu_new (NewGELUActivation) — активация gate в FFN

```
gelu_new(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
```
- `sqrt(2/pi)` = math.sqrt(2.0/math.pi) ≈ 0.7978845608028654 (вычислять как sqrt(2/pi), не хардкодить округлённо).
- `x^3` = `torch.pow(x, 3.0)` (= x*x*x).
- Считается в dtype входа (bf16) — НЕТ upcast в f32 внутри gelu. (Точность tanh в bf16 — как в torch; для bit-exact на CUDA использовать тот же порядок: tmp = x + 0.044715*x^3; arg = 0.7978845608*tmp; 0.5*x*(1+tanh(arg)).)

---

## 5. Relative position bias (считается ОДИН раз, в block 0, передаётся во все слои)

### 5.1 `_relative_position_bucket` (bidirectional=True, num_buckets=32, max_distance=128)

Вход: `relative_position` = `memory_position - query_position` (int), форма [S, S].
```
def relative_position_bucket(rp, bidirectional=True, num_buckets=32, max_distance=128):
    relative_buckets = 0
    # bidirectional ветка (encoder):
    num_buckets //= 2                                    # 32 -> 16
    relative_buckets += (rp > 0).to(long) * num_buckets  # +16 там, где rp>0
    rp = abs(rp)
    # теперь rp в [0, inf)
    max_exact = num_buckets // 2                          # 16 // 2 = 8
    is_small = rp < max_exact                             # rp < 8
    # логарифмическая часть:
    rp_if_large = max_exact + ( log(rp.float() / max_exact)
                                / log(max_distance / max_exact)
                                * (num_buckets - max_exact) ).to(long)
                # = 8 + floor_via_long( log(rp/8) / log(128/8) * (16-8) )
                # log(128/8)=log(16); множитель (num_buckets-max_exact)=8
    rp_if_large = min(rp_if_large, num_buckets - 1)       # clamp до 15
    relative_buckets += where(is_small, rp, rp_if_large)
    return relative_buckets                               # int в [0, 31]
```
Bit-exact детали:
- `num_buckets //= 2` делает 16; `max_exact = 8`. Итоговые бакеты в [0,31] (0..15 для rp<=0, 16..31 для rp>0).
- `.to(torch.long)` = усечение к нулю (floor для положительных). `rp.float()` — деление и log в f32.
- `log` — натуральный логарифм. При rp==0 в малой ветке (is_small=True) берётся rp, ветка large не используется (но вычисляется и для rp=0 даёт log(0)=-inf → потом where выбирает is_small=True путь, так что -inf/nan отбрасывается; в реализации можно безопасно считать large только при rp>=max_exact, но порядок torch: считает обе, выбирает where). Для bit-exact достаточно: при is_small → rp, иначе → clamp(rp_if_large,15).
- `log(max_distance/max_exact)` = log(128/8)=log(16). Это скаляр.

### 5.2 `compute_bias(query_length=S, key_length=S)` (в block 0)

```
context_position = arange(S)[:, None]        # [S,1]   (past_seen_tokens=0)
memory_position  = arange(S)[None, :]        # [1,S]
relative_position = memory_position - context_position   # [S,S], = k - q
bucket = _relative_position_bucket(relative_position, bidirectional=True, 32, 128)  # [S,S] int
values = relative_attention_bias_embedding(bucket)       # lookup в [32,64] → [S, S, H=64]
values = values.permute(2,0,1).unsqueeze(0)              # → [1, H=64, S, S]
return values                                            # position_bias, dtype = bf16 (dtype весов embedding)
```
- Embedding lookup: `relative_attention_bias.weight` имеет форму [num_buckets=32, n_heads=64]; индексируется бакетами → [S,S,64], затем permute в [1,64,S,S].
- Этот bias считается ОДИН раз (в block 0, `has_relative_attention_bias=True`) и затем `position_bias` передаётся неизменным во все block 1..23 (там `position_bias is not None` → compute пропускается).
- В FLUX (mask=None) position_bias НЕ модифицируется маской: `if mask is not None` — пропускается. Так что bias = чистый relative bias [1,64,512,512], одинаковый для всех слоёв.

---

## 6. T5Attention.forward (self-attention, без масштаба 1/sqrt(d))

Вход: `hidden_states` [B,S,D]=[B,512,4096], `position_bias` [1,H,S,S] (None только в block 0 → считается там).

```
# проекции (bias=False):
q = hidden_states @ Wq.T                 # [B,S,4096]
k = hidden_states @ Wk.T                 # [B,S,4096]
v = hidden_states @ Wv.T                 # [B,S,4096]
# reshape в головы: view(B,S,H,Dh).transpose(1,2):
q = q.view(B,S,H,Dh).transpose(1,2)      # [B,H,S,Dh]=[B,64,512,64]
k = k.view(B,S,H,Dh).transpose(1,2)      # [B,H,S,Dh]
v = v.view(B,S,H,Dh).transpose(1,2)      # [B,H,S,Dh]

# scores БЕЗ масштабирования (НЕТ /sqrt(Dh)!):
scores = q @ k.transpose(-1,-2)          # [B,H,S,S]   = matmul(q, k^T)

# position_bias (в block 0 вычислить через compute_bias; иначе пришедший):
scores = scores + position_bias          # broadcast [1,H,S,S] на [B,H,S,S]
                                          # (mask=None в FLUX → маска не добавляется)

# softmax в f32, обратно в bf16:
attn = softmax(scores.float(), dim=-1).type_as(scores)   # upcast f32, softmax по k-оси (последняя), → bf16
# dropout p=0.1 но training=False → no-op

out = attn @ v                           # [B,H,S,Dh]
out = out.transpose(1,2).contiguous()    # [B,S,H,Dh]
out = out.reshape(B,S,inner_dim=4096)    # [B,S,4096]
out = out @ Wo.T                         # [B,S,4096]   (bias=False)
return out
```
Bit-exact гочи:
- **НЕТ деления на sqrt(d_kv)**. T5 не масштабирует scores; масштаб «впитан» в инициализацию весов. Это самое частое место ошибки.
- `scores += position_bias` — bias добавляется ПЕРЕД softmax, к сырым (немасштабированным) логитам.
- softmax считается в **f32** (`scores.float()`), затем `.type_as(scores)` → bf16. Сумма exp по 512 ключам в f32.
- reshape голов: `view(B,S,H,Dh)` затем `transpose(1,2)` — порядок «head-major по последней оси D» (голова h занимает d-срез [h*Dh:(h+1)*Dh]). На выходе `transpose(1,2).contiguous().reshape(B,S,-1)` собирает обратно в тот же порядок. Транспоз тут реальный (не просто reshape) — нужен `.contiguous()` перед reshape, иначе раскладка неверна.
- `k.transpose(-1,-2)` для scores; q@k^T = einsum("bhqd,bhkd->bhqk").

---

## 7. T5LayerSelfAttention (residual #1)

```
normed = T5LayerNorm_layer0(hidden_states)         # RMSNorm, weight = block.i.layer.0.layer_norm.weight
attn_out = T5Attention(normed, position_bias)      # §6, возвращает (attn_out, position_bias)
hidden_states = hidden_states + attn_out           # residual (dropout p=0.1 → no-op в inference)
return hidden_states, position_bias
```
Гоча: норма ПЕРЕД attention (pre-norm). Residual прибавляет attn_out к НЕнормированному входу.

---

## 8. T5DenseGatedActDense + T5LayerFF (residual #2)

T5LayerFF.forward:
```
normed = T5LayerNorm_layer1(hidden_states)          # RMSNorm, weight = block.i.layer.1.layer_norm.weight
ff_out = DenseGatedActDense(normed)                 # ниже
hidden_states = hidden_states + ff_out              # residual (dropout → no-op)
return hidden_states
```
T5DenseGatedActDense.forward (вход `x` [B,S,4096]):
```
hidden_gelu   = gelu_new(x @ wi_0.T)                # [B,S,10240], gate через gelu_new (§4)
hidden_linear = x @ wi_1.T                           # [B,S,10240], линейный
h             = hidden_gelu * hidden_linear          # поэлементно [B,S,10240]
# dropout p=0.1 → no-op
# каст в dtype wo (bf16) — тут уже bf16, no-op
out           = h @ wo.T                             # [B,S,4096]
return out
```
Bit-exact гочи:
- Порядок: gate=wi_0→gelu, lin=wi_1 (БЕЗ активации), произведение gate*lin, затем wo.
- gelu_new применяется ТОЛЬКО к ветке wi_0.
- Все три Linear без bias.
- (Заметка про fp32-wo для 8-битной квантизации в коде неактуальна: тут bf16, каст no-op.)

---

## 9. T5Block.forward (полный слой i)

```
hidden_states, position_bias = T5LayerSelfAttention(hidden_states, attention_mask=None, position_bias)  # §7
# clamp inf только если dtype==float16 — тут bf16, ПРОПУСКАЕТСЯ (для bf16 НЕТ clamp)
hidden_states = T5LayerFF(hidden_states)            # §8
# clamp inf только для float16 → пропускается
return hidden_states, position_bias
```
Гоча: clamp на inf срабатывает ТОЛЬКО при `dtype==torch.float16`. FLUX в bf16 → этих clamp НЕТ. Не реализовывать их для bf16.

---

## 10. T5Stack / T5EncoderModel.forward (полный энкодер)

```
input_ids: [B, 512]  (pad=0, eos=1)
# attention_mask = None в FLUX → mask НЕ строится (см. §11)

inputs_embeds = shared.weight[input_ids]            # embedding lookup, [B,512,4096], БЕЗ масштаба (нет *sqrt(d))
hidden_states = inputs_embeds                        # dropout p=0.1 → no-op

position_bias = None
for i in 0..23:
    hidden_states, position_bias = T5Block_i(hidden_states, attention_mask=None, position_bias)
    # block 0 вычисляет position_bias (§5) и возвращает; block 1..23 переиспользуют

hidden_states = final_layer_norm(hidden_states)      # RMSNorm, encoder.final_layer_norm.weight
# dropout → no-op
return last_hidden_state = hidden_states              # [B,512,4096]
```
Bit-exact гочи:
- embed_tokens БЕЗ масштабирования (в отличие от Gemma: тут НЕТ умножения на sqrt(d_model)).
- position_bias считается лениво в block 0 и шарится — НЕ пересчитывать в каждом слое.
- final_layer_norm — ещё одна RMSNorm после последнего блока.
- Выход FLUX берёт `[0]` = last_hidden_state, кастит в bf16 (уже bf16). Это и есть `prompt_embeds` для FLUX [B,512,4096].
- НЕТ pooling, НЕТ projection head на выходе encoder (T5EncoderModel отдаёт сырой last_hidden_state).

---

## 11. attention_mask (если когда-нибудь передаётся; в FLUX — None)

Для общего порта (на случай явного attention_mask [B,S] из 1/0):
`create_bidirectional_mask` → `eager_mask`:
```
min_dtype = torch.finfo(dtype).min          # для bf16 ≈ -3.3895e38 (НЕ -inf!)
mask4d[b,1,q,k] = 0.0   если позиция k валидна (attention_mask[b,k]==1)
                = min_dtype  иначе
# затем position_bias = position_bias + mask4d (в block 0, внутри if mask is not None)
```
- Маскирование аддитивное: к position_bias прибавляется 0 (валидно) или `finfo(dtype).min` (pad). НЕ `-inf`, а минимальное конечное значение dtype (для bf16 это -3.3895313892515355e+38).
- Маска зависит только от key-позиции (bidirectional, broadcast по q): [B,1,S,S], столбцы pad-ключей = min_dtype.
- **В FLUX этот путь НЕ исполняется** (attention_mask=None). Реализовать опционально, но дефолт = без маски.

---

## 12. Итоговый порядок операций (один проход, для чек-листа)

1. embed: `h = shared.weight[input_ids]` → [B,512,4096], без масштаба.
2. for i in 0..23:
   a. `n = rmsnorm(h, block.i.layer.0.layer_norm.weight)`
   b. q,k,v = n@Wq.T, n@Wk.T, n@Wv.T → reshape [B,64,512,64]
   c. `scores = q@k^T` (БЕЗ /sqrt(64))
   d. в block 0: `pos_bias = compute_bias()` [1,64,512,512]; иначе берётся из block 0
   e. `scores += pos_bias` (mask=None)
   f. `attn = softmax(scores.float(), -1).bf16`
   g. `o = (attn@v).transpose(1,2).contiguous().reshape(B,512,4096) @ Wo.T`
   h. `h = h + o`
   i. `n2 = rmsnorm(h, block.i.layer.1.layer_norm.weight)`
   j. `g = gelu_new(n2@wi_0.T); l = n2@wi_1.T; ff = (g*l)@wo.T`
   k. `h = h + ff`
3. `h = rmsnorm(h, final_layer_norm.weight)`
4. return h [B,512,4096] (bf16).

## WEIGHT KEYS
Embedding (tied): `shared.weight` [32128, 4096] bf16 — таблица эмбеддингов (== encoder.embed_tokens). На входе lookup input_ids→[B,512,4096], БЕЗ масштаба.

Per-block i=0..23:
- `encoder.block.{i}.layer.0.SelfAttention.q.weight` [4096,4096] (Linear, x@W.T, no bias)
- `encoder.block.{i}.layer.0.SelfAttention.k.weight` [4096,4096]
- `encoder.block.{i}.layer.0.SelfAttention.v.weight` [4096,4096]
- `encoder.block.{i}.layer.0.SelfAttention.o.weight` [4096,4096]
- `encoder.block.{i}.layer.0.layer_norm.weight` [4096]  (RMSNorm gain, pre-attn)
- `encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight` [10240,4096]  (gate → gelu_new)
- `encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight` [10240,4096]  (linear)
- `encoder.block.{i}.layer.1.DenseReluDense.wo.weight` [4096,10240]  (down)
- `encoder.block.{i}.layer.1.layer_norm.weight` [4096]  (RMSNorm gain, pre-FF)

ТОЛЬКО block 0:
- `encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight` [32, 64]  (nn.Embedding[num_buckets=32, n_heads=64]; lookup бакетами; bias шарится во все 24 слоя)

Финал:
- `encoder.final_layer_norm.weight` [4096]  (RMSNorm gain после block 23)

Всего 219 ключей. Файлы: model-00001-of-00002.safetensors, model-00002-of-00002.safetensors; маппинг в model.safetensors.index.json. Все веса bf16. Linear-веса хранятся как [out,in] (PyTorch nn.Linear) → forward = x @ W.T. Bias отсутствует везде. `_keys_to_ignore_on_load_unexpected=[r"decoder"]`, `_tied_weights_keys={"encoder.embed_tokens.weight":"shared.weight"}`.

## GOTCHAS
1. **НЕТ масштаба 1/sqrt(d_kv) в attention.** scores = q@k^T напрямую (+ pos_bias). Самое частое место ошибки. T5 «впитал» масштаб в веса.

2. **T5LayerNorm = RMSNorm с upcast в f32:** variance=mean(x32^2) по последней оси D (без вычитания среднего), x*rsqrt(var+eps), eps ВНУТРИ rsqrt; затем КАСТ результата в bf16 ПЕРЕД умножением на weight (т.к. weight bf16); out = weight_bf16 * x_norm_bf16. НЕТ bias, НЕТ `1+w` (gain напрямую, не как Gemma).

3. **FLUX вызывает энкодер БЕЗ attention_mask** → padding-токены (pad_id=0 до длины 512) НЕ маскируются, полноценно участвуют в attention. Не добавлять -inf на pad. input_ids всегда [B,512] (padding="max_length", max_length=512, truncation=True).

4. **embed_tokens БЕЗ масштабирования** (нет *sqrt(d_model), в отличие от Gemma).

5. **relative position bias считается ОДИН раз в block 0** (has_relative_attention_bias=True) и шарится во все 24 слоя неизменным. Форма [1,64,512,512], добавляется к scores ПЕРЕД softmax. bidirectional=True для энкодера (num_buckets делится на 2 → 16, бакеты 16..31 для rp>0).

6. **_relative_position_bucket:** num_buckets//=2 (32→16), max_exact=8; малые rp<8 → rp напрямую; большие → 8 + long(log(rp/8)/log(16)*8), clamp до 15; +16 если rp>0. log в f32, .to(long) усекает к нулю. relative_position = memory(k) - context(q).

7. **softmax в f32:** softmax(scores.float(),-1).type_as(scores)→bf16. Сумма exp по 512 ключам в f32.

8. **gelu_new (точная формула):** 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). sqrt(2/pi)≈0.7978845608. Считается в bf16 (без upcast). Применяется ТОЛЬКО к ветке wi_0; wi_1 без активации; затем поэлементное произведение gate*lin → wo.

9. **Reshape голов — реальный transpose:** view(B,S,H,Dh).transpose(1,2) на входе; на выходе transpose(1,2).contiguous().reshape(B,S,4096). Нужен .contiguous() перед финальным reshape (голова h = d-срез [h*64:(h+1)*64]).

10. **clamp на inf ТОЛЬКО для float16** — в bf16 НЕ исполняется, не реализовывать.

11. **inner_dim=4096 == d_model — совпадение** (H*Dh=64*64). Все 4 проекции q/k/v/o квадратные [4096,4096], no bias.

12. **Pre-norm + residual:** норма ПЕРЕД sub-блоком, residual прибавляется к НЕнормированному входу (h = h + sublayer(norm(h))). Два residual на блок (attn, FF). final_layer_norm после последнего блока.

13. **dropout p=0.1 везде no-op** в inference (training=False).

14. Если когда-нибудь передан явный attention_mask: аддитивная маска использует **torch.finfo(bf16).min ≈ -3.3895e38**, НЕ -inf; зависит только от key-позиции (broadcast по q); прибавляется к position_bias в block 0 (внутри if mask is not None).

15. Выход — сырой last_hidden_state [B,512,4096], БЕЗ pooling/projection. FLUX берёт [0] и кастит в bf16 → prompt_embeds.
