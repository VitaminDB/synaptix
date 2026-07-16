# FLUX.1-dev VAE (AutoencoderKL, 16 latent channels) — DECODE path

# Implementation-спек: FLUX.1-dev VAE (AutoencoderKL, 16ch) — DECODE

## 0. Что это и чем отличается от нашей текущей реализации

Наш текущий `crates/synaptix-nn/src/vae/autoencoder_kl.rs` УЖЕ почти полностью покрывает FLUX VAE — архитектура VAE одинакова для SD/SDXL/SD3/FLUX, отличаются только числа конфига и две вещи в обвязке. FLUX VAE = тот же `Decoder` из diffusers `vae.py`, тот же `UNetMidBlock2D` (resnet, attn, resnet), те же `UpDecoderBlock2D` (N resnet + Upsample2D), тот же `ResnetBlock2D`, тот же `Attention`/`AttnProcessor2_0`. GroupNorm eps = 1e-6 везде, активация SiLU, single-head spatial attention в mid-блоке.

**Ровно 4 отличия от нашей SDXL-реализации, которые надо внести:**

1. **`latent_channels = 16` (не 4).** Меняет только `conv_in` (16→512 вместо 4→512) и форму входа `z`. Наш loader config-driven — достаточно нового `AutoencoderKlConfig` с `latent_channels=16`. `conv_in.weight` = `(512, 16, 3, 3)`.

2. **НЕТ `post_quant_conv` и НЕТ `quant_conv`** (`use_post_quant_conv=false`, `use_quant_conv=false`). В state_dict ЭТИХ КЛЮЧЕЙ ВООБЩЕ НЕТ (проверено: `quant_conv present? False`). Наш `AutoencoderKlDecoder::load` сейчас БЕЗУСЛОВНО грузит `post_quant_conv.*` → для FLUX это упадёт «key not found». Надо сделать `post_quant_conv: Option<Conv2dLayer>` и грузить его только если `use_post_quant_conv`. В `decode()`: `if let Some(pq) = &self.post_quant_conv { z = pq.forward(z)?; }` — иначе z идёт напрямую в `conv_in`. Для FLUX **latent = вход decoder напрямую**, без post_quant_conv.

3. **`scaling_factor=0.3611`, `shift_factor=0.1159`** (у SDXL shift=None, scale=0.13025). Формула входа decode (делает ПАЙПЛАЙН, не сам decoder; см. `pipeline_flux.py:1010`):
   ```
   z = latents / scaling_factor + shift_factor      # ВНИМАНИЕ: деление и СЛОЖЕНИЕ, не (z-shift)*scale
   image = vae.decode(z).sample
   ```
   Это надо положить в обёртку txt2img/encode-decode-API, а сам `decoder.decode()` оставляет raw-выход. Encode-сторона (если понадобится): `latents = (encoder_moments_mean - shift_factor) * scaling_factor`. Для txt2img нужен ТОЛЬКО decode → достаточно `z/scale + shift`.

4. **`mid_block_add_attention=true`** — у нас уже есть `VaeAttention` в mid. FLUX это тоже true, ничего менять не надо (но НЕ удалять attn). `force_upcast=true` → весь VAE считать в **f32** (веса BF16 в файле, но decode гонится в float32; см. §7).

Вывод: правки минимальны — добавить FLUX-конфиг, сделать post_quant_conv опциональным, обернуть scale/shift в API decode. Всё остальное (resnet/mid/up/norm/attn/upsample/conv_out) уже корректно для bit-exact.

---

## 1. Конфиг FLUX.1-dev VAE (config.json)

```
in_channels        = 3
out_channels       = 3
latent_channels    = 16
block_out_channels = [128, 256, 512, 512]
layers_per_block   = 2
norm_num_groups    = 32
act_fn             = "silu"
scaling_factor     = 0.3611
shift_factor       = 0.1159
use_quant_conv     = false
use_post_quant_conv= false
mid_block_add_attention = true
force_upcast       = true
sample_size        = 1024
down_block_types = 4× "DownEncoderBlock2D"
up_block_types   = 4× "UpDecoderBlock2D"
```

Производные величины:
- `resnet_eps = 1e-6`, `groupnorm eps = 1e-6` (ВСЕ нормы в VAE), `output_scale_factor = 1.0` (без деления).
- `reversed_block_out_channels = [512, 512, 256, 128]` (decoder идёт reversed).
- mid `in_channels = block_out_channels[-1] = 512`.
- mid attention: `attention_head_dim = 512` → `heads = 512//512 = 1` (ОДНА голова), `dim_head = 512`, `scale = 512**-0.5`.
- число resnet в up-блоке = `layers_per_block + 1 = 3`.
- VAE-сжатие = 2^(len(block_out_channels)-1) = 2^3 = 8× по H и W (3 upsample-блока, последний up-блок без upsample).

---

## 2. Общая структура DECODE (точный forward)

Вход: `z : [B, 16, h, w]` (для FLUX 1024×1024 картинки latent = 128×128 после _unpack; h=w=128).
Выход: `image : [B, 3, H, W]`, H = h·8, W = w·8 (1024×1024). Затем пайплайн делает `image*0.5+0.5` clamp[0,1].

```
# (делает ПАЙПЛАЙН до вызова decoder)
z = latents / 0.3611 + 0.1159          # [B,16,h,w], f32

# === Decoder.forward ===
# (post_quant_conv ОТСУТСТВУЕТ для FLUX — пропускаем)
h0 = conv_in(z)                         # Conv2d 16->512, k3 s1 p1 -> [B,512,h,w]

# --- mid_block (UNetMidBlock2D) ---
h1 = resnets[0](h0)                     # ResnetBlock2D 512->512 -> [B,512,h,w]
h2 = attentions[0](h1)                  # VAE spatial self-attn (residual внутри) -> [B,512,h,w]
h3 = resnets[1](h2)                     # ResnetBlock2D 512->512 -> [B,512,h,w]

# --- up_blocks (4 шт, reversed channels [512,512,256,128]) ---
# up_block i: 3 resnet + (upsample если i != 3)
cur = h3
for i in 0..4:
    out_ch = [512,512,256,128][i]
    for r in 0..3:
        in_ch = (prev_ch if r==0 else out_ch)
        cur = resnet_i_r(cur)           # ResnetBlock2D in_ch->out_ch
    if i != 3:
        cur = upsample_i(cur)           # nearest x2 + Conv2d k3 s1 p1 (out_ch->out_ch)
    prev_ch = out_ch

# --- post ---
h = group_norm(cur, num_groups=32, eps=1e-6) ; h = silu(h)   # conv_norm_out (128ch) + SiLU
image = conv_out(h)                     # Conv2d 128->3, k3 s1 p1
```

### Точные формы по шагам (B=1, latent h=w=128, FLUX 1024²):

| шаг | tensor | форма |
|-----|--------|-------|
| вход z | z | [1,16,128,128] |
| conv_in | h0 | [1,512,128,128] |
| mid resnet0 | h1 | [1,512,128,128] |
| mid attn | h2 | [1,512,128,128] |
| mid resnet1 | h3 | [1,512,128,128] |
| up0 (3× resnet 512→512, +upsample) | | [1,512,256,256] |
| up1 (3× resnet 512→512, +upsample) | | [1,512,512,512] |
| up2 (3× resnet 512→256, +upsample) | | [1,256,1024,1024] |
| up3 (3× resnet 256→128, NO upsample) | | [1,128,1024,1024] |
| conv_norm_out+silu | | [1,128,1024,1024] |
| conv_out | image | [1,3,1024,1024] |

**Где меняются каналы внутри up-блока** (resnet[0] меняет каналы, conv_shortcut появляется только при смене):
- up0: resnets 512→512, 512→512, 512→512 (нет shortcut). upsample 512→512.
- up1: 512→512, 512→512, 512→512 (нет shortcut). upsample 512→512.
- up2: 512→256 (shortcut 1×1 512→256!), 256→256, 256→256. upsample 256→256.
- up3: 256→128 (shortcut 1×1 256→128!), 128→128, 128→128. НЕТ upsample.

Upsample увеличивает H,W ДО resnet'ов следующего блока? **НЕТ** — upsample в конце текущего блока (после его 3 resnet). Порядок: resnet,resnet,resnet, потом upsample. up2-resnet работают на 256×256, ИХ upsample → 512×512 на ВХОДЕ — нет. Перечитай таблицу: up0 resnet на 128², upsample → 256²; up1 resnet на 256², upsample → 512²; up2 resnet на 512², upsample → 1024²; up3 resnet на 1024², без upsample. Финальное разрешение = 1024².

---

## 3. ResnetBlock2D — точный forward (bit-exact)

Из `resnet.py:319`. Для VAE: `temb=None` (temb_channels=None → time_emb_proj=None), `time_embedding_norm="default"`, `non_linearity=silu`, `output_scale_factor=1.0`, `up=down=False`, `pre_norm=True`, eps=1e-6, groups=32.

```
def resnet(x):                          # x:[B,Cin,H,W]
    h = group_norm(x, ng=32, eps=1e-6, weight=norm1.w, bias=norm1.b)
    h = silu(h)
    h = conv1(h)                        # Conv2d Cin->Cout, k3 s1 p1
    # temb is None -> skip time_emb_proj и сложение temb
    h = group_norm(h, ng=32, eps=1e-6, weight=norm2.w, bias=norm2.b)
    h = silu(h)
    # dropout = identity (inference)
    h = conv2(h)                        # Conv2d Cout->Cout, k3 s1 p1
    res = (conv_shortcut(x) if use_in_shortcut else x)   # Conv2d 1×1 Cin->Cout (есть bias)
    out = (res + h) / 1.0               # output_scale_factor=1 -> деления нет
    return out
```

**Bit-exact гочи:**
- `use_in_shortcut = (in_channels != out_channels)` → conv_shortcut 1×1 stride1 pad0 **с bias** (`conv_shortcut_bias=True`). Появляется ТОЛЬКО в up2.resnets.0 (512→256) и up3.resnets.0 (256→128).
- Порядок сложения: **`input_tensor + hidden_states`** (residual ПЕРВЫМ слагаемым). У нас `conv2.forward_add(&h, &res)` → даёт `conv2(h)+res` = `hidden_states + input_tensor`. Сложение float коммутативно для одинаковых dtype → bit-identical. OK.
- groupnorm считается над [B,C,H,W] по группам каналов (C/32 каналов на группу), упкаст внутри group_norm в f32 (наш op делает; PyTorch GroupNorm в f32 если входной f32 — а тут весь decode f32).
- НЕТ деления на output_scale_factor (=1).

Наша `ResnetBlock2D::forward` уже точно это (norm1→silu→conv1→norm2→silu→conv2_add). Совпадает.

---

## 4. SiLU и GroupNorm — точные формулы

- **SiLU** (`nn.SiLU`, она же swish): `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`. Без аппроксимаций.
- **GroupNorm** (`nn.GroupNorm`, eps=1e-6, affine=True): для входа [B,C,H,W], группы G=32, каналов в группе Cg=C/32:
  ```
  reshape x -> [B, G, Cg*H*W]
  mean = mean over last dim ; var = var over last dim (biased, /N не /(N-1))
  xhat = (x - mean) / sqrt(var + eps)
  reshape back -> [B,C,H,W]
  y = xhat * weight[c] + bias[c]      # weight,bias формы [C], broadcast по H,W
  ```
  eps **внутри sqrt**: `sqrt(var + 1e-6)`. Статистики считаются в f32 (весь decode f32). num_groups=32 для ВСЕХ норм (128/32=4, 256/32=8, 512/32=16 каналов на группу — все делятся на 32).

---

## 5. VAE mid-block self-attention (`AttnProcessor2_0`) — точный forward

Из `attention_processor.py:2705`. Параметры из `UNetMidBlock2D`: `heads=1`, `dim_head=512`, `bias=True` (q,k,v,out все с bias), `norm_num_groups=32` (есть встроенный `group_norm` ВНУТРИ attention), `residual_connection=True`, `upcast_softmax=True`, `rescale_output_factor=1.0`, `scale_qk=True` → `scale=512**-0.5`. `spatial_norm=None`, `norm_cross=None`, `norm_q=norm_k=None`.

```
def vae_attn(x):                        # x:[B,512,H,W], H=W=h (latent res)
    residual = x
    # input_ndim == 4:
    B,C,H,W = x.shape ; S = H*W
    hs = x.view(B, C, S).transpose(1,2)         # [B, S, C]  (НЕ contiguous обязательно)
    # group_norm ВНУТРИ attention: transpose(1,2) -> [B,C,S], GN, transpose back
    hs = group_norm(hs.transpose(1,2), ng=32, eps=1e-6, w=group_norm.w, b=group_norm.b).transpose(1,2)  # [B,S,C]
    q = to_q(hs)                                # Linear C->C (+bias)  [B,S,C]
    k = to_k(hs) ; v = to_v(hs)                 # Linear C->C (+bias)
    # heads=1, head_dim = C/1 = C = 512
    q = q.view(B, S, 1, 512).transpose(1,2)     # [B,1,S,512]
    k = k.view(B, S, 1, 512).transpose(1,2)
    v = v.view(B, S, 1, 512).transpose(1,2)
    # SDPA: scale = 1/sqrt(512) (default = 1/sqrt(head_dim))
    out = sdpa(q, k, v, scale=512**-0.5, is_causal=False, attn_mask=None)  # [B,1,S,512]
    out = out.transpose(1,2).reshape(B, S, 512)  # [B,S,C]
    out = to_out[0](out)                         # Linear C->C (+bias)
    # to_out[1] = Dropout = identity
    out = out.transpose(-1,-2).reshape(B, C, H, W)  # обратно в [B,C,H,W]
    out = out + residual                         # residual_connection=True
    out = out / 1.0                              # rescale_output_factor=1
    return out
```

**SDPA внутренне (bit-exact):**
```
attn = softmax( (q @ k^T) * scale , dim=-1 )    # [B,1,S,S]
out  = attn @ v                                 # [B,1,S,512]
```
- `scale = 512**-0.5 ≈ 0.044194173824159216`. Применяется к (q·kᵀ), НЕ к q отдельно. (В нашем `scaled_dot_attention(q,k,v,scale,None)` scale передаётся как множитель скоров — совпадает.)
- **`upcast_softmax=True`**: softmax считается в f32. Так как весь decode уже в f32 (force_upcast), различий нет, но при f16-пути софтмакс ОБЯЗАН быть f32.
- `attn_mask=None`, `is_causal=False` — полный bidirectional self-attention по spatial-позициям.
- Один head, head_dim=512=C → фактически full-rank single-head attention над всеми H·W spatial-позициями.

Наша `VaeAttention::forward` делает то же (GN→[B,HW,C]→q/k/v→reshape[B,1,HW,C]→SDPA scale=C^-0.5→to_out→back→+x). **Bit-exact совпадает.** Единственная разница раскладки: diffusers использует `view(B,C,S).transpose(1,2)` без принудительного contiguous; наш код делает `reshape→permute→contiguous`. Результат численно тождественен (Linear применяется к одинаковым числам). OK.

**Размеры attention для FLUX 1024²:** mid работает на latent-разрешении h=w=128 (после conv_in, до любого upsample), S = 128·128 = 16384. q·kᵀ = [B,1,16384,16384] — большая матрица (1.07 ГБ f32 на батч). Для bit-exact это нормально; для перфа на 1024² можно flash-attention, но числа должны совпасть с naive softmax(qkᵀ·scale)@v в f32.

---

## 6. Upsample2D (nearest x2 + conv) — точный forward

Из `upsampling.py:140`, `use_conv=True`, `use_conv_transpose=False`, `interpolate=True`, kernel_size=3, padding=1, norm=None.

```
def upsample(x):                        # x:[B,C,H,W]
    # norm=None -> skip
    # F.interpolate(scale_factor=2.0, mode="nearest") — ТОЧНАЯ дупликация пикселей
    x = nearest_upsample_2x(x)          # [B,C,2H,2W]
    x = conv(x)                         # Conv2d C->C, k3 s1 p1 (+bias)
    return x
```

**Bit-exact гочи:**
- `mode="nearest"` со scale_factor=2 = каждый пиксель копируется в блок 2×2. Bit-exact формула индекса (PyTorch nearest): `out[i,j] = in[floor(i/2), floor(j/2)]`. Для целого ×2 = простая дупликация (наш `upsample_nearest2x` через cat/reshape это даёт).
- В bf16-пути PyTorch апкастит до f32 для interpolate (issue 86679) и кастит назад — но при force_upcast (f32 decode) этого не происходит, разницы нет.
- conv ПОСЛЕ upsample (не до). kernel=3 padding=1 stride=1 → размеры сохраняются.
- НЕТ ConvTranspose (это другой путь, `UpSample` класс — НЕ используется в AutoencoderKL).

Наш код: `up.forward(&upsample_nearest2x(&h)?)` = nearest потом conv. **Совпадает.**

---

## 7. force_upcast / dtype (КРИТИЧНО для bit-exact)

- `force_upcast=true` → diffusers гонит VAE forward в **float32**, даже если веса хранятся BF16. В пайплайне: `self.vae.to(dtype=torch.float32)` (либо upcast внутри `decode` через `_decode` с автокастом). На практике: входной z и ВСЕ веса VAE кастятся в f32 перед decode, decode целиком в f32, выход f32.
- В нашем файле веса хранятся **BF16** (проверено `dtype: BF16`). Для bit-exact надо: при загрузке **сконвертировать каждый вес BF16→F32** (или грузить как f32), и считать decode в f32. Если оставить BF16-веса и f32-активации — conv/linear будут смешивать dtype → расхождение. **Грузить веса в F32.**
- Reference-проверка: сгенерировать z, прогнать `AutoencoderKL.from_pretrained(...).to(torch.float32)`, `vae.decode(z/0.3611+0.1159).sample`, сравнить наш f32-выход. Гейт: per-row/per-pixel max-abs (НЕ глобальный cos — см. память про cos-гейт скрывающий per-row баг), целевая точность ~1e-4..1e-5 на f32.

---

## 8. Точный список весов decoder (HF state_dict) и формы

Префиксы: `decoder.*`. **`post_quant_conv` и `quant_conv` ОТСУТСТВУЮТ** (не грузить!). Полный список ключей см. поле `weight_keys`.

Loader-схема (config-driven, наш `AutoencoderKlDecoder::load`):
- `decoder.conv_in.{weight[512,16,3,3],bias[512]}` — Conv2d 16→512 k3 p1.
- `decoder.mid_block.resnets.{0,1}.{norm1,norm2}.{weight,bias}[512]`, `.{conv1,conv2}.{weight[512,512,3,3],bias[512]}`.
- `decoder.mid_block.attentions.0.group_norm.{weight,bias}[512]`, `.{to_q,to_k,to_v,to_out.0}.{weight[512,512],bias[512]}`.
- `decoder.up_blocks.{i}.resnets.{r}.*` для i∈0..3, r∈0..2 (3 resnet на блок):
  - norm/conv формы зависят от каналов (см. таблицу §2). conv_shortcut ТОЛЬКО у up2.resnets.0 (512→256, weight[256,512,1,1]) и up3.resnets.0 (256→128, weight[128,256,1,1]).
- `decoder.up_blocks.{i}.upsamplers.0.conv.{weight,bias}` для i∈0..2 (НЕ у i=3): up0,up1 [512,512,3,3]; up2 [256,256,3,3].
- `decoder.conv_norm_out.{weight,bias}[128]` — GroupNorm 32 groups eps1e-6.
- `decoder.conv_out.{weight[3,128,3,3],bias[3]}` — Conv2d 128→3 k3 p1.

Linear-веса PyTorch хранятся как [out,in] → matmul = `x @ W^T + b`. Наш `Linear::forward` это уже учитывает.

---

## 9. Изменения в коде (минимальный diff к autoencoder_kl.rs)

1. **Config:** добавить `pub fn flux()` в `AutoencoderKlConfig` + поля `shift_factor: Option<f32>`, `use_post_quant_conv: bool`:
   ```
   AutoencoderKlConfig {
     in_channels:3, out_channels:3, latent_channels:16,
     block_out_channels: vec![128,256,512,512], layers_per_block:2,
     norm_num_groups:32, norm_eps:1e-6, scaling_factor:0.3611,
     // НОВЫЕ: shift_factor: Some(0.1159), use_post_quant_conv: false,
   }
   ```
2. **`AutoencoderKlDecoder`:** `post_quant_conv: Option<Conv2dLayer>`. В `load`: грузить только `if cfg.use_post_quant_conv`. В `decode`: `if let Some(pq)=&self.post_quant_conv { z = pq.forward(z)? }`.
3. **Веса:** грузить decoder.* в F32 (конвертировать из BF16) при force_upcast.
4. **API decode с scale/shift** (в txt2img-обёртке, НЕ в decoder.decode):
   ```
   z = latents.div_scalar(0.3611)?.add_scalar(0.1159)?   # /scale + shift
   image = decoder.decode(&z)?
   image = image.mul_scalar(0.5)?.add_scalar(0.5)?.clamp(0,1)?   # денорм для png
   ```
   `decoder.decode` остаётся без scale/shift (raw sample).

Всё остальное (ResnetBlock2D, VaeAttention, UpBlock, upsample_nearest2x, GroupNormLayer, conv_out) — **не трогать, уже bit-exact.**

---

## 10. Псевдокод полного decode (для сверки)

```
fn flux_vae_decode(latents[B,16,h,w]) -> image[B,3,h*8,w*8]:
    z = latents / 0.3611 + 0.1159                       # f32
    h = conv_in(z)                                       # ->[B,512,h,w]
    # mid
    h = resnet(h, mid.resnets.0)                         # 512->512
    h = vae_attn(h, mid.attentions.0)                    # spatial self-attn, scale=512^-0.5, +residual
    h = resnet(h, mid.resnets.1)                         # 512->512
    # up
    rev = [512,512,256,128]; prev = 512
    for i in 0..4:
        oc = rev[i]
        for r in 0..3:
            ic = prev if r==0 else oc
            h = resnet(h, up[i].resnets[r])              # ic->oc, shortcut если ic!=oc
        if i != 3:
            h = nearest_upsample_2x(h)                   # x2 H,W
            h = conv(h, up[i].upsamplers.0.conv)         # oc->oc k3 p1
        prev = oc
    # out
    h = group_norm(h, 32, 1e-6, conv_norm_out.w, conv_norm_out.b)
    h = silu(h)
    image = conv_out(h)                                  # 128->3 k3 p1
    return image                                         # raw sample [-~1,~1]
```


## WEIGHT KEYS
DECODER (префикс decoder.*; post_quant_conv/quant_conv ОТСУТСТВУЮТ):

conv_in:
  decoder.conv_in.weight [512,16,3,3]   decoder.conv_in.bias [512]

mid_block.resnets.{0,1} (512->512):
  decoder.mid_block.resnets.{r}.norm1.weight [512]  .norm1.bias [512]
  decoder.mid_block.resnets.{r}.conv1.weight [512,512,3,3]  .conv1.bias [512]
  decoder.mid_block.resnets.{r}.norm2.weight [512]  .norm2.bias [512]
  decoder.mid_block.resnets.{r}.conv2.weight [512,512,3,3]  .conv2.bias [512]
  (нет conv_shortcut — каналы не меняются)

mid_block.attentions.0 (single-head, dim_head=512, heads=1):
  decoder.mid_block.attentions.0.group_norm.weight [512]  .group_norm.bias [512]
  decoder.mid_block.attentions.0.to_q.weight [512,512]  .to_q.bias [512]
  decoder.mid_block.attentions.0.to_k.weight [512,512]  .to_k.bias [512]
  decoder.mid_block.attentions.0.to_v.weight [512,512]  .to_v.bias [512]
  decoder.mid_block.attentions.0.to_out.0.weight [512,512]  .to_out.0.bias [512]

up_blocks (i∈0..3, reversed channels [512,512,256,128]; resnets r∈0..2):
  up_blocks.0 (все 512->512, нет shortcut):
    decoder.up_blocks.0.resnets.{0,1,2}.norm1.{weight,bias}[512]
    decoder.up_blocks.0.resnets.{0,1,2}.conv1.{weight[512,512,3,3],bias[512]}
    decoder.up_blocks.0.resnets.{0,1,2}.norm2.{weight,bias}[512]
    decoder.up_blocks.0.resnets.{0,1,2}.conv2.{weight[512,512,3,3],bias[512]}
    decoder.up_blocks.0.upsamplers.0.conv.{weight[512,512,3,3],bias[512]}
  up_blocks.1 (все 512->512, нет shortcut, идентичные формы up_blocks.0):
    decoder.up_blocks.1.resnets.{0,1,2}.{norm1,norm2}.{weight,bias}[512]
    decoder.up_blocks.1.resnets.{0,1,2}.{conv1,conv2}.{weight[512,512,3,3],bias[512]}
    decoder.up_blocks.1.upsamplers.0.conv.{weight[512,512,3,3],bias[512]}
  up_blocks.2 (resnet0: 512->256 + shortcut; resnet1,2: 256->256; upsample 256):
    decoder.up_blocks.2.resnets.0.norm1.{weight,bias}[512]
    decoder.up_blocks.2.resnets.0.conv1.weight[256,512,3,3]  .conv1.bias[256]
    decoder.up_blocks.2.resnets.0.norm2.{weight,bias}[256]
    decoder.up_blocks.2.resnets.0.conv2.weight[256,256,3,3]  .conv2.bias[256]
    decoder.up_blocks.2.resnets.0.conv_shortcut.weight[256,512,1,1]  .conv_shortcut.bias[256]
    decoder.up_blocks.2.resnets.{1,2}.{norm1,norm2}.{weight,bias}[256]
    decoder.up_blocks.2.resnets.{1,2}.{conv1,conv2}.{weight[256,256,3,3],bias[256]}
    decoder.up_blocks.2.upsamplers.0.conv.weight[256,256,3,3]  .conv.bias[256]
  up_blocks.3 (resnet0: 256->128 + shortcut; resnet1,2: 128->128; БЕЗ upsample):
    decoder.up_blocks.3.resnets.0.norm1.{weight,bias}[256]
    decoder.up_blocks.3.resnets.0.conv1.weight[128,256,3,3]  .conv1.bias[128]
    decoder.up_blocks.3.resnets.0.norm2.{weight,bias}[128]
    decoder.up_blocks.3.resnets.0.conv2.weight[128,128,3,3]  .conv2.bias[128]
    decoder.up_blocks.3.resnets.0.conv_shortcut.weight[128,256,1,1]  .conv_shortcut.bias[128]
    decoder.up_blocks.3.resnets.{1,2}.{norm1,norm2}.{weight,bias}[128]
    decoder.up_blocks.3.resnets.{1,2}.{conv1,conv2}.{weight[128,128,3,3],bias[128]}
    (НЕТ upsamplers)

out:
  decoder.conv_norm_out.weight [128]  decoder.conv_norm_out.bias [128]
  decoder.conv_out.weight [3,128,3,3]  decoder.conv_out.bias [3]

ИТОГО decoder = 138 ключей. Все веса dtype=BF16 в файле -> грузить как F32 (force_upcast).
Файл: models/black-forest-labs/FLUX.1-dev/vae/diffusion_pytorch_model.safetensors (244 ключа: 138 decoder + 106 encoder; нет top-level quant_conv/post_quant_conv).

## GOTCHAS
BIT-EXACT подводные камни FLUX VAE decode:

1. SCALE/SHIFT ФОРМУЛА: decode-вход = latents / 0.3611 + 0.1159 (ДЕЛЕНИЕ на scale, СЛОЖЕНИЕ shift). НЕ (z-shift)*scale, НЕ z*scale. Источник: pipeline_flux.py:1010 `latents = (latents / scaling_factor) + shift_factor`. Encode (если нужен) = (mean - shift) * scale. Применяет ПАЙПЛАЙН, не сам decoder — decoder.decode даёт raw sample.

2. НЕТ post_quant_conv/quant_conv: ключей в state_dict вообще нет. Наш текущий AutoencoderKlDecoder::load БЕЗУСЛОВНО грузит post_quant_conv.* -> для FLUX упадёт. Сделать Option и грузить только if use_post_quant_conv=true (для FLUX false). latent идёт прямо в conv_in.

3. force_upcast=true: веса в файле BF16, но decode целиком в F32. Грузить decoder-веса как F32 (конвертировать BF16->F32). Смешивание BF16-веса + F32-актив = расхождение. Reference: vae.to(float32).decode(...).

4. latent_channels=16 (не 4): conv_in.weight = [512,16,3,3]. Только это меняется в структуре; всё config-driven.

5. GroupNorm eps = 1e-6 ВЕЗДЕ (норм в resnet, attn group_norm, conv_norm_out), внутри sqrt: sqrt(var+1e-6). var = biased (делить на N, не N-1). num_groups=32 для всех (128/256/512 все делятся на 32). Статистики в f32.

6. output_scale_factor=1.0 -> в ResnetBlock2D НЕТ деления `(input+hidden)/scale`. residual + conv2(h), коммутативно.

7. conv_shortcut появляется ТОЛЬКО при смене каналов: up2.resnets.0 (512->256, 1×1 weight[256,512,1,1] +bias) и up3.resnets.0 (256->128, weight[128,256,1,1] +bias). conv_shortcut_bias=True (есть bias). НЕ забыть, иначе residual-форма не сойдётся.

8. mid attention = SINGLE head: attention_head_dim=512=in_channels -> heads = 512//512 = 1, head_dim=512. scale = dim_head**-0.5 = 512**-0.5 ≈ 0.0441941738 (применяется к q·kᵀ, default SDPA scale=1/sqrt(head_dim)). residual_connection=True (out += x). upcast_softmax=True (softmax в f32). bias=True на to_q/k/v/to_out.0. rescale_output_factor=1. norm_q/norm_k=None, нет qk-нормы. Внутренний group_norm перед q/k/v (ng=32, eps=1e-6). attn_mask=None, is_causal=False (full bidirectional).

9. attention раскладка: diffusers view(B,C,HW).transpose(1,2) -> [B,HW,C] БЕЗ обязательного contiguous; наш reshape+permute+contiguous численно тождественен (Linear к одинаковым числам). После to_out: transpose(-1,-2).reshape(B,C,H,W). to_out[1]=Dropout=identity в inference.

10. Upsample = nearest x2 (mode="nearest", scale_factor=2, точная дупликация floor(i/2)) ПОТОМ conv2d k3 s1 p1 (+bias). conv ПОСЛЕ upsample. НЕ ConvTranspose (класс UpSample с deconv в vae.py НЕ используется AutoencoderKL). Последний up-блок (i=3) БЕЗ upsample.

11. SiLU = x*sigmoid(x) = x/(1+exp(-x)), без tanh-аппроксимации. nonlinearity во всех resnet + conv_norm_out post-act.

12. Порядок up-блоков: 3 resnet СНАЧАЛА, потом upsample (в конце блока). Каналы меняет resnets[0]. Разрешение: up0 на 128² ->upsample 256², up1 256²->512², up2 512²->1024², up3 1024² без upsample. Финал 1024² (8× от latent 128²).

13. mid-attention на FLUX 1024² работает на S=16384 (latent 128²), qkᵀ = [B,1,16384,16384] ~1ГБ f32. Bit-exact = naive softmax((qkᵀ)*scale, dim=-1)@v в f32. Flash-attn можно для перфа, но числа сверять с naive.

14. Denorm выхода для PNG (делает image_processor): image*0.5+0.5, clamp(0,1). НЕ часть decoder.decode.

15. Linear-веса [out,in] -> y = x@Wᵀ+b. Conv-веса [out_ch,in_ch,kh,kw]. PyTorch conv pad=1 для k3 (same), pad=0 для k1.

16. temb=None во всём VAE (temb_channels=None): time_emb_proj отсутствует, в resnet НЕ складывается temb, time_embedding_norm path для "default" просто norm2 без модуляции.

17. Гейтить корректность per-pixel/per-row max-abs, НЕ глобальным cos (cos скрывает локальные баги в паре строк — см. историю про "теряет контекст"). Целевая точность f32 ~1e-4..1e-5.
