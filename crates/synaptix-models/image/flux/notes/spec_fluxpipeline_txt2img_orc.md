# FluxPipeline (txt2img orchestration, FLUX.1-dev)

FluxPipeline txt2img bit-exact. Источник diffusers/pipelines/flux/pipeline_flux.py. ТОЛЬКО оркестрация. B=батч(промпты*nimg, обычно 1), H=W=1024 дефолт. guidance-distilled: БЕЗ CFG, 1 forward/шаг.

КОНСТАНТЫ: vae_scale_factor=2**(4-1)=8. image_processor с factor*2=16. default_sample_size=128 -> H=W=128*8=1024. num_channels_latents=in_channels(64)//4=16. VAE: scaling_factor=0.3611, shift_factor=0.1159, no quant/post_quant_conv, force_upcast=true. Scheduler FlowMatchEuler: num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=true, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096, time_shift_type=exponential, stochastic=False. CLIP: 12 слоёв, h=768, LN eps=1e-5, quick_gelu, eos_id=2(config), токены bos=49406 eos=pad=49407. T5: 24 слоя, d_model=4096, RMSNorm eps=1e-6, gelu_new, gated-gelu, eos=1 pad=0, max=512. Дефолт: steps=28, guidance_scale=3.5, true_cfg_scale=1.0(выкл), nimg=1.
Активации: quick_gelu(x)=x*sigmoid(1.702*x); gelu_new=0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))); silu(x)=x*sigmoid(x).

1) _get_clip_prompt_embeds -> pooled[B,768]: tokenizer padding=max_length,max_length=77,truncation -> ids[B,77]. CLIPTextModel(output_hidden_states=False) -> .pooler_output (НЕ last_hidden_state). каст bf16. repeat(1,nimg)+view -> [B,768]. ПУЛИНГ (modeling_clip 565-588): last_hidden_state=final_layer_norm(enc) (eps=1e-5 после 12 слоёв); eos_token_id==2 => pooled=last_hidden_state[arange(B), input_ids.argmax(-1)]. argmax = позиция первого токена 49407 (макс id=eos). Порт: pos=первый индекс id 49407. НЕ id=2.

2) _get_t5_prompt_embeds -> [B,512,4096]: T5TokenizerFast padding=max_length,max_length=512,truncation -> ids[B,512]. T5EncoderModel(output_hidden_states=False)[0]=last_hidden_state. attention_mask НЕ передаётся (FLUX не маскирует T5). каст bf16. repeat(1,nimg,1)+view -> [B,512,4096].

3) encode_prompt -> (prompt_embeds[B,512,4096]=T5, pooled[B,768]=CLIP, text_ids=zeros(512,3) bf16 без батча, всегда нули).

4) prepare_latents: H_lat=2*(H//16), W_lat=2*(W//16) -> 128,128. shape=[B,16,128,128]. latents_raw=randn_tensor(shape,gen,device,bf16) N(0,1).
_pack_latents [B,16,128,128]->[B,4096,64]: view(B,16,64,2,64,2) оси(B,C,h,ph,w,pw); permute(0,2,4,1,3,5) оси(B,h,w,C,ph,pw); reshape(B,4096,64). seq=h*64+w, feat=c*4+ph*2+pw. РЕАЛЬНОЕ переразложение. image_seq_len=4096.
_prepare_latent_image_ids(_,64,64): ids=zeros(64,64,3); ids[...,1]+=arange(64)[:,None]; ids[...,2]+=arange(64)[None,:]; reshape(4096,3).to(device,bf16). к0=0, к1=row, к2=col. seq=row*64+col. форма[4096,3] 2D.

5) __call__:
- H=H or 1024, W=W or 1024. B_eff=B*nimg. do_true_cfg=False (true_cfg=1.0).
- encode_prompt -> prompt_embeds, pooled, text_ids[512,3].
- prepare_latents -> latents[B,4096,64], latent_image_ids[4096,3].
- sigmas=np.linspace(1.0,1/28,28). image_seq_len=latents.shape[1]=4096. mu=calculate_shift(4096,256,4096,0.5,1.15): m=0.65/3840, b=0.5-m*256, mu=4096*m+b=1.15.
- set_timesteps(use_dynamic_shifting=True): sigmas=time_shift_exp(mu,1,s)=exp(mu)/(exp(mu)+(1/s-1)); timesteps=sigmas*1000 [28]f32 убыв; sigmas=cat([sigmas,0]) [29] last=0. begin_index=0.
- guidance=full([1],3.5).expand(B_eff) float32 (guidance_embeds=true).
- LOOP 28: timestep=t.expand(B_eff).to(bf16); noise_pred=transformer(hidden_states=latents, timestep=timestep/1000, guidance, pooled_projections=pooled, encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_image_ids, return_dict=False)[0]; latents=scheduler.step(noise_pred,t,latents,return_dict=False)[0]. (один forward, neg пропущен.)
  step (stochastic=False): sample.to(float32); sigma=sigmas[idx]; sigma_next=sigmas[idx+1]; dt=sigma_next-sigma (отриц); prev=sample+dt*noise_pred в f32; idx+=1; prev.to(bf16). Эйлер flow-match, model_output=скорость.
- ДЕКОД: _unpack_latents(latents,H,W,8): view(B,64,64,16,2,2); permute(0,3,1,4,2,5); reshape(B,16,128,128). Обратное к pack. latents=latents/0.3611+0.1159 (ДЕЛЕНИЕ первым). image=vae.decode(latents)[0] [B,3,1024,1024] ~[-1,1]. postprocess(do_normalize=True): image=(image*0.5+0.5).clamp(0,1); permute(0,2,3,1); *255 round uint8 -> PIL.

ФОРМЫ B=1 1024кв: clip_ids[1,77]i64; pooled[1,768]bf16; t5_ids[1,512]i64; prompt_embeds[1,512,4096]bf16; text_ids[512,3]bf16; latents_raw[1,16,128,128]; latents[1,4096,64]; latent_image_ids[4096,3]; guidance[1]f32=3.5; timesteps[28]f32; sigmas[29]f32 last=0; timestep->transformer=t/1000 bf16; t->step=скаляр f32; noise_pred[1,4096,64]; unpacked[1,16,128,128]; vae_out[1,3,1024,1024]; final[1,1024,1024,3]uint8. mu=1.15, image_seq_len=4096.

## WEIGHT KEYS
Оркестрация без весов (склейка). Ключи под-моделей по каталогам model_index.json.

CLIP (text_encoder/): text_model.embeddings.{token_embedding.weight[49408,768], position_embedding.weight[77,768]}; text_model.encoder.layers.{0..11}.{self_attn.{q,k,v,out}_proj.{weight,bias}, layer_norm1.{w,b}, mlp.fc1.{w,b}, mlp.fc2.{w,b}, layer_norm2.{w,b}}; text_model.final_layer_norm.{weight,bias} (eps=1e-5, нужен для пулинга). text_projection НЕ используется (pooler_output = last_hidden_state на EOS).

T5 (text_encoder_2/): shared.weight[32128,4096]; encoder.block.{0..23}.layer.0.SelfAttention.{q,k,v,o}.weight (без bias); encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight[32,64] (ТОЛЬКО block0, общий); encoder.block.{0..23}.layer.0.layer_norm.weight (RMS eps=1e-6); encoder.block.{0..23}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight (gated-gelu, без bias); encoder.block.{0..23}.layer.1.layer_norm.weight; encoder.final_layer_norm.weight.

TRANSFORMER (transformer/, отдельный спек): x_embedder, context_embedder, time_text_embed.{timestep_embedder,guidance_embedder,text_embedder}, transformer_blocks.{0..18}.*, single_transformer_blocks.{0..37}.*, norm_out.*, proj_out.*

VAE (vae/, ТОЛЬКО decoder для txt2img): decoder.conv_in.{w,b}, decoder.mid_block.*, decoder.up_blocks.{0..3}.*, decoder.conv_norm_out.{w,b}, decoder.conv_out.{w,b}. encoder.* НЕ нужен. quant/post_quant_conv ОТСУТСТВУЮТ.

SCHEDULER без весов. TOKENIZERS: tokenizer/ (CLIP vocab.json+merges.txt), tokenizer_2/ (T5 spiece.model).
Из конфигов нужны: vae.scaling_factor=0.3611, vae.shift_factor=0.1159, transformer.in_channels=64, guidance_embeds=true, scheduler shift=3.0/use_dynamic_shifting=true/base_shift=0.5/max_shift=1.15/base_image_seq_len=256/max_image_seq_len=4096/num_train_timesteps=1000.

## GOTCHAS
BIT-EXACT КАМНИ:
1. CLIP-пулинг = last_hidden_state[arange(B), input_ids.argmax(-1)] (eos_token_id==2 => легаси-ветка). argmax = первый токен 49407 (endoftext, макс id), НЕ id=2. final_layer_norm (eps=1e-5) ДО пулинга обязателен. Берём pooler_output, не весь last_hidden_state, не text_projection.
2. T5 БЕЗ attention_mask: padding-токены (id=0) участвуют в self-attention. Маскировать = расхождение. Всегда паддинг до 512.
3. ДВА timestep в loop: transformer <- timestep/1000 (bf16, в (0,1]); scheduler.step <- сырое t (f32, до 1000). Деление /1000 только на входе трансформера.
4. scheduler.step апкаст в f32: sample.to(float32), prev=sample+dt*noise_pred в f32, потом .to(bf16). dt=sigma_next-sigma ОТРИЦАТЕЛЬНЫЙ. Эйлер flow-match: prev=x+dt*v (model_output=скорость).
5. dynamic shifting: sigmas=exp(mu)/(exp(mu)+(1/s-1)) от linspace(1.0,1/N,N), mu=1.15 (1024кв). НЕ статический shift=3.0. Затем timesteps=sigmas*1000, sigmas дополняется 0.0 (len N+1).
6. pack permute=(0,2,4,1,3,5), unpack permute=(0,3,1,4,2,5), round-trip identity. reshape после permute = реальное переразложение (row-major). seq=row*(W_lat/2)+col, feat=c*4+ph*2+pw. latent_image_ids в том же row-major порядке иначе RoPE-координаты разъедутся.
7. latent-размеры: H_lat=2*(H//16), W_lat=2*(W//16) (//16*2 для чётности под 2x2 pack). num_channels_latents=in_channels//4=16.
8. VAE-денорм ПОРЯДОК: latents/scaling_factor(0.3611) ПОТОМ +shift_factor(0.1159). force_upcast=true. нет post_quant_conv.
9. text_ids[512,3] и latent_image_ids[4096,3] — 2D без батча. text_ids всегда нули. img_ids: к0=0, к1=row(0..63), к2=col(0..63).
10. guidance=full([1],3.5).expand(B) float32 (НЕ bf16). guidance_embeds=true всегда для FLUX.1-dev.
11. Финал-денорм: (image*0.5+0.5).clamp(0,1) -> *255 round uint8. do_normalize=True, безусловно.
12. randn_tensor: CPU-генератор -> шум на CPU нужным dtype -> .to(device). Bit-exact требует репликации torch RNG; иначе гейтить подачей внешнего latents (latents!=None пропускает И генерацию И pack -> latents подаются УЖЕ packed [B,4096,64]).
13. true_cfg_scale=1.0 дефолт => do_true_cfg=False => один forward/шаг, neg-ветка и noise_pred=neg+scale*(pos-neg) НЕ выполняются.
14. begin_index=0 (set_begin_index(0)) => step_index 0..27 синхронно с i, без nonzero-поиска после первого шага.
15. dtype: bf16 для эмбеддингов/латентов/noise_pred; f32 для guidance, timesteps/sigmas, и внутри step (upcast). Расхождение dtype = дрейф за 28 шагов.
