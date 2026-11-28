# kijai__ComfyUI-KJNodes__issues-minimax-h3

Source: https://github.com/kijai/ComfyUI-KJNodes/issues?q=h3
Fetched: 2026-11-28



---

# ISSUES in kijai/ComfyUI-KJNodes matching "H3" (top 34 by comments)


### #750 [closed] MiniMax H3 Low VRAM attention - breakage
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/750  by SveSop 2026-09-06T16:27:16Z  comments: 27

It seems that the `MiniMax H3 VRAM Attention`  node breaks with this error:

```
[INFO] Comfy model compiler graph breaks: 0
[ERROR] !!! Exception during processing !!! minimax_block_lowmem_forward() got an unexpected keyword argument 'attention'
[ERROR] Traceback (most recent call last):
  File "C:\ComfyUI\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "C:\ComfyUI\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "C:\ComfyUI\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "C:\ComfyUI\ComfyUI\comfy_api\latest\_io.py", line 2046, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "C:\ComfyUI\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "C:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "C:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "C:\ComfyUI\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "C:\ComfyUI\ComfyUI\comfy\k_diffusion\sampling.py", line 1619, in sample_er_sde
    denoised = model(x, sigmas[i] * s_in, **extra_args)
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\model_base.py", line 208, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\model_base.py", line 252, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "C:\ComfyUI\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 577, in forward
    graph_out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload,
              denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
  File "C:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 753, in _forward
    h = blocks_replace[("double_block", i)](
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         "layout": layout, "transformer_options": transformer_options},
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        {"original_block": block_wrap})["img"]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy_extras\nodes_sparse_attention.py", line 303, in block_patch
    return extra["original_block"](args)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\ComfyUI\ComfyUI\comfy\ldm\minimax\model.py", line 751, in block_wrap
    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                   ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         transformer_options=args["transformer_options"], attention=args.get("attention"))}
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
TypeError: minimax_block_lowmem_forward() got an unexpected keyword argument 'attention'
```

This was using the new https://github.com/Comfy-Org/ComfyUI/commit/e308cc73b466584b0c17be695e5de1a17438bb40 node, but it also breaks with other nodes like: https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes/tree/main/ComfyUI-H3-SLA-Attention when using CK attention.

Using sageattention `Patch Sage Attention KJ` makes the https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes/tree/main/ComfyUI-H3-SLA-Attention work together with `MiniMax H3 Low VRAM attention` node.

So, maybe some comfy-kitchen changes + sparse attention messes up with the Low VRAM node?

> **Comment by kijai (2026-09-06T17:44:14Z):**

Should be fixed now, but the low VRAM attention shouldn't be needed with the comfy core sparse attention.


> **Comment by SveSop (2026-09-06T18:09:54Z):**

Trying to squeeze every byte of vram max you know 👍

Separate issue tho, but after the comfy-aimdo update to 0.5.2 (or backend comfy-compiler update), i am no longer able to use the `Model Preview Override` node with H3 anymore. Generations seems to use up to 96% of my 16GB vram for most 10s video's.. (before it was 75-80%) This seems to work just fine normally, but the added overhead from the preview node causes vram to spill into shared mem, and the generation grind to a halt, going from 35-40 s/it -> 100-130 s/it ..

Disadvantages to use nightly branch i suppose, and i do not think its the preview node's "fault" in a way, but rather vram budget calculation/usages after the comfy-aimdo 0.5.xx branch perhaps?


> **Comment by kijai (2026-09-06T20:08:21Z):**

> Trying to squeeze every byte of vram max you know 👍
> 
> Separate issue tho, but after the comfy-aimdo update to 0.5.2 (or backend comfy-compiler update), i am no longer able to use the `Model Preview Override` node with H3 anymore. Generations seems to use up to 96% of my 16GB vram for most 10s video's.. (before it was 75-80%) This seems to work just fine normally, but the added overhead from the preview node causes vram to spill into shared mem, and the generation grind to a halt, going from 35-40 s/it -> 100-130 s/it ..
> 
> Disadvantages to use nightly branch i suppose, and i do not think its the preview node's "fault" in a way, but rather vram budget calculation/usages after the comfy-aimdo 0.5.xx branch perhaps?

That's weird, there's no added VRAM cost of the preview since it's done between steps.


> **Comment by SveSop (2026-09-06T22:28:36Z):**

FL2VA MiniMax H3 model
LightX2V_4step_v1.2 turbo lora (str 0.7)
ModelAttentionBackend = comfy kitchen attention
(No additional SLA/Low VRAM/other tweak nodes)
Steps 8, ER_SDE/Beta
1280x736 x 10s

`Model Preview Override` disabled:
```
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [07:01<00:00, 52.67s/it]
[INFO] Comfy model compiler graph breaks: 1
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 2677MB Staged. 0 patches attached. Force pre-loaded 128 weights: 692 KB.
[INFO] Prompt executed in 484.26 seconds
```
Same workflow/prompt/seed/models with `Model Preview Override` enabled:
```
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
 62%|█████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                             | 5/8 [06:39<05:50, 116.79s/it]
```
Tends to get slower and slower...

This started happening yesterday after i pulled: https://github.com/Comfy-Org/ComfyUI/commit/250b2e9551a7bc7a8ebb5beb07e0fecd2983e04a

Suspect: The comfy-aimdo 0.5.2 update, or possibly the Comfy-compiler . I had comfy-aimdo 0.4.15 before the update.

GPU Load is 100%, but temp is < 50 degrees, indicating low work. Usually is around 70-72 degrees when running. 4070Ti Super (16GB). Shared GPU Memory seems to grow.. 2.7GB when the above happened.
I can run 1344x768 x 15s without stalling/slowdown issues, when i disable the preview node.

I do not immediately think its the preview node doing anything wrong, so i did not want to report that node as a bug... Then again, i have zero clue of all the amounts of various models and PR's on the comfy main git repo.. 

Feel free to close this as non-relevant tho 😄 


> **Comment by Castelmatto (2026-09-07T08:59:39Z):**

I’ve tested the new comfy core sparse attention node Kijai have just implemented (tested at default values using Sol-attn) and when active it causes a noticeable progressive increase in memory allocation as if leaking memory. Is it something to be expected? I have not tested yet if https://github.com/Comfy-Org/ComfyUI/commit/a99d1f9c14f4594fe1e9d7e17c1997bf104d89fb fixes this problem 


> **Comment by SveSop (2026-09-07T21:43:58Z):**

@kijai 
https://github.com/Comfy-Org/ComfyUI/issues/16144

This is the issue... ran with `--disable-comfy-compiler` flag and the preview node seems to run without issues, using < 88% vram, as opposed to > 98% and stalling...


> **Comment by Djanghost (2026-09-08T10:25:56Z):**

If it helps resolve this issue, I was using the Model Sparse Attention node before it was merged, and I wasn't experiencing any problems. So this issue arose either after the last change to Model Sparse Attention right before it was merged, or right after the merge—but it's impossible to know for sure because we can't view the different versions of the PR.


> **Comment by kijai (2026-09-08T10:31:29Z):**

> If it helps resolve this issue, I was using the Model Sparse Attention node before it was merged, and I wasn't experiencing any problems. So this issue arose either after the last change to Model Sparse Attention right before it was merged, or right after the merge—but it's impossible to know for sure because we can't view the different versions of the PR.

I'm getting confused, there's like 3 different issues in this thread. The low VRAM attention does not have any interaction with the sparse attention in current code.


> **Comment by Castelmatto (2026-09-08T10:38:29Z):**

If i can make things more clear i believe 

> I’ve tested the new comfy core sparse attention node Kijai have just implemented (tested at default values using Sol-attn) and when active it causes a noticeable progressive increase in memory allocation as if leaking memory. Is it something to be expected? I have not tested yet if [Comfy-Org/ComfyUI@a99d1f9](https://github.com/Comfy-Org/ComfyUI/commit/a99d1f9c14f4594fe1e9d7e17c1997bf104d89fb) fixes this problem

this problem has been fixed by the PR mentioned.

> [@kijai](https://github.com/kijai) [Comfy-Org/ComfyUI#16144](https://github.com/Comfy-Org/ComfyUI/issues/16144)
> 
> This is the issue... ran with `--disable-comfy-compiler` flag and the preview node seems to run without issues, using < 88% vram, as opposed to > 98% and stalling...

This is the same problem I was having and that it was wrongly correlated to the preview node. It has now been fixed.

> > If it helps resolve this issue, I was using the Model Sparse Attention node before it was merged, and I wasn't experiencing any problems. So this issue arose either after the last change to Model Sparse Attention right before it was merged, or right after the merge—but it's impossible to know for sure because we can't view the different versions of the PR.
> 
> I'm getting confused, there's like 3 different issues in this thread. The low VRAM attention does not have any interaction with the sparse attention in current code.

The low-vram error has been fixed by your latest commit, and indeed is not really necessary when using the new attention node.


> **Comment by Djanghost (2026-09-08T10:38:35Z):**

> > If it helps resolve this issue, I was using the Model Sparse Attention node before it was merged, and I wasn't experiencing any problems. So this issue arose either after the last change to Model Sparse Attention right before it was merged, or right after the merge—but it's impossible to know for sure because we can't view the different versions of the PR.
> 
> I'm getting confused, there's like 3 different issues in this thread. The low VRAM attention does not have any interaction with the sparse attention in current code.

yea that's pretty weird.. I spent several hours trying different things with Claude's help, but the only way to get back to a “normal” state for now is to use this argument to disable the compiler. For my part, what I’m seeing on this node is a big memory leak when, for example, I change the node’s `end_percent` by going back to dense blocks before rendering is complete. That’s when the OOM occurs.


> **Comment by SveSop (2026-09-08T11:28:53Z):**

The problem i think, is that the recent changes to comfy-aimdo has made "everything" use more vram. Even with `--disable-comfy-compiler` i am using more vram than with comfy-aimdo==0.4.15. So, i suppose this "compiler" uses enough overhead to make "everything else" spill over into OOM territory. Disabling the compiler helps a bit, but its not the root cause, nor is the preview node.

Using `--disable-comfy-compiler ` makes things run again, but its still on shaky grounds, because using a ref. workload with 3-4 images + video still tends to grind to a halt for me, where it did not before.

Just my theory - Something (comfy-aimdo?) makes generation generally use more vram -> All other subnodes = higher risk of oom.

So yes @kijai This thread has (by my own admission) snowballed into a multiple problems thread. I am sorry for that.

1. The fix to MiniMax H3 Low VRAM Attention is fixed in your recent PR.
2. Other nodes that makes stuff go OOM (preview node), can be alleviated using `--disable-comfy-compiler`, but is not by itself at fault.

So, for this issue in itself (Low VRAM Attention), the issue is closed as fixed (no longer crashing).


> **Comment by kijai (2026-09-08T11:30:26Z):**

> The problem i think, is that the recent changes to comfy-aimdo has made "everything" use more vram. Even with `--disable-comfy-compiler` i am using more vram than with comfy-aimdo==0.4.15. So, i suppose this "compiler" uses enough overhead to make "everything else" spill over into OOM territory. Disabling the compiler helps a bit, but its not the root cause, nor is the preview node.
> 
> Using `--disable-comfy-compiler ` makes things run again, but its still on shaky grounds, because using a ref. workload with 3-4 images + video still tends to grind to a halt for me, where it did not before.
> 
> Just my theory - Something (comfy-aimdo?) makes generation generally use more vram -> All other subnodes = higher risk of oom.
> 
> So yes @kijai This thread has (by my own admission) snowballed into a multiple problems thread. I am sorry for that.
> 
> 1. The fix to MiniMax H3 Low VRAM Attention is fixed in your recent PR.
> 2. Other nodes that makes stuff go OOM (preview node), can be alleviated using `--disable-comfy-compiler`, but is not by itself at fault.
> 
> So, for this issue in itself (Low VRAM Attention), the issue is closed as fixed (no longer crashing).

Have you tried with --vram-headroom? 


> **Comment by Djanghost (2026-09-08T11:47:52Z):**

> > The problem i think, is that the recent changes to comfy-aimdo has made "everything" use more vram. Even with `--disable-comfy-compiler` i am using more vram than with comfy-aimdo==0.4.15. So, i suppose this "compiler" uses enough overhead to make "everything else" spill over into OOM territory. Disabling the compiler helps a bit, but its not the root cause, nor is the preview node.
> > Using `--disable-comfy-compiler ` makes things run again, but its still on shaky grounds, because using a ref. workload with 3-4 images + video still tends to grind to a halt for me, where it did not before.
> > Just my theory - Something (comfy-aimdo?) makes generation generally use more vram -> All other subnodes = higher risk of oom.
> > So yes [@kijai](https://github.com/kijai) This thread has (by my own admission) snowballed into a multiple problems thread. I am sorry for that.
> > 
> > 1. The fix to MiniMax H3 Low VRAM Attention is fixed in your recent PR.
> > 2. Other nodes that makes stuff go OOM (preview node), can be alleviated using `--disable-comfy-compiler`, but is not by itself at fault.
> > 
> > So, for this issue in itself (Low VRAM Attention), the issue is closed as fixed (no longer crashing).
> 
> Have you tried with --vram-headroom?

same issue with --vram-headroom, this is the first thing i've tested yesterday


> **Comment by SveSop (2026-09-08T12:08:59Z):**

I also tested `--vram-headroom` with 1, 2 and 3 without any change. Maybe comfy-compiler "lives outside of aimdo vram budget" somehow? 😏 


> **Comment by kijai (2026-09-08T18:00:10Z):**

Couple of possible fixes to the preview issue are now in, needs ComfyUI core to be up to date too.


> **Comment by Djanghost (2026-09-08T18:48:12Z):**

> Couple of possible fixes to the preview issue are now in, needs ComfyUI core to be up to date too.

Unfortunatly this doesn't fix the vram issue, --disable-comfy-compiler seems to be the best option for now


> **Comment by SveSop (2026-09-08T18:49:39Z):**

Still seems to bork for me with the `Model Preview Override` node enabled when using comfy-compiler 😢 

Running lower resolution, or shorter video does not grind to a halt, so i guess with the new comfy-aimdo using a lot more vram than before it is causing the model to be a lot more fragile than it used to be in terms of resolution/length on such ungodly low VRAM cards as 16GB.


> **Comment by kijai (2026-09-08T22:42:00Z):**

Well I can't replicate any of the issues, the preview node at least has next to no impact on VRAM use.


> **Comment by SveSop (2026-09-08T23:29:47Z):**

I don't know what to say. Rendering 1344x768 10s now with preview node using 80'ish % vram... with `--disable-comfy-compiler`
The comfy-compiler for me atleast seems to have a huge impact on vram usage for me, and when using it, it is just enough to get me over the edge for workflows that works perfectly fine without it.

I guess i can come back to this comfy-compiler from time to time when there is an update and test if it does anything useful. I dunno. Old 40xx series (SM89) card is probably not hugly tested with the new compiler thingy perhaps?


> **Comment by Rkkss (2026-09-09T09:24:35Z):**

My bug report was closed. 
Should I make new thread of should I just report it here?

https://github.com/Comfy-Org/ComfyUI/issues/16175#issuecomment-5594380385

It's not fixed Btw. Previously, it oom'ed at step 4/6 now it's oom at step 5/6. I run the same seed  fine without the model sparse attention node



> **Comment by Djanghost (2026-09-09T09:33:06Z):**

> My bug report was closed. Should I make new thread of should I just report it here?
> 
> [Comfy-Org/ComfyUI#16175 (comment)](https://github.com/Comfy-Org/ComfyUI/issues/16175#issuecomment-5594380385)
> 
> It's not fixed Btw. Previously, it oom'ed at step 4/6 now it's oom at step 5/6. I run the same seed fine without the model sparse attention node

Same here... i think you can start a new thread, but do it in the Comfy repo instead of the Kijai repo, since the error is coming from the Comfy compiler. This thread here should be closed as it seems the H3 Low VRAM attention breakage was fixed.


> **Comment by SveSop (2026-09-09T12:07:50Z):**

> Well I can't replicate any of the issues, the preview node at least has next to no impact on VRAM use.

https://github.com/kijai/ComfyUI-KJNodes/commit/b3ec064dde7d122b333660918e1200e928e67ff1

Seems to have fixed the "next to no impact on VRAM use" for me. Was able to run the workflow with comfy-compiler AND model preview without oom after this fix. Probably needs more testing, but first glance, it made an impact for me atleast.

```
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [09:06<00:00, 68.31s/it]
[INFO] Comfy model compiler graph breaks: 3, rogues: 0
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 2677MB Staged. 0 patches attached. Force pre-loaded 128 weights: 692 KB.
[INFO] Prompt executed in 00:10:14
```



> **Comment by Djanghost (2026-09-09T18:03:15Z):**

> > Well I can't replicate any of the issues, the preview node at least has next to no impact on VRAM use.
> 
> [b3ec064](https://github.com/kijai/ComfyUI-KJNodes/commit/b3ec064dde7d122b333660918e1200e928e67ff1)
> 
> Seems to have fixed the "next to no impact on VRAM use" for me. Was able to run the workflow with comfy-compiler AND model preview without oom after this fix. Probably needs more testing, but first glance, it made an impact for me atleast.
> 
> ```
> 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [09:06<00:00, 68.31s/it]
> [INFO] Comfy model compiler graph breaks: 3, rogues: 0
> [INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
> [INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 2677MB Staged. 0 patches attached. Force pre-loaded 128 weights: 692 KB.
> [INFO] Prompt executed in 00:10:14
> ```

I confirm 🙌 no more issue with long renderings, still using more vram than before the new compiler but much better than yesterday


> **Comment by kijai (2026-09-09T19:40:04Z):**

Using more VRAM is not a bad thing if it's not slower, it's possibly just offloading more accurately?


> **Comment by Djanghost (2026-09-09T20:28:18Z):**

I'll admit I haven't measured it precisely yet, but I don't notice any significant improvement in rendering speed for now (and it's not slower than before). All's working, that's what matters most : )
Thanks a lot for the Model Sparse Attention node btw.


> **Comment by SveSop (2026-09-10T01:17:21Z):**

> Using more VRAM is not a bad thing if it's not slower, it's possibly just offloading more accurately?

Absolutely. It's no problem using 98% vram, as long as you dont risk nodes doing something unexpected, as seems to be the case with the nvenc usage in the preview node. Seemingly the driver allocated something that aimdo/comfy-compiler was unaware of, and instead of (probably what comfy-compiler is supposed to do) re-arrange some vram usage, it spilled over into shared mem, and bob's your new aunt.

Since the problem seems fixed for now, i close this issue. Thanks for fixing @kijai 👍 


> **Comment by kijai (2026-09-10T05:39:08Z):**

> > Using more VRAM is not a bad thing if it's not slower, it's possibly just offloading more accurately?
> 
> Absolutely. It's no problem using 98% vram, as long as you dont risk nodes doing something unexpected, as seems to be the case with the nvenc usage in the preview node. Seemingly the driver allocated something that aimdo/comfy-compiler was unaware of, and instead of (probably what comfy-compiler is supposed to do) re-arrange some vram usage, it spilled over into shared mem, and bob's your new aunt.
> 
> Since the problem seems fixed for now, i close this issue. Thanks for fixing @kijai 👍 

Yeah nvenc was wrong choice otherwise too, CPU h264 ended up being faster which is also something I changed in the last update to the node.



### #766 [open] Model Preview Override node isn't working
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/766  by mery-96-new 2026-10-02T15:12:49Z  comments: 14

Hey, today I updated portable comfy from  "update_comfyui.bat" and suddenly my Model Preview Override stopped showing any preview and only showing noise, It was working normally before the update, now I don't know if update did something to it or not, can you please confirm if it's happening with anyone else as well? or is there a way to debug it or show the report on cmd for this node? Like if its running or not crashing because of some of my other nodes? Tho it was working before updating so I don't think my custom node is doing anything to it. 

<img width="406" height="791" alt="Image" src="https://github.com/user-attachments/assets/7766561f-ee11-4b1d-87d4-491096e1d4fc" />

> **Comment by kijai (2026-10-02T21:57:12Z):**

It's working fine for me with everything up to date.


> **Comment by mery-96-new (2026-10-03T00:31:39Z):**

Thanks for updating me about it, I'll just download fresh portable copy and move my stuff there to check. 


> **Comment by MrSeri0us (2026-10-03T14:42:00Z):**

Hi, exact same issue for me on ComfyUI 0.38.0. This is happening with Minimax H3.


> **Comment by mery-96-new (2026-10-03T15:21:13Z):**

So, I downloaded fresh copy of comfy portable and did update again using "update_comfyui.bat" and it's not working again. but I noticed something it's only happening when we apply the **"Preview_frames"**  option in the node, if we keep it 1 then it shows the very first frame which is the reference picture without an issue, but the moment we set the **"preview_frames"** to show video for minimax h3 the problem is there.

**With preview frames on:**  

<img width="854" height="682" alt="Image" src="https://github.com/user-attachments/assets/a1ca16a9-9ddd-46d3-bdb1-ca041b449014" />

**Without preview frames applied:**

<img width="820" height="750" alt="Image" src="https://github.com/user-attachments/assets/5e536ace-2576-4ca7-a234-f2a2d5a15a42" />


**This is my comfyui about:** 

<img width="914" height="183" alt="Image" src="https://github.com/user-attachments/assets/e1e685e8-e21a-478c-94a4-479a1828e402" />


**I have no other custom nodes either:** 

<img width="636" height="271" alt="Image" src="https://github.com/user-attachments/assets/4dca89d1-e087-47fa-8080-52fae5f63f2d" />

**_You already have so much on table and I don't wanna stress you about it so I'll just revert back to old version for now but I hope it gets fixed because this node is such a life saver, thank you <3_**


> **Comment by mery-96-new (2026-10-03T15:47:47Z):**

> Hi, exact same issue for me on ComfyUI 0.38.0. This is happening with Minimax H3.

Hey @MrSeri0us  please do this:
Step 1: Delete **ComfyUI-KJNodes** custom node pack from the Custom nodes folder
Step 2: Download it again from this repo and install the **requirements.txt** file again
Step 3: Test and please tell me if it worked for you

I just deleted **ComfyUI-KJNodes** folder and downloaded it again and now it's working, on updated comfy for me its working now.
Please test.

LMAO its working now on any option I select in tiny_vae even when its None or any other its working XD

<img width="452" height="799" alt="Image" src="https://github.com/user-attachments/assets/4cb3c639-8525-4319-af91-82966a04aa4d" />


> **Comment by kijai (2026-10-03T16:00:37Z):**

I set preview frames manually when I tested this last night, and it worked fine with Minimax, with or without the tiny vae. So I don't currently know how to replicate the issue.


> **Comment by SveSop (2026-10-03T16:20:49Z):**

I just updated comfyui to the latest commit now 30 minutes ago, and i have the same issue. Workflow is the same. I have set preview frames to the same number of frames that the video generates. Running the same workflow with NO changes 2 hours ago, it worked just fine, so something has changed in the last comfy commits since yesterday.

I have not tried to download it again tho, but i suspect it is not my git (nightly) folder that is "broken", but it could be one of the requirements that needs update to a newer version perhaps?


> **Comment by kijai (2026-10-03T16:33:11Z):**

Any of you using the commandline argument to disable API nodes? Could be related to that as how it works did change recently.


> **Comment by SveSop (2026-10-03T16:39:49Z):**

I do not disable any API nodes. However, setting the `tiny_vae` setting to none shows animation. Poor quality as expected tho.
I have not paid too much attention to this node tho, but i notice that all the other tiny vae models have a _encode and _decode prefix, my taeh3.safetensor does not? I dunno whats up with that?




> **Comment by mery-96-new (2026-10-03T16:49:11Z):**

These are my parameteres when its not working:
```
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-auto-launch --disable-fast-disk --offline
```

When working :
```
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-auto-launch --disable-fast-disk
```

For me **--offline** is the culprit. Removing it fixed it for me, maybe for others as well disabling API nodes too causing it? Please confirm guys.


> **Comment by SveSop (2026-10-03T16:50:29Z):**

`.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --max-upload-size 2048 --disable-auto-launch`

More or less as plain as it can be...


> **Comment by SveSop (2026-10-03T17:08:47Z):**

Eh? Okay.. So, ran some generations earlier today - all ok. Then i updated comfyui -> Preview node "dead" -> then i come here -> select "none" as a test -> preview node low-res preview.

But just for lulz, i selected the taeh3.safetensor again... lo and behold, it  works now! So? Some cache/model shit? I mean, i have restarted comfy a couple of times in the meantime to no effect. But, seems to be working okay again now, without doing much more than selecting none -> run generation -> select taeh3 -> run generation.




> **Comment by kijai (2026-10-03T17:28:56Z):**

> These are my parameteres when its not working:
> 
> ```
> .\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-auto-launch --disable-fast-disk --offline
> ```
> 
> When working :
> 
> ```
> .\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-auto-launch --disable-fast-disk
> ```
> 
> For me **--offline** is the culprit. Removing it fixed it for me, maybe for others as well disabling API nodes too causing it? Please confirm guys.

`--disable-partner-nodes` is probably what you want now, the offline and API nodes behaviour was separated in recent update.


> **Comment by MrSeri0us (2026-10-04T18:26:58Z):**

Removing the --offline argument resolved for me too



### #721 [open] [BUG] MiniMaxH3MemoryEfficientSageAttentionPatch fails on RTX 5060 Ti (Compute Capability 12.x) with SageAttention 1.0.6
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/721  by lwjuiqyweqwy 2026-08-06T13:12:30Z  comments: 5

RuntimeError: sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.
2026-08-06T21:02:48.014287 - [ERROR] !!! Exception during processing !!! sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.
...
RuntimeError: sageattention is not new enough version or could not determine CUDA architecture...

[H3_Turbo_Stable_4V4A（AV各4步）.json](https://github.com/user-attachments/files/30789670/H3_Turbo_Stable_4V4A.AV.4.json)

> **Comment by kijai (2026-08-06T13:25:59Z):**

As the error says, you need newer version of sageattention.


> **Comment by ahkimkoo (2026-08-07T03:43:02Z):**

same trouble, already sageattention 1.0.6


> **Comment by zzh383635387 (2026-08-08T09:38:44Z):**

same problem,I had sovled by gpt5.6
In ComfyUI_windows_portable folder
first install :   python_embeded\python.exe -m pip install -U "triton-windows<3.8"
then
python_embeded\python.exe -m pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl"
SageAttention will be install version 2.2.0
restart ComfyUI 



> **Comment by hackedbyxx (2026-08-13T06:40:21Z):**

> same problem,I had sovled by gpt5.6 In ComfyUI_windows_portable folder first install : python_embeded\python.exe -m pip install -U "triton-windows<3.8" then python_embeded\python.exe -m pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl" SageAttention will be install version 2.2.0 restart ComfyUI

it solve my problem, thanks


> **Comment by scarab72 (2026-08-17T07:20:23Z):**

Comfy is flooded with bugs every time it's updated; I only use it because it's free, but I wouldn't use it if it were paid.



### #707 [closed] Model Preview Override for MiniMax H3
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/707  by norazlan-dev 2026-08-03T12:42:51Z  comments: 4

Is there a  plan to make "Model Preview Override" node compatible with MiniMax H3?

> **Comment by kijai (2026-08-03T13:01:41Z):**

No need, if you have https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite installed you can just enable these settings, then refresh the page and next generation should have preview:

<img width="945" height="652" alt="Image" src="https://github.com/user-attachments/assets/70625b1b-9d97-42e8-9746-6fc496cec8e4" />


> **Comment by norazlan-dev (2026-08-03T13:08:54Z):**

Aw I like having a separate node for preview. No problem, not an issue.


> **Comment by kijai (2026-08-04T09:01:31Z):**

Sorry I was thinking of wrong node, the ModelPreviewOverrideKJ should now support Minimax too.


> **Comment by norazlan-dev (2026-08-05T01:22:32Z):**

Oh thank you. Yeah the live preview displayed on the subgraph seem to be broken on the current frontend, plus your node is more flexible for the sampling preview. Thanks again.



### #712 [closed] MiniMax H3 Memory Efficient SageAttention patch outputs only gray noise on SM89
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/712  by magicwang1111 2026-08-04T08:17:58Z  comments: 4

### Summary

The new `MiniMaxH3MemoryEfficientSageAttentionPatch` produces a completely corrupted MiniMax H3 video on an RTX 4090 D (SM89). The workflow completes normally, but every frame is gray low-frequency noise / blurred blocks; no subject or scene is recognizable.

An unpatched comparison run using the same input image, prompt, model, sampler, scheduler, resolution, frame count, and step count produces a normal recognizable video. The comparison run used a different random seed, so this is not a bit-exact A/B, but the patched result is fully corrupted rather than a normal seed-dependent quality variation.

### Steps to reproduce

1. Use KJNodes commit `1289b52fbb6d64a339a4047b9ea74cf7758ccf1e` (`1.4.8`).
2. Load `minimax_h3_fl2va_pruned_int8_convrot.safetensors` with `UNETLoader`, `weight_dtype=default`.
3. Connect the model through `MiniMaxH3MemoryEfficientSageAttentionPatch`.
4. Use the patched model for both `BasicScheduler` and `BasicGuider`.
5. Generate MiniMax H3 I2V with:
   - Input/output size: `768 x 1376`
   - Frames: `362` at `24 fps` (15.08 s)
   - Sampler: `res_multistep`
   - Scheduler: `simple`
   - Steps: `20`
   - Denoise: `1.0`
6. Decode with `minimax_h3_video_vae_fp16.safetensors` and save as H.264 MP4.

### Actual result

- All 362 frames contain only gray noise / blurred blocks.
- No recognizable person, object, or background appears at any timestamp.
- The sampler completes all 20 steps and ComfyUI reports no exception.
- Patched generation time was about `22:21`.

Relevant log lines:

```text
Native ops: float8_e5m2, convrot_w4a4, float8_e4m3fn, int8_tensorwise
model weight dtype torch.bfloat16, manual cast: torch.bfloat16
Applying MiniMax H3 Memory Efficient Sage Attention Patch to all transformer blocks
100% ... 20/20 [20:52<00:00, 62.64s/it]
Prompt executed in 00:22:21
```

### Expected result

A recognizable video comparable to the unpatched MiniMax H3 output, with reduced peak VRAM usage but without complete image corruption.

### Comparison

The unpatched run used:

- The same input image
- The same prompt (matching SHA-256: `0a7caca1f8dced7b1459c395dae326be307c697726145497cd0a3286020663b1`)
- The same diffusion model and VAE
- The same sampler, scheduler, steps, denoise, resolution, frame count, and FPS
- A different random seed

The unpatched result contains a clear subject and background throughout. The patched result is corrupted from the first frame through the last frame.

### Environment

- OS: Windows
- GPU: NVIDIA GeForce RTX 4090 D, compute capability `8.9` / SM89
- NVIDIA driver: `591.86` (driver reports CUDA 13.1)
- Python: `3.11.15`
- ComfyUI commit: `16e3f3034f2bba1fff6c70cbd759339778555cd6`
- KJNodes: `1.4.8`, commit `1289b52fbb6d64a339a4047b9ea74cf7758ccf1e`
- PyTorch: `2.11.0+cu130`
- PyTorch CUDA runtime: `13.0`
- SageAttention: `2.2.0+cu130torch2.9.0andhigher.post4`
- comfy-kitchen: `0.2.26`
- triton-windows: `3.6.0.post26`

This appears to exercise the SM89 FP8 branch in `_sageattn_int8_fp8_nhd`. Could there be a layout, scaling, or output-value issue in the new MiniMax H3 wrapper on SM89?

I have the failing 15-second MP4, the normal comparison MP4, sampled-frame contact sheets, and the workflow JSON available if needed. I have not attached the raw workflow here because it contains a long private prompt and local paths.


> **Comment by magicwang1111 (2026-08-04T08:22:47Z):**

https://github.com/user-attachments/assets/1d2eb81b-f0cf-4d3c-96ee-3a7a8f73e9af
[20260804minimaxpatch.json](https://github.com/user-attachments/files/30694465/20260804minimaxpatch.json)


> **Comment by kijai (2026-08-04T08:31:36Z):**

It works on my 4090 so it's not sm89 problem, could be sageattn version issue, there's newer one out than post4.


> **Comment by magicwang1111 (2026-08-04T08:44:28Z):**

(comfyui) PS D:\ComfyUI> pip show sageattention
Name: sageattention
Version: 2.2.0+cu130torch2.10.0andhigher.post6
Summary: Accurate and efficient plug-and-play low-bit attention.
Home-page: https://github.com/thu-ml/SageAttention
Author: SageAttention team
Author-email:
License: Apache 2.0 License
Location: D:\miniconda3\envs\comfyui\Lib\site-packages
Requires:
Required-by:
(comfyui) PS D:\ComfyUI>

fix it,thx


> **Comment by magicwang1111 (2026-08-04T09:11:16Z):**

https://github.com/user-attachments/assets/61db4b47-129b-46d6-955e-6e296a2ca39e
https://github.com/user-attachments/assets/959592b5-1772-4f37-a1f6-a8725ff8fb35
2s generate normal,15s generate still problem



### #728 [closed] [Feature Request] Node for saving the combined latent of Minimax H3 Model
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/728  by bhusann 2026-08-07T14:37:26Z  comments: 3

Hey kijay, is it possible to create a node that saves the generated latent (video + audio) which minimax produces, so that I can decode it later with vae nodes?



> **Comment by bhusann (2026-08-07T16:10:17Z):**

<img width="438" height="309" alt="Image" src="https://github.com/user-attachments/assets/30bba523-bcfa-44d4-a2bc-53aae7409044" /> 

there is no separate audio and video latents from it, it has only single latent, i dont know the current nodes can do it, but how can i load it for decoding from load latent node after we have saved it with save latent node (if you create it in the future)




> **Comment by kijai (2026-08-07T16:13:35Z):**

The LTXV Separate and Concat AV Latent -nodes should work for the separation, maybe that can be saved already?


> **Comment by sdman88 (2026-08-09T18:53:46Z):**

> The LTXV Separate and Concat AV Latent -nodes should work for the separation, maybe that can be saved already?

Unfortunately the LTXV separate and concat AV latent nodes don't seem compatible with minimax H3  model when I tried them, it throws an error when trying to vae decode if IIRC. Definitely having a node that can split and concat the minimax latent would be helpful if it's even possible so latent upscale would be possible with minimax model directly. 



### #713 [closed] SageAttention post6: 362-frame MiniMax H3 output is gray noise while 56-frame output works
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/713  by magicwang1111 2026-08-04T09:31:48Z  comments: 3

### Summary

Follow-up to #712 after upgrading SageAttention from `post4` to `post6`.

SageAttention `2.2.0+cu130torch2.10.0andhigher.post6` fixes the short generation, but a 15.08-second / 362-frame MiniMax H3 generation still completes as gray noise for the entire video. A 2.33-second / 56-frame comparison produces normal recognizable imagery.

The post6 release specifically mentions fixing an out-of-bounds issue that caused black/noise outputs, so I expected both lengths to work.

### Important comparison caveat

Inspection of the workflow metadata embedded in both MP4 files found that duration is **not the only changed variable**:

- Failing 15.08 s run: `UNETLoader -> MiniMaxH3MemoryEfficientSageAttentionPatch`
- Working 2.33 s run: `OTUNetLoaderW8A8(enable_convrot=true, on_the_fly_quantization=false) -> MiniMaxH3MemoryEfficientSageAttentionPatch`

The two runs also use different random seeds. Therefore this report does not claim that sequence length alone is proven to cause the corruption. It may instead be an interaction between the patch, the standard `UNETLoader`, the int8 ConvRot checkpoint, and/or long sequence length.

### Shared settings

- Input image: identical
- Prompt: identical (SHA-256 `0a7caca1f8dced7b1459c395dae326be307c697726145497cd0a3286020663b1`)
- Model file: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Weight dtype: `default`
- Resolution: `768 x 1376`
- FPS: `24`
- Sampler: `res_multistep`
- Scheduler: `simple`
- Steps: `20`
- Denoise: `1.0`
- Both models pass through `MiniMaxH3MemoryEfficientSageAttentionPatch`

### Results

| Run | Frames / duration | Model loader path | Result |
|---|---:|---|---|
| Long | 362 / 15.08 s | `UNETLoader` | Gray low-frequency noise in every sampled frame; no recognizable subject or scene |
| Short | 56 / 2.33 s | `OTUNetLoaderW8A8`, ConvRot enabled | Normal recognizable person and background |

The two comparison videos are already uploaded here: [#712 comparison comment](https://github.com/kijai/ComfyUI-KJNodes/issues/712#issuecomment-5176938168).

### Long-run log

The failing long run completes all 20 sampling steps without an exception:

```text
Native ops: float8_e5m2, int8_tensorwise, convrot_w4a4, float8_e4m3fn
model weight dtype torch.bfloat16, manual cast: torch.bfloat16
Applying MiniMax H3 Memory Efficient Sage Attention Patch to all transformer blocks
100% ... 20/20 [20:56<00:00, 62.81s/it]
Prompt executed in 00:22:25
```

### Expected result

The 362-frame run should produce recognizable video rather than gray noise, or the node should document/validate unsupported loader and sequence combinations instead of silently completing with corrupted output.

### Environment

- Windows
- GPU: NVIDIA GeForce RTX 4090 D, SM89 / compute capability 8.9
- NVIDIA driver: 591.86
- PyTorch: `2.11.0+cu130`
- PyTorch CUDA runtime: `13.0`
- SageAttention: `2.2.0+cu130torch2.10.0andhigher.post6`
- ComfyUI: `16e3f3034f2bba1fff6c70cbd759339778555cd6`
- KJNodes: `1289b52fbb6d64a339a4047b9ea74cf7758ccf1e` / 1.4.8
- comfy-kitchen: `0.2.26`
- triton-windows: `3.6.0.post26`

Could you clarify whether `MiniMaxH3MemoryEfficientSageAttentionPatch` is expected to support the standard `UNETLoader` with the int8 ConvRot checkpoint for 362-frame sequences? If so, the post6 long output is still silently corrupted.


> **Comment by magicwang1111 (2026-08-04T09:33:07Z):**

https://github.com/user-attachments/assets/493ed6e2-4ac8-42f4-9e54-0144a3db37d5
https://github.com/user-attachments/assets/ee0ef4a6-56aa-4983-9bd4-31b6617651c7


> **Comment by magicwang1111 (2026-08-04T09:57:16Z):**

https://github.com/user-attachments/assets/55c78d58-04cf-410d-9825-1934308e88ad

10s also normal


> **Comment by magicwang1111 (2026-08-05T02:03:44Z):**

https://github.com/user-attachments/assets/72bced55-a732-48fa-bfa6-de69a852cfd3
14s video generation have same problme



### #718 [closed] Audio Live Preview during sampling is it possible?
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/718  by neural-dreamer 2026-08-05T21:18:12Z  comments: 3

I do not know if this is technically possible or not, but with the advent of the LTX 2 model and especially the MiniMax H3 model, it seems to me that the idea of Audio Live Preview may become relevant. Because on the MiniMax H3 model, generation takes a relatively long time and sometimes the final sound does not match the expectation. Therefore, having an Audio Live Preview during generation would be very useful.

Is it technically possible to implement this?

> **Comment by kijai (2026-08-05T21:34:06Z):**

Well not that easily, we'd have to use the full audio VAE on each step to decode it. It's possible but somewhat costly to do.


> **Comment by neural-dreamer (2026-08-05T21:46:00Z):**

> Well not that easily, we'd have to use the full audio VAE on each step to decode it. It's possible but somewhat costly to do.

So we can't make some kind of TAE Vae for audio or another way to output it at a lower cost? 
Sorry if the question is stupid.


> **Comment by SveSop (2026-08-06T22:23:50Z):**

Although interesting to have, i am not entirely sure how useful it would be? Unless ofc all the audio is generated first... Useful ofc if its spoken sentences does not match or something, but if it refines audio and video each step, i think "all" audio would sound horrible until sigma 0.0.

Then again, i do not really know much about this, other than i love the new preview node 👍 



### #725 [open] "[ERROR] !!! Exception during processing !!! cannot access local variable 'q_int8' where it is not associated with a value"with Adding MiniMax H3 Mem Eff Sage Attention Patch
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/725  by sovf4717 2026-08-07T11:00:29Z  comments: 2

### Output
D:\ComfyUI_all>cd /d D:\ComfyUI_all

D:\ComfyUI_all>call E:\miniconda3\Scripts\activate.bat E:\miniconda3

(base) D:\ComfyUI_all>call conda activate comfyui_conda

(comfyui_conda) D:\ComfyUI_all>set COMFYUI_ENABLE_MIOPEN=1

(comfyui_conda) D:\ComfyUI_all>set MIOPEN_FIND_MODE=2

(comfyui_conda) D:\ComfyUI_all>set PYTORCH_MIOPEN_SUGGEST_NHWC=1

(comfyui_conda) D:\ComfyUI_all>set NUMEXPR_MAX_THREADS=12

(comfyui_conda) D:\ComfyUI_all>set OMP_NUM_THREADS=6

(comfyui_conda) D:\ComfyUI_all>Set XFORMERS_MORE_DETAILS=1

(comfyui_conda) D:\ComfyUI_all>set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

(comfyui_conda) D:\ComfyUI_all>python -s "ComfyUI\main.py" --windows-standalone-build --disable-dynamic-vram
[INFO] setup plugin alembic.autogenerate.schemas
[INFO] setup plugin alembic.autogenerate.tables
[INFO] setup plugin alembic.autogenerate.types
[INFO] setup plugin alembic.autogenerate.constraints
[INFO] setup plugin alembic.autogenerate.defaults
[INFO] setup plugin alembic.autogenerate.comments
[INFO] setup plugin alembic.autogenerate.checkconstraint_byname
[START] Security scan
[DONE] Security scan
## ComfyUI-Manager: installing dependencies done.
** ComfyUI startup time: 2026-08-07 18:37:12.204
** Platform: Windows
** Python version: 3.12.13 | packaged by Anaconda, Inc. | (main, Jul  9 2026, 14:26:47) [MSC v.1942 64 bit (AMD64)]
** Python executable: E:\miniconda3\envs\comfyui_conda\python.exe
** ComfyUI Path: D:\ComfyUI_all\ComfyUI
** ComfyUI Base Folder Path: D:\ComfyUI_all\ComfyUI
** User directory: D:\ComfyUI_all\ComfyUI\user
** ComfyUI-Manager config path: D:\ComfyUI_all\ComfyUI\user\__manager\config.ini
** Log path: D:\ComfyUI_all\ComfyUI\user\comfyui.log
[INFO]
Prestartup times for custom nodes:
[INFO]    0.0 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\rgthree-comfy
[INFO]    5.9 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-Manager
[INFO]
[INFO] Found comfy_kitchen backend cuda: {'available': False, 'disabled': True, 'unavailable_reason': 'Extension file not found: E:\\miniconda3\\envs\\comfyui_conda\\Lib\\site-packages\\comfy_kitchen\\backends\\cuda\\_C.abi3.pyd', 'capabilities': []}
[INFO] Found comfy_kitchen backend triton: {'available': True, 'disabled': True, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'int8_linear', 'quantize_and_rotate_rowwise', 'quantize_int8_rowwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'w4a8_int8_linear']}
[INFO] Found comfy_kitchen backend eager: {'available': True, 'disabled': False, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'convrot_w4a4_linear', 'dequantize_convrot_w4a4_weight', 'dequantize_int8_convrot_weight', 'dequantize_int8_convrot_weight_dtype', 'dequantize_int8_embedding', 'dequantize_int8_simple', 'dequantize_int8_simple_dtype', 'dequantize_mxfp8', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'dequantize_w4a8_int8_weight', 'gemv_awq_w4a16', 'int8_linear', 'prepare_int4_weight_for_int8_linear', 'quantize_and_rotate_rowwise', 'quantize_convrot_w4a4_weight', 'quantize_int8_convrot_weight', 'quantize_int8_rowwise', 'quantize_int8_tensorwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'quantize_svdquant_w4a4', 'quantize_w4a8_int8_weight', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'rotate_int8_convrot_weight', 'scaled_mm_mxfp8', 'scaled_mm_nvfp4', 'scaled_mm_svdquant_w4a4', 'stochastic_rounding_fp8', 'w4a8_int8_linear']}
[INFO] Found comfy_kitchen backend hip: {'available': True, 'disabled': False, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'convrot_w4a4_linear', 'dequantize_convrot_w4a4_weight', 'dequantize_int8_convrot_weight_dtype', 'dequantize_int8_simple_dtype', 'dequantize_per_tensor_fp8', 'dequantize_w4a8_int8_weight', 'gemv_awq_w4a16', 'int8_linear', 'quantize_and_rotate_rowwise', 'quantize_convrot_w4a4_weight', 'quantize_int8_convrot_weight', 'quantize_int8_rowwise', 'quantize_int8_tensorwise', 'quantize_per_tensor_fp8', 'quantize_svdquant_w4a4', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'scaled_mm_svdquant_w4a4', 'stochastic_rounding_fp8', 'w4a8_int8_linear']}
[INFO] Checkpoint files will always be loaded safely.
[INFO] Total VRAM 16304 MB, total RAM 31862 MB
[INFO] pytorch version: 2.12.0+rocm7.14.0
[INFO] AMD arch: gfx1200
[INFO] ROCm version: (7, 14)
[INFO] Set vram state to: NORMAL_VRAM
[INFO] Device: cuda:0 AMD Radeon RX 9060 XT : native
[INFO] Using async weight offloading with 2 streams
[INFO] Enabled pinned memory 12744.0
[aiter] Triton ops only: CK and HIP ops (and their JIT build) are skipped.
[INFO] Using sub quadratic optimization for attention, if you have memory or speed issues try using: --use-split-cross-attention
[INFO] Python version: 3.12.13 | packaged by Anaconda, Inc. | (main, Jul  9 2026, 14:26:47) [MSC v.1942 64 bit (AMD64)]
[INFO] ComfyUI version: 0.30.0
[INFO] comfy-aimdo version: 0.0.0
[INFO] comfy-kitchen version: 0.2.27
[WARNING] Dynamic vram disabled with argument. If you have any issues with dynamic vram enabled please give us a detailed reports as this argument will be removed soon. If you use gguf we recommend keeping dynamic vram enabled and using native ComfyUI model formats instead. ComfyUI native formats like fp8 will be faster even if they are larger than your memory.
[INFO] [Prompt Server] web root: E:\miniconda3\envs\comfyui_conda\Lib\site-packages\comfyui_frontend_package\static
[INFO] Asset seeder disabled
[INFO] No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
[INFO] ### Loading: ComfyUI-Manager (V3.41)
[INFO] [ComfyUI-Manager] network_mode: public
[INFO] [ComfyUI-Manager] ComfyUI per-queue preview override detected (PR #11261). Manager's preview method feature is disabled. Use ComfyUI's --preview-method CLI option or 'Settings > Execution > Live preview method'.
[INFO] ### ComfyUI Version: v0.30.0-22-g2340099d | Released on '2026-08-07'

[rgthree-comfy] Loaded 48 exciting nodes. 🎉

[rgthree-comfy] ComfyUI's new Node 2.0 rendering may be incompatible with some rgthree-comfy nodes and features, breaking some rendering as well as losing the ability to access a node's properties (a vital part of many nodes). It also appears to run MUCH more slowly spiking CPU usage and causing jankiness and unresponsiveness, especially with large workflows. Personally I am not planning to use the new Nodes 2.0 and, unfortunately, am not able to invest the time to investigate and overhaul rgthree-comfy where needed. If you have issues when Nodes 2.0 is enabled, I'd urge you to switch it off as well and join me in hoping ComfyUI is not planning to deprecate the existing, stable canvas rendering all together.

[INFO]
Import times for custom nodes:
[INFO]    0.0 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\websocket_image_save.py
[INFO]    0.0 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\Anomalous_Model_Browser
[INFO]    0.0 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\comfyui-inpaint-cropandstitch
[INFO]    0.0 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-Spectrum-MiniMax-H3
[INFO]    0.0 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-Director
[INFO]    0.0 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\rgthree-comfy
[INFO]    0.1 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-SolAttn_triton
[INFO]    0.1 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-KJNodes
[INFO]    0.9 seconds: D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-Manager
[INFO]
[INFO] Context impl SQLiteImpl.
[INFO] Will assume non-transactional DDL.
[INFO] Using RAM pressure cache.
[INFO] Starting server

[INFO] To see the GUI go to: http://127.0.0.1:8188
[INFO] [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json
[INFO] [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/alter-list.json
[INFO] [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/github-stats.json
[INFO] [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json
[WARNING] [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
[WARNING] [DEPRECATION WARNING] Detected import of deprecated legacy API: /extensions/core/groupNode.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
[WARNING] [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui/components/buttonGroup.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
[WARNING] [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui/components/button.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
[INFO] [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json
[INFO] got prompt
[INFO] VAE load device: cuda:0, offload device: cpu, dtype: torch.float32
[INFO] [MiniMaxEnhance] storyboard via custom (http://localhost:8084/v1, model ''), 2 image(s)...
[WARNING] [MiniMaxEnhance] No model name set for custom. Enter the name of the model you have loaded there. — passing your text through unchanged.
[INFO] [MiniMaxEnhance] 330 chars:
Cinematic desert chase, late afternoon golden hour, anamorphic lens, shallow depth of field, fine film grain, warm sand tones against a deep blue sky, handheld energy.
Audio: a low engine rumble that rises and falls with the throttle, wind across the microphone, sand hissing under the tyres, a sparse percussive score underneath.
[INFO] Found quantization metadata version 1
[INFO] VAE load device: cuda:0, offload device: cpu, dtype: torch.float16
[INFO] Found quantization metadata version 1
[INFO] Using MixedPrecisionOps for text encoder
[INFO] CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
[INFO] Found quantization metadata version 1
[INFO] Detected mixed precision quantization
[INFO] Using mixed precision operations
[INFO] Native ops: convrot_w4a4, int8_tensorwise , emulated ops: float8_e5m2, nvfp4, mxfp8, float8_e4m3fn
[INFO] model weight dtype torch.bfloat16, manual cast: torch.bfloat16
[INFO] model_type FLOW_AV
[INFO] Using sage attention mode: auto
[INFO] Applying MiniMax H3 Memory Efficient Sage Attention Patch to all transformer blocks
[INFO] [MiniMaxDirector] t2va | 768x416 | 124 frames (5.17s @24fps) | 3 shots | refs: 0 img / 0 vid / 0 audio
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] loaded partially; 14639.80 MB usable, 13763.55 MB loaded, 12120.28 MB offloaded, 875.22 MB buffer reserved, lowvram patches: 0
[INFO] Requested to load MiniMaxH3
[INFO] 0 models unloaded.
[INFO] loaded partially; 6259.64 MB usable, 5443.05 MB loaded, 14553.09 MB offloaded, 918.83 MB buffer reserved, lowvram patches: 0
  0%|                                                                                            | 0/4 [00:01<?, ?it/s]
[ERROR] !!! Exception during processing !!! cannot access local variable 'q_int8' where it is not associated with a value
[ERROR] Traceback (most recent call last):
  File "D:\ComfyUI_all\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\ComfyUI_all\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-KJNodes\nodes\preview_override_node.py", line 685, in __call__
    return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\miniconda3\envs\comfyui_conda\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\k_diffusion\sampling.py", line 1460, in sample_res_multistep
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0., cfg_pp=False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\miniconda3\envs\comfyui_conda\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\k_diffusion\sampling.py", line 1418, in res_multistep
    denoised = model(x, sigmas[i] * s_in, **extra_args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\model_base.py", line 203, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\model_base.py", line 247, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\miniconda3\envs\comfyui_conda\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\miniconda3\envs\comfyui_conda\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\ldm\minimax\model.py", line 501, in forward
    out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\ldm\minimax\model.py", line 637, in _forward
    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\miniconda3\envs\comfyui_conda\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\miniconda3\envs\comfyui_conda\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy\ldm\minimax\model.py", line 260, in forward
    x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\miniconda3\envs\comfyui_conda\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\miniconda3\envs\comfyui_conda\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-KJNodes\nodes\ltxv_nodes.py", line 2096, in minimax_sageattn_forward
    o = _sageattn_int8_fp8_nhd(qkv, dtype)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-KJNodes\nodes\ltxv_nodes.py", line 1946, in _sageattn_int8_fp8_nhd
    del q_int8, q_scale, k_int8, k_scale
        ^^^^^^
UnboundLocalError: cannot access local variable 'q_int8' where it is not associated with a value

[INFO] Prompt executed in 150.23 seconds
[INFO] got prompt
[INFO] Using split attention in VAE
[INFO] Using split attention in VAE
[INFO] VAE load device: cuda:0, offload device: cpu, dtype: torch.bfloat16
[INFO] Requested to load ZImageTEModel_
[INFO] loaded completely;  3836.12 MB loaded, full load: True
[INFO] CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cuda:0, dtype: torch.float16
[INFO] Found quantization metadata version 1
[INFO] Detected mixed precision quantization
[INFO] Using mixed precision operations
[INFO] Native ops: convrot_w4a4, int8_tensorwise , emulated ops: float8_e5m2, nvfp4, mxfp8, float8_e4m3fn
[INFO] model weight dtype torch.float8_e4m3fn, manual cast: torch.bfloat16
[INFO] model_type FLOW
[INFO] Using sage attention mode: auto
[ERROR] !!! Exception during processing !!! MiniMax H3 Memory Efficient Sage Attention Patch can only be applied to a MiniMax H3 model.
[ERROR] Traceback (most recent call last):
  File "D:\ComfyUI_all\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\ComfyUI_all\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI_all\ComfyUI\custom_nodes\ComfyUI-KJNodes\nodes\ltxv_nodes.py", line 2179, in execute
    raise RuntimeError("MiniMax H3 Memory Efficient Sage Attention Patch can only be applied to a MiniMax H3 model.")
RuntimeError: MiniMax H3 Memory Efficient Sage Attention Patch can only be applied to a MiniMax H3 model.

[INFO] Prompt executed in 9.56 seconds

### Site-packges
(comfyui_conda) D:\ComfyUI_all>pip list
Package                                    Version
------------------------------------------ -----------------
aiofiles                                   25.1.0
aiohappyeyeballs                           2.7.1
aiohttp                                    3.14.3
aiohttp_socks                              0.11.0
aiosignal                                  1.4.0
alembic                                    1.19.0
amd-aiter                                  0.0.0
amd-torch-device-gfx1200                   2.12.0+rocm7.14.0
amd-torchvision-device-gfx1200             0.27.0+rocm7.14.0
annotated-doc                              0.0.5
annotated-types                            0.8.0
anyio                                      4.14.2
attrs                                      26.1.0
av                                         18.0.0
blake3                                     1.0.9
certifi                                    2026.7.22
cffi                                       2.1.1
chardet                                    7.4.3
charset-normalizer                         3.4.9
click                                      8.4.2
color-matcher                              0.6.0
colorama                                   0.4.6
comfy-aimdo                                0.0.0
comfy-angle                                0.1.0
comfy-kitchen                              0.2.27
comfyui-embedded-docs                      0.5.9
comfyui_frontend_package                   1.48.6
comfyui_workflow_templates                 0.11.31
comfyui-workflow-templates-core            0.3.295
comfyui-workflow-templates-json            0.1.30
comfyui-workflow-templates-media-api       0.3.84
comfyui-workflow-templates-media-assets-01 0.1.19
comfyui-workflow-templates-media-image     0.3.160
comfyui-workflow-templates-media-other     0.3.229
comfyui-workflow-templates-media-video     0.3.101
contourpy                                  1.3.3
cryptography                               50.0.0
cycler                                     0.12.1
ddt                                        1.7.2
docutils                                   0.23
einops                                     0.8.2
filelock                                   3.32.2
flash_attn                                 2.8.4
fonttools                                  4.63.0
frozenlist                                 1.8.0
fsspec                                     2026.7.0
gitdb                                      4.0.12
GitPython                                  3.1.58
greenlet                                   3.5.4
h11                                        0.16.0
h2                                         4.4.1
hf-xet                                     1.6.0
hpack                                      4.2.0
httpcore                                   1.0.9
httpx                                      0.28.1
huggingface_hub                            1.26.0
hyperframe                                 6.1.0
idna                                       3.18
ImageIO                                    2.37.4
Jinja2                                     3.1.6
jsonschema                                 4.26.0
jsonschema-specifications                  2025.9.1
kiwisolver                                 1.5.0
kornia                                     0.8.3
kornia_rs                                  0.1.14
Mako                                       1.4.1
markdown-it-py                             4.2.0
MarkupSafe                                 3.0.3
matplotlib                                 3.11.1
matrix-nio                                 0.26.0
mdurl                                      0.1.2
mpmath                                     1.3.0
mss                                        10.2.0
multidict                                  6.7.1
nanobind                                   2.14.0
networkx                                   3.6.1
numpy                                      2.5.1
opencv-python-headless                     5.0.0.93
packaging                                  26.3
pillow                                     12.3.0
pip                                        26.1.2
propcache                                  0.5.2
psutil                                     7.2.2
pycparser                                  3.0
pycryptodome                               3.23.0
pydantic                                   2.13.4
pydantic_core                              2.46.4
pydantic-settings                          2.14.2
PyGithub                                   2.9.1
Pygments                                   2.20.0
PyJWT                                      2.13.0
PyNaCl                                     1.6.2
PyOpenGL                                   3.1.10
pyparsing                                  3.3.2
python-dateutil                            2.9.0.post0
python-dotenv                              1.2.2
python-socks                               2.8.2
PyYAML                                     6.0.3
referencing                                0.37.0
regex                                      2026.7.19
requests                                   2.34.2
rich                                       15.0.0
rocm                                       7.14.0
rocm-bootstrap                             0.1.0
rocm-sdk-core                              7.14.0
rocm-sdk-devel                             7.14.0
rocm-sdk-device-gfx1200                    7.14.0
rocm-sdk-libraries                         7.14.0
rocm-sdk-libraries-custom                  7.2.1
rocm-sdk-libraries-gfx120X-all             7.13.0
rpds-py                                    2026.6.3
safetensors                                0.8.0
sageattention                              2.2.0
scipy                                      1.18.0
sentencepiece                              0.2.2
setuptools                                 82.0.0
shellingham                                1.5.4
simpleeval                                 1.0.7
six                                        1.17.0
smmap                                      5.0.3
spandrel                                   0.4.2
SQLAlchemy                                 2.0.51
sympy                                      1.14.0
tokenizers                                 0.22.2
toml                                       0.10.2
torch                                      2.12.0+rocm7.14.0
torchaudio                                 2.11.0+rocm7.14.0
torchsde                                   0.2.6
torchvision                                0.27.0+rocm7.14.0
tqdm                                       4.70.0
trampoline                                 0.1.2
transformers                               5.14.1
triton-windows                             3.7.1.post27
typer                                      0.27.1
typing_extensions                          4.16.0
typing-inspection                          0.4.2
unpaddedbase64                             2.1.0
urllib3                                    2.7.0
uv                                         0.12.1
wheel                                      0.47.0
yarl                                       1.24.5

(comfyui_conda) D:\ComfyUI_all>

> **Comment by yyyingjjjie (2026-08-08T08:57:10Z):**

同样的报错，应该是节点bug，AI分析了一下：
确认，你贴的报错和我上一轮分析完全吻合，一行都不差：
报错发生在 ltxv_nodes.py 第 1941 行：
_qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(...)
AttributeError: 'NoneType' object has no attribute 'qk_int8_sv_f8_accum_f32_fuse_v_scale_attn'
这正是我说的那个位置：_qattn_sm89 是 None（KJNodes 在你的 SageAttention Windows 版里找不到内核函数），你的显卡是 RTX 5070 Ti（sm120），走的就是 1941 行这个 fp32 分支，所以一执行到 MiniMax H3 的自注意力就直接崩了。而且它也印证了 sageplus_sm89_available 是 False（否则会走 1943 行的 fp16 分支）。
结论不变：这不是工作流问题，是 KJNodes 实验节点和这个 SageAttention 版本的内核查找方式不兼容。内核其实都在，只是注册在 torch.ops.sageattention_qattn_sm89.* 里，节点没去那里找。
所以两条路：
不用这个节点（保持旁路），靠前面的 Patch Sage Attention KJ 节点就够用；
我帮你改一下 KJNodes 的查找逻辑，让它在找不到模块属性时改从 torch.ops 取内核——改动就几行，我已经确认你机器上所有需要的内核（包括 fp32/fp16 两个版本）都存在，改完这个节点就能正常跑。
要我直接改吗？


> **Comment by sovf4717 (2026-08-08T09:02:54Z):**

这个节点能加快生成速度吗？还是说只是像描述那样减少显存峰值占用
“实验性功能！
为 MiniMax H3 模型的自注意力模块启用定制版 SageAttention 优化，用于降低显存峰值占用；该选项会强制覆盖原有注意力计算模式。必须安装最新版本的 SageAttention 库才能正常使用。”
改吧



### #692 [open] LTX2 Memory Efficient Sage Attention Patch crashes on RTX 5090 because SageAttention ops are registered under torch.ops, not _qattn_sm89 attributes
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/692  by Setmaster 2026-07-10T05:05:20Z  comments: 2

## Describe the bug

On Windows with an RTX 5090, enabling `LTX2MemoryEfficientSageAttentionPatch` for an LTX2 workflow crashes at the first sampling step with:

```text
AttributeError: module 'sageattention._qattn_sm89' has no attribute 'qk_int8_sv_f8_accum_f32_fuse_v_scale_attn'
```

The workflow runs when this node is bypassed. Native ComfyUI `--use-sage-attention` is also present in my launch args and is not the failing path here. The failure only starts when `LTX2MemoryEfficientSageAttentionPatch` replaces the LTX2 `attn1.forward` calls.

## Why this looks like a KJNodes compatibility issue

In `nodes/ltxv_nodes.py`, the `sm120` branch calls private SageAttention extension module attributes:

```python
_qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(...)
```

On this install, those functions are not exposed as attributes on `sageattention._qattn_sm89`, but the corresponding kernels are available as registered PyTorch custom ops after importing `sageattention.core`:

```python
torch.ops.sageattention_qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn
torch.ops.sageattention_qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
torch.ops.sageattention_qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
```

SageAttention itself is not generally broken on this machine. Both local smoke tests below succeed on the RTX 5090:

```python
from sageattention import sageattn
from sageattention.core import sageattn_qk_int8_pv_fp8_cuda

o = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
o = sageattn_qk_int8_pv_fp8_cuda(
    q, k, v,
    tensor_layout="NHD",
    is_causal=False,
    qk_quant_gran="per_warp",
    pv_accum_dtype="fp32+fp16",
)
```

## Environment

```text
OS: Windows
Python: 3.13.7
GPU: NVIDIA GeForce RTX 5090
CUDA capability: (12, 0), SageAttention reports sm120
torch: 2.9.1+cu130
torch CUDA: 13.0
sageattention: 2.2.0+cu130torch2.9.0.post3
triton-windows: 3.5.1.post24
comfyui-kjnodes: 1.4.7 from Comfy Registry
ComfyUI: recent master when tested
```

## Reproduction

1. Use an LTX2 / LTX2.3 workflow that includes `LTX2MemoryEfficientSageAttentionPatch`.
2. Enable the node.
3. Queue the prompt on an RTX 5090 / sm120 system with the environment above.
4. Sampling fails near step 0 with the `_qattn_sm89` missing attribute error.

In my workflow, the node is `LTX2MemoryEfficientSageAttentionPatch`, with `triton_kernels=True`. Turning `triton_kernels` off is unlikely to address this specific error because that widget controls fused RoPE/helper kernels around the LTX2 forward path, not the failing SageAttention attention op call.

## Relevant stack trace excerpt

```text
File "<ComfyUI>\custom_nodes\comfyui-kjnodes\nodes\ltxv_nodes.py", line 1863, in ltx2_sageattn_forward
    o = _sageattn_int8_fp8_nhd(qkv, dtype)

File "<ComfyUI>\custom_nodes\comfyui-kjnodes\nodes\ltxv_nodes.py", line 1808, in _sageattn_int8_fp8_nhd
    _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(...)

AttributeError: module 'sageattention._qattn_sm89' has no attribute 'qk_int8_sv_f8_accum_f32_fuse_v_scale_attn'
```

## Local verification

The direct module attribute is absent:

```python
import sageattention.core as c
hasattr(c._qattn_sm89, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn")
# False
```

The registered op is present:

```python
import torch
import sageattention.core
hasattr(torch.ops.sageattention_qattn_sm89, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn")
# True
```

The public local SageAttention function path also works:

```text
public_sageattn (1, 128, 8, 64) torch.float16 True
public_fp8_cuda (1, 128, 8, 64) torch.float16 True
```

## Suggested fix

This is intentionally narrower than PR #518. I am not asking to replace the LTX2 low-memory implementation with only SageAttention's high-level local Python function, since that was rejected for losing the memory benefit of this node.

The narrow fix would be to keep the existing `_sageattn_int8_fp8_nhd` low-memory path, but call the registered local PyTorch custom ops when the direct `_qattn_sm89` attributes are absent. For example, resolve the op from either:

```python
_qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn
```

or:

```python
torch.ops.sageattention_qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn
```

and update `sageplus_sm89_available` detection to check the same call surface. That should preserve the node's memory-conscious q/k/v lifetime behavior while supporting current Windows SageAttention wheels that register kernels via `torch.ops`.

## Related items I found

- #518 - related but broader high-level API / legacy fallback PR, closed unmerged
- #325 - related `sm120` / `qk_quant_gran="per_warp"` discussion, closed
- #670 - related RTX 5090 SageAttention KJ crash, but for the generic patch node and a hard abort, not this LTX2 `_qattn_sm89` AttributeError


> **Comment by feigo313 (2026-08-29T13:10:48Z):**

Follow-up regression after the KJNodes update on 2026-08-29.

Environment:
- Windows 10/11
- RTX 5090 D (compute capability sm120)
- ComfyUI 0.33.1
- PyTorch 2.8.0+cu128
- Launch args: `--use-sage-attention --disable-cuda-malloc`
- MiniMax-H3 Turbo workflow using `LTX2MemoryEfficientSageAttentionPatch`

The updated `nodes/ltxv_nodes.py` again enters the `sm120/sm121` branch and calls `_qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(...)`. On this installation `_qattn_sm89` resolves to `None`, producing:

```text
AttributeError: 'NoneType' object has no attribute 'qk_int8_sv_f8_accum_f32_fuse_v_scale_attn'
```

The current local file was updated at 2026-08-29 20:50:45. A confirmed temporary workaround is to use generic `sageattn(q, k, v, tensor_layout="NHD", is_causal=False)` for sm120/sm121 when the KJNodes low-VRAM kernel is unavailable. This passes `py_compile` and a real RTX 5090 D/sm120 BF16 tensor smoke test.

Could the `sm120/sm121` branch be changed to resolve the registered SageAttention `torch.ops` kernels (or fall back to generic `sageattn`) instead of unconditionally dereferencing `_qattn_sm89`? The existing availability check should cover the exact callable used by the branch.


> **Comment by feigo313 (2026-08-29T13:31:03Z):**

Correction to my previous comment: the ComfyUI checkout has since been upgraded. The current local runtime checkout is ComfyUI v0.34.2 (commit `169fcf35a2fc163fec31338b816503ddac0d3fcf`, 2026-08-27), while the pasted error report was generated on the older v0.33.1. The SageAttention/KJNodes regression remains reproducible at the KJNodes `sm120/sm121` branch; PyTorch remains `2.8.0+cu128` on RTX 5090 D (compute capability 12.0).



### #739 [closed] Issue with ComfyUI v0.33.0, Model Preview Override (for MiniMax-H3) does not work in LOOP flow
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/739  by javawock7618 2026-08-16T02:51:04Z  comments: 2

This may be a rare case, but Model Preview Override does not work after the second iteration inside a loop.

* Initial generation → Works
* 1st LOOP iteration → Works
* 2nd LOOP iteration → Does not work

The workflow itself completes successfully without any errors.

Here is my workflow:
https://huggingface.co/javawock7618/comfy-MiniMax-H3-workflows

It worked correctly up to ComfyUI v0.32.0. This issue has been occurring since v0.33.0.


> **Comment by javawock7618 (2026-08-17T06:31:45Z):**

I suspected the loop processing might be the cause, so I tried removing it from the loop.

However, the preview worked for the first and second runs, while on the third run, it updated the preview node that was connected for the second run.

It seems like something goes wrong when the third sampling process starts within the queue.


> **Comment by javawock7618 (2026-08-23T13:14:54Z):**

My workflow had two Model Preview Override nodes, but when I reduced them to just one, all the sampling previews started updating correctly. I realized that if the model is the same, there is no need to have a separate one for each sampler.

The issue is resolved, so I’ll close this. Sorry for the trouble.



### #768 [closed] Hey Kijai, I noticed that MiniMax H3 seems to be able to do TTM-like and trajectory guidance stuff. You interested?
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/768  by ry1356 2026-10-11T18:15:25Z  comments: 1

I merged the FL2AV node and the ref2av node together, using the ref video as guidance for TTM and trajectory. But essentially it's still work on the ref2av. 
I'm using the pure FL2AV model. 
As shown in the video, the guidance seems a bit unstable, but in some way it should be able to achieve that functionality. 
After digging into the code, I found that the ref video seems to follow the "hard conditioning" route. 
So I had an idea: can we define a custom conditioning and make the ref video or other used as guidance take a different route? 

https://github.com/user-attachments/assets/8a8b7266-861f-40ab-9d3d-8be774991b49

I don't really know much about the code or the MiniMax model, so please point it out if I'm wrong.

> **Comment by ry1356 (2026-10-11T23:48:55Z):**

So here's the flow I'm thinking of:

1. Add a control video input to image to video, and the output conditioning includes control conditioning.
2. When it reaches ksampler, do a check like:

>if control_conditioning is not None and ref_video_conditioning is None:
     ref_video_conditioning = control_conditioning  
     delta = sigma
     conditioning_strength = conditioning * delta OR conditioning * delta
    ...

But the problem now is, I can't find where to make the change for step 2, so I can't really tell if this idea will actually work.

What about you think?




### #716 [closed] Workflow for MiniMax-H3-TAE
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/716  by alexfredo 2026-08-05T19:28:39Z  comments: 1

Is there a comfyui workflow for this? Thanks https://huggingface.co/Kijai/MiniMax-H3-TAE

> **Comment by alexfredo (2026-08-05T22:42:10Z):**

Someone knows what nodes I need to put in comfyui for use it ? thanks




### #710 [closed] [FEATURE REQUEST] MiniMax H3 Chunk FeedForward
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/710  by Moussack 2026-08-04T03:40:10Z  comments: 1

Got OOM above 158 frames 😭😭

> **Comment by kijai (2026-08-04T09:00:41Z):**

Chunking ffn doesn't really help with this model, because attention uses more VRAM than that.



### #706 [closed] MiniMax H3 Chunk FeedForward Is it possible?
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/706  by neural-dreamer 2026-08-03T04:48:42Z  comments: 1

Hi!
Is it possible to add Chunk FeedForward to the MiniMax H3 model?

> **Comment by kijai (2026-08-03T05:39:04Z):**

Generally the attention is the peak VRAM on this model, so it wouldn't help.



### #726 [closed] Commit bdcb886a4705a03cf40f4a7226de9fc7c059fc90 breaks audio on H3 when using EasyCache
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/726  by mholtgraewe 2026-08-07T11:38:16Z  comments: 1

https://github.com/Comfy-Org/ComfyUI/commit/bdcb886a4705a03cf40f4a7226de9fc7c059fc90 breaks audio generation when using the EasyCache node in a H3 ref2vid workflow. Please see the before and after examples. 

Bypassing the EasyCache node or reverting to https://github.com/Comfy-Org/ComfyUI/commit/2eb609766a749e3104485979615e062e401bab97 fixes the issue.

https://github.com/user-attachments/assets/d006ef62-5931-4330-aed3-616cef213ada

https://github.com/user-attachments/assets/afbfa233-4ca9-4796-9da1-4b9bac924b53

Settings:
res_multistep / simple / 20 steps / res 16:9 at 0.2MP



> **Comment by mholtgraewe (2026-08-07T11:58:46Z):**

Nevermind, I just saw you've already addressed the issue in https://github.com/Comfy-Org/ComfyUI/pull/15390.



### #711 [closed] Model Preview Override node is not working for H3
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/711  by lewddev 2026-08-04T07:20:28Z  comments: 1

<img width="1333" height="461" alt="Image" src="https://github.com/user-attachments/assets/544f63a0-9883-4f0a-b983-0f39fd464930" />
The preview is not showing.

> **Comment by kijai (2026-08-04T09:01:46Z):**

Should work now.



### #714 [closed] Missing @triton.jit on _quant_query_per_thread_int8_i64_kernel
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/714  by Code-Venkas 2026-08-04T15:33:51Z  comments: 1

# Missing `@triton.jit` on `_quant_query_per_thread_int8_i64_kernel` (87f810e)

`87f810e` ("Work around sageattention issue with large sequences (MinimaxH3)") vendors two int8 quantization kernels into `nodes/ltxv_nodes.py`, but only the *key* kernel got the `@triton.jit` decorator. The *query* kernel is left as a plain Python function, so launching it with Triton's `kernel[grid](...)` syntax raises immediately.

Any MiniMax H3 sampling run through `MiniMaxH3MemoryEfficientSageAttentionPatch` fails on the first step.

## Error

```
File "custom_nodes\ComfyUI-KJNodes\nodes\ltxv_nodes.py", line 2081, in minimax_sageattn_forward
  o = _sageattn_int8_fp8_nhd(qkv, dtype)
File "custom_nodes\ComfyUI-KJNodes\nodes\ltxv_nodes.py", line 1882, in _sageattn_int8_fp8_nhd
  q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(q, k, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
File "custom_nodes\ComfyUI-KJNodes\nodes\ltxv_nodes.py", line 1843, in _per_thread_int8_i64
  _quant_query_per_thread_int8_i64_kernel[grid](
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
TypeError: 'function' object is not subscriptable
```

Note this is *not* a missing-Triton problem — `HAS_TRITON` is `True` here. If Triton were absent the `if HAS_TRITON:` block would never define the symbol and this would be a `NameError` instead.

## Cause

`nodes/ltxv_nodes.py`, current `main`:

```
1749  if HAS_TRITON:
1750      # Vendored from sageattention's quant_per_thread.py with the row offsets promoted to int64 ...
1751      def _quant_query_per_thread_int8_i64_kernel(Input, Output, Scale, L,     <-- no decorator
...
1777      @triton.jit
1778      def _quant_key_per_thread_int8_i64_kernel(Input, Output, Scale, L,       <-- has it
```

## Fix

```diff
     # Vendored from sageattention's quant_per_thread.py with the row offsets promoted to int64 to avoid overflow on large sequences.
+    @triton.jit
     def _quant_query_per_thread_int8_i64_kernel(Input, Output, Scale, L,
```

Confirmed locally — after adding the decorator both symbols load as `JITFunction`, and MiniMax H3 sampling runs:

```
HAS_TRITON: True
_quant_query_per_thread_int8_i64_kernel   JITFunction   subscriptable=True
_quant_key_per_thread_int8_i64_kernel     JITFunction   subscriptable=True
```

## Environment

| | |
|---|---|
| KJNodes | `87f810e` |
| ComfyUI | 0.30.0 |
| GPU | RTX 3090 Ti (SM86) |
| OS | Windows 11 |
| Python | 3.12.12 |
| PyTorch | 2.13.0+cu130 |
| triton-windows | 3.7.1.post27 |
| SageAttention | 2.2.0+cu130torch2.10.0andhigher.post6 |

> **Comment by kijai (2026-08-04T15:36:54Z):**

Was already fixed.



### #791 [open] Any chance of a Krea2PromptWeight style node for Minimax H3?
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/791  by rslaton1 2026-11-06T20:56:46Z  comments: 0




### #715 [open] MiniMax H3 AudioVideoMask node support
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/715  by zwukong 2026-08-05T10:49:30Z  comments: 0

LTXVAudioVideoMask node is very useful, thanks so much. MiniMax H3 AudioVideoMask node is needed too. Hope you can support a bit. I already see some guy use audio to drive video in MiniMax H3 , just lantent noise mask


### #737 [open] [FEATURE] Minimax H3 LoRa Loader Advanced
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/737  by GZ-572 2026-08-14T18:10:07Z  comments: 0

Can we please have an advanced LoRa loader for Minimax H3, that allow us to remove the audio from the LoRas. 
User created LoRa can have bad audio, it will be great if we disable it.

Thank you so much for everything.


### #731 [open] Context Windows node for MiniMax H3?
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/731  by neural-dreamer 2026-08-08T13:29:09Z  comments: 0

Is it possible to get Context Windows to work with the MiniMax H3 model? Or maybe separate Context Windows node like for LTX?
I tried using the Context Windows (Manual) node, but I couldn't get it to work.


### #722 [open] Error when extracting diff between minimax H3 model variants as lora
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/722  by mcDandy 2026-08-06T19:52:35Z  comments: 0

I tried extracting diff between fl and ref as lora but it failed.

Reason to do it: To save space on ssd. I had to download 2 19GB models. Used custom extension to get cosine similarity between the variants. Only *..adaln_proj.linear.weight did not result in streight 1.0. That is without any decomposition large saving.

found on: 35e5956193769d18a13136cdedb73a36a05c73e6

The error I am getting: 
```
[ERROR] !!! Exception during processing !!! 'Linear' object has no attribute 'weight_scale'
[ERROR] Traceback (most recent call last):
  File "D:\stability\Data\Packages\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\stability\Data\Packages\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\stability\Data\Packages\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\stability\Data\Packages\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "D:\stability\Data\Packages\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\stability\Data\Packages\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\stability\Data\Packages\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\lora_nodes.py", line 211, in execute
    kp = original.get_key_patches("diffusion_model.")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\stability\Data\Packages\ComfyUI\comfy\model_patcher.py", line 867, in get_key_patches
    weight, set_func, convert_func = get_key_weight(self.model, k)
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\stability\Data\Packages\ComfyUI\comfy\model_patcher.py", line 234, in get_key_weight
    weight = getattr(op, op_keys[1])
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\stability\Data\Packages\ComfyUI\venv\Lib\site-packages\torch\nn\modules\module.py", line 1967, in __getattr__
    raise AttributeError(
AttributeError: 'Linear' object has no attribute 'weight_scale'
```

weigth key in each block: (part of output from custom node (key:; shape of fl version; shape of ref version; per dim0 cossim))
```
diffusion_model.blocks.49.norm1.weight: [5376] [5376]tensor(1.0000)
diffusion_model.blocks.49.norm2.weight: [5376] [5376]tensor(1.0000)
diffusion_model.blocks.49.attn.qkv_proj.weight: [21504, 5376] [21504, 5376]tensor([0.9944, 0.9931, 0.9997,  ..., 0.9998, 0.9998, 0.9998])
diffusion_model.blocks.49.attn.qkv_proj.weight_scale: [21504, 1] [21504, 1]tensor([1., 1., 1.,  ..., 1., 1., 1.])
diffusion_model.blocks.49.attn.qkv_proj.comfy_quant: [72] [72]tensor(1.0000)
diffusion_model.blocks.49.attn.q_norm.weight: [128] [128]tensor(1.0000)
diffusion_model.blocks.49.attn.k_norm.weight: [128] [128]tensor(1.0000)
diffusion_model.blocks.49.attn.out_proj.weight: [5376, 7168] [5376, 7168]tensor([0.9998, 0.9999, 0.9996,  ..., 0.9999, 0.9999, 0.9998])
diffusion_model.blocks.49.attn.out_proj.weight_scale: [5376, 1] [5376, 1]tensor([1., 1., 1.,  ..., 1., 1., 1.])
diffusion_model.blocks.49.attn.out_proj.comfy_quant: [72] [72]tensor(1.0000)
diffusion_model.blocks.49.mlp.fc1.weight: [28672, 5376] [28672, 5376]tensor([0.9972, 0.9994, 0.9989,  ..., 0.9996, 0.9997, 0.9996])
diffusion_model.blocks.49.mlp.fc1.weight_scale: [28672, 1] [28672, 1]tensor([1., 1., 1.,  ..., 1., 1., 1.])
diffusion_model.blocks.49.mlp.fc1.comfy_quant: [72] [72]tensor(1.0000)
diffusion_model.blocks.49.mlp.fc2.weight: [5376, 14336] [5376, 14336]tensor([0.9995, 0.9997, 0.9996,  ..., 0.9996, 0.9997, 0.9994])
diffusion_model.blocks.49.mlp.fc2.weight_scale: [5376, 1] [5376, 1]tensor([1., 1., 1.,  ..., 1., 1., 1.])
diffusion_model.blocks.49.mlp.fc2.comfy_quant: [72] [72]tensor(1.0000)
diffusion_model.blocks.49.adaln_proj.linear.bias: [96768] [96768]tensor(0.9999)
diffusion_model.blocks.49.adaln_proj.linear.weight: [96768, 8] [96768, 8]tensor([-0.8217, -0.7610, -0.9150,  ..., -0.3507, -0.6802, -0.3397])
```


### #759 [open] Why does using MiniMax H3 Mem Eff Sage Attention Patch cause the video to appear grayscale at high resolution?
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/759  by sol-zs451 2026-09-23T10:44:10Z  comments: 0

<img width="445" height="145" alt="Image" src="https://github.com/user-attachments/assets/0b168a6d-b82c-41d5-8af5-f8fb9a5d7c68" />

<img width="536" height="660" alt="Image" src="https://github.com/user-attachments/assets/423c9b00-cf2e-47c2-90a7-40fd02132d74" />

<img width="990" height="559" alt="Image" src="https://github.com/user-attachments/assets/dc10bacb-8dd1-46d5-9882-93e88845f9c2" />

The resolution is 1088*1920, and there are no errors, but the generated video is grayish. It works fine with Kitchen, although it might be a little slower.


### #738 [open] Feature Request: Streamed QKV + Near-Lossless 4-bit K/V Compression for MiniMax H3 Low-VRAM Attention
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/738  by eliowang6 2026-08-16T02:22:16Z  comments: 0

Hi, I would like to suggest a possible extension to the existing **MiniMax H3 Low VRAM Attention** implementation.

The problem appears in long-sequence H3 Ref2VA workloads with long target videos and multiple image/video references.

On a 12 GB GPU, I encountered:

```text
Currently allocated     : 6.09 GiB
Requested               : 6.11 GiB
Device limit            : 11.94 GiB
Free                    : 0 bytes
```

After discussing this with another H3 attention implementation author, the ~6.11 GiB allocation was identified as almost exactly the size of H3's full BF16 fused-QKV output at approximately **152,263 tokens**.

So even with model offloading and low-VRAM attention, H3 can still OOM simply because `qkv_proj` materializes the entire BF16 QKV tensor at once.

## Proposed direction

A possible solution would be to avoid creating the full BF16 QKV tensor at all.

Instead:

```text
hidden states
     ↓
streamed / chunked QKV projection
     │
     ├── Q → BF16/FP16
     │
     ├── K → rotate → pack directly to INT4
     │
     └── V → rotate → pack directly to INT4
```

Then the attention kernel would consume packed K/V directly:

```text
Q BF16
+
packed K4 / V4
      ↓
load one K/V tile
      ↓
dequantize in registers/shared memory
      ↓
attention computation
```

This is important because doing:

```text
full BF16 qkv_proj
→ split Q/K/V
→ quantize K/V
```

does not solve the OOM: the ~6 GiB tensor has already been allocated.

## Potential memory reduction

For this example workload:

```text
BF16 Q ≈ 2.04 GiB
BF16 K ≈ 2.04 GiB
BF16 V ≈ 2.04 GiB

Total ≈ 6.1 GiB
```

Keeping Q in BF16 while storing K/V at fixed-width 4-bit would theoretically reduce the payload to roughly:

```text
Q BF16 ≈ 2.04 GiB
K INT4 ≈ 0.51 GiB
V INT4 ≈ 0.51 GiB

Total ≈ 3.06 GiB + scales/metadata
```

That could be enough to make some currently impossible 12 GB H3 Ref2VA workloads runnable.

## Why not naive INT4?

I am specifically thinking of a **quality-oriented rotated INT4 scheme**, inspired by work such as TurboQuant rather than simple scalar INT4.

TurboQuant demonstrated that rotation/distribution shaping plus attention-aware quantization can compress LLM K/V to very low precision with near-quality-neutral results in its evaluated settings.

HyperQuant also provides evidence that aggressive rotation-based low-bit quantization can remain very accurate in large diffusion/video models, including LTX-2, although its LTX-2 experiment covered linear weights rather than K/V activations.

So near-lossless H3 K/V compression should still be treated as an experimental hypothesis and validated directly on H3.

A conservative first implementation might use:

* Q: BF16/FP16
* K/V: fixed-width rotated INT4
* per-group or per-tile scales
* no variable-length entropy coding initially
* reference/text/audio conditioning K/V: BF16 or INT8
* generated-video K/V: INT4
* optional higher precision for first/last Transformer blocks
* optional higher precision for final diffusion steps

Fixed-width INT4 seems more practical than variable-length Rice coding here because it allows predictable Triton tile addressing.

## Important: dense fallback also needs packed K/V support

If the optimized path uses packed K/V but a dense fallback does:

```text
packed K/V
→ fully expand to BF16 K/V
→ dense attention
```

the original multi-gigabyte allocation can reappear.

So both optimized attention and dense fallback would ideally consume packed K/V tile-by-tile.

## Why this may fit KJNodes

The existing H3 Low VRAM Attention implementation already focuses on reducing H3 activation lifetime and peak VRAM.

This proposal targets another remaining peak that cannot be solved purely by earlier activation release:

**the fused QKV projection itself.**

It could potentially complement:

* MiniMax H3 Low VRAM Attention
* early activation release
* FeedForward chunking
* model offloading

without skipping diffusion steps or reusing stale activations.

Would this be something you would be interested in experimenting with as an H3-specific low-VRAM backend?



### #736 [open] MiniMax H3 Memory Efficient SageAttention Patch fails on newer SageAttention wheels because `_qattn_sm89` resolves to `None`
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/736  by QQWWLOOK 2026-08-14T14:07:52Z  comments: 0

## Describe the bug

`MiniMax H3 Mem Eff Sage Attention Patch` fails on an RTX 5070 Ti (Blackwell, `sm120`) with a newer SageAttention wheel.

The compiled SageAttention operators are available under `torch.ops.sageattention_qattn_sm89`, but `nodes/ltxv_nodes.py::_resolve_qattn()` only probes attributes on the imported Python / `.pyd` modules. As a result, `_qattn_sm89` is resolved as `None` even though the required operators are registered and callable through `torch.ops`.

The runtime later fails with:

```text
AttributeError: 'NoneType' object has no attribute
'qk_int8_sv_f8_accum_f32_fuse_v_scale_attn'
```

## Relevant traceback

```text
File "ComfyUI-KJNodes/nodes/ltxv_nodes.py", line 2106, in minimax_sageattn_forward
    o = _sageattn_int8_fp8_nhd(qkv, dtype)

File "ComfyUI-KJNodes/nodes/ltxv_nodes.py", line 1951, in _sageattn_int8_fp8_nhd
    _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(...)

AttributeError: 'NoneType' object has no attribute
'qk_int8_sv_f8_accum_f32_fuse_v_scale_attn'
```

## Environment

- GPU: NVIDIA GeForce RTX 5070 Ti
- Compute capability: `(12, 0)` / `sm120`
- OS: Windows
- Python: 3.13.11
- PyTorch: 2.9.1+cu130
- CUDA reported by PyTorch: 13.0
- ComfyUI: 0.31.1
- Node: `MiniMaxH3MemoryEfficientSageAttentionPatch`

## Verification

The old module-attribute probe fails, while the operators are present in the new registration namespace:

```python
import torch
import sageattention.core

ns = torch.ops.sageattention_qattn_sm89

print(hasattr(ns, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn"))
print(hasattr(ns, "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf"))
print(hasattr(ns, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf"))
```

Output:

```text
True
True
True
```

## Proposed fix

Include the `torch.ops` namespace as a candidate while retaining compatibility with older SageAttention wheels:

```diff
 def _resolve_qattn(arch):
     try:
         core = importlib.import_module("sageattention.core")
     except Exception:
         return None
-    candidates = [getattr(core, f"_qattn_{arch}", None), getattr(core, f"{arch}_compile", None)]
+    candidates = [
+        getattr(torch.ops, f"sageattention_qattn_{arch}", None),
+        getattr(core, f"_qattn_{arch}", None),
+        getattr(core, f"{arch}_compile", None),
+    ]
```

After this change, `_resolve_qattn("sm89")` returns `_OpNamespace`, and the MiniMax H3 Memory Efficient SageAttention node can reach the registered Blackwell-compatible kernels instead of failing on `None`.

This preserves the existing fallback candidates for older SageAttention package layouts.



### #741 [open] Node's tooltip to have consumer/noob friendly info?
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/741  by CherrySkyArt 2026-08-21T02:08:53Z  comments: 0

Hello,

I am a noob, and when I was looking at some KJ nodes, the tooltip was very technical.   

As a noob I'd like a line at the end/start that says something simple like "Increasing head_chunks slows generation, but allows higher VRAM consumption, no quality loss" idk if that is true or not, but nothing in the tooltip explicitly explains effects on end results or number, since I dont know understand any technicals :( 

I am grateful for the nodes and tooltips <3 but I'm giving some feedback b/c i'm like "I'm still so confused, i dont understand the help-tooltip at all"

--- 
Specifically the nodes I was looking at was "MiniMax H3 Low VRAM Attention" and "MiniMax H3 Chunk FeedForward". 

<img width="1417" height="727" alt="Image" src="https://github.com/user-attachments/assets/c54d47b4-cd20-409d-95d7-d69829ec02e1" />


The actual node writes:
` Chunks the MiniMax H3 feedforward (SwiGLU) over the packed token dim to reduce peak VRAM usage. Each chunk still goes through ComfyUI's fused INT8 kernel when the model is quantized, and since activations are quantized per-token the output matches the unchunked model.`

cheers <3


### #745 [open] Feature Request: can "minimax_nodes.py" include layer-based attn strength filtering for AUDIO in LoRAs like "LTX2LoraLoaderAdvanced"?
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/745  by 808charlie 2026-08-26T09:55:08Z  comments: 0

Is there potential to include layer-based attention strength filtering for AUDIO in LoRAs like "LTX2LoraLoaderAdvanced"?

This was useful in Ltx2.3 as it allowed visual styles from LoRAs to be utilized, whilst the audio could be attenuated.

Is this sort of concept viable in Minimax H3?

Your LTX Lora Loader Advanced was useful in its own right, but also gave me enough understanding to crudely hammer together a version of RGthree's PowerLoraLoader allowing audio adjustment of V2A, A2V and Audio laryers https://github.com/808charlie/rgthree-comfy

No worry if not; thanks for H3 nodes to date and all your other unceasing work for the community.


### #763 [open] MemoryEfficientSageAttentionPatch silently produces wrong attention output above ~199.7k tokens (int32 offset overflow)
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/763  by DreamRefiner 2026-09-26T17:58:01Z  comments: 0

### Summary
`MiniMaxH3MemoryEfficientSageAttentionPatch` (nodes/ltxv_nodes.py, sm120 branch) silently corrupts the attention output once the sequence exceeds roughly **199,700 tokens**. No NaN, no error, no warning — the output is just wrong. At the length where it first crossed the line, relative error vs. SDPA jumped from ~4% to ~19%.

This is reachable in normal use: a 1920×1088 second pass of about 14 s (87 latent frames) plus references is already over the line.

### Environment
- GPU: RTX PRO 6000 Blackwell Workstation (sm120), Windows
- ComfyUI 0.37.2, PyTorch 2.12.0+cu130
- SageAttention 2.2.0.post5 (cu130), comfy-kitchen 0.2.35
- ComfyUI-KJNodes: latest as of 2026-09-24

### Measurements (real H3 shapes, 48 heads × 128 head_dim, compared against SDPA)
| tokens  | KJ patch rel. error | correct result rel. error |
|---------|---------------------|---------------------------|
| 199,000 | 3.9%                | 3.9% (bit-identical)      |
| 201,781 | **19%**             | 4%                        |

### Cause (from our analysis)
When Q/K/V reach the quantization step as non-contiguous (strided) tensors, an internal offset is computed in 32-bit and overflows past ~199.7k tokens, so reads go out of bounds and pick up garbage instead of crashing.

Calling `.contiguous()` on Q/K/V before quantization fixes it:
- below the threshold, output is **bit-identical** to the current node;
- above it, output is correct (~4% error, same as below the line).

Cost: roughly 9 GB of extra temporary VRAM at ~200k tokens.

### Real-world symptom
- Workflow: MiniMax H3 ref2va, 2-pass (960×544 → 3D latent upscale ×2 → 1920×1088), 87 latent frames, 7 reference images + reference audio → 201,781 tokens in pass 2.
- With only a turbo LoRA the degradation is subtle. With a full-coverage LoRA stacked on top (a LoKr touching all attn/mlp layers), pass 2 collapses into a uniform brown/gray field.
- Same workflow, same seed, with the `.contiguous()` fix: clean output.
- Shorter clips below the line are unaffected, which is why this is easy to miss.

### Possibly related (upstream SageAttention, not this node)
Even with contiguous inputs, the fp8 attention kernel itself starts returning wrong results at roughly **299.6k tokens** (18% error at 310k), again silently. It may be worth adding a sequence-length guard with a warning, or falling back to exact attention past that length.

### Suggestion
1. Make Q/K/V contiguous before quantization (or use 64-bit offsets).
2. Add a length check that warns or falls back instead of silently returning bad output.

Happy to share the patched node and a standalone repro script if useful.


### #723 [open] [Bug] MiniMaxH3MemoryEfficientSageAttentionPatch: Fatal Python error: Aborted after 3-4 runs on RTX 5070 Ti (sm120), escalates to driver TDR
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/723  by shiloha-z 2026-08-07T07:11:03Z  comments: 0

Environment: RTX 5070 Ti (sm120), ComfyUI v0.30.1, PyTorch 2.11.0+cu130, sageattention 2.2.0+cu130torch2.10.0andhigher.post5, Windows 10. Repo: https://github.com/kijai/ComfyUI-KJNodes

The MiniMaxH3MemoryEfficientSageAttentionPatch node applies fine (sage 2.2.0 detects sm120), but after 3-4 consecutive generations the ComfyUI process dies with "Fatal Python error: Aborted" during sampling (observed at step 4/8), and the crash escalates to nvlddmkm TDR — the whole GPU becomes unavailable until reboot. 

Repro:
1. Load the H3 turbo workflow with this patch node enabled
2. Queue 4 runs, changing noise_seed each time 
3. Runs 1-3 complete clean (~140s each), run 4 aborts mid-sampling

My AI Analysis: in _sageattn_int8_fp8_nhd (ltxv_nodes.py:1927-1944), sm120/sm121 dispatches to the _qattn_sm89 kernels (Ada-compiled) — there is no native sm120 kernel. The sm89 cubins run on Blackwell in compatibility mode and corrupt memory after repeated calls. Related: issue #721. 


### #769 [open] An error occurred during the execution of the node.
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/769  by tianchendiwang 2026-10-13T05:34:34Z  comments: 0

An error occurred during the execution of the node.

# ComfyUI Error Report
## Error Details
- **Node ID:** 138
- **Node Type:** MiniMaxH3MemoryEfficientSageAttentionPatch
- **Exception Type:** RuntimeError
- **Exception Message:** RuntimeError: sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.

## Stack Trace
```
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\execution.py", line 314, in process_inputs
    result = f(**inputs)

  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)

  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\comfy_api\latest\_io.py", line 2208, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)

  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\ltxv_nodes.py", line 2183, in execute
    raise RuntimeError("sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.")

```
## System Information
- **ComfyUI Version:** 0.39.0
- **Arguments:** F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\main.py --auto-launch --preview-method auto --cuda-malloc
- **OS:** win32
- **Python Version:** 3.13.11 (tags/v3.13.11:6278944, Dec  5 2025, 16:26:58) [MSC v.1944 64 bit (AMD64)]
- **Embedded Python:** false
- **PyTorch Version:** 2.9.1+cu130
## Devices

- **Name:** cuda:0 NVIDIA GeForce RTX 4070 Ti : cudaMallocAsync
  - **Type:** cuda
  - **VRAM Total:** 12878086144
  - **VRAM Free:** 11553079296
  - **Torch VRAM Total:** 33554432
  - **Torch VRAM Free:** 25034752

## Logs
```
2026-10-13T13:13:13.574202 - ^[[38;5;153m[13:13:13] ℹ️    - TE MAN H3素材转接口 (TE_H3_Reference_Bridge)^[[0m2026-10-13T13:13:13.574212 - 
2026-10-13T13:13:13.574227 - ^[[38;5;153m[13:13:13] ℹ️    - TE MAN HappyH Video (TE_image_pro_happyh_video)^[[0m2026-10-13T13:13:13.574237 - 
2026-10-13T13:13:13.574252 - ^[[38;5;153m[13:13:13] ℹ️    - TE MAN 图片压缩 (TE_image_pro_compress)^[[0m2026-10-13T13:13:13.574262 - 
2026-10-13T13:13:13.574276 - ^[[38;5;153m[13:13:13] ℹ️    - TE MAN Text文本处理 (TE_image_pro_text_tools)^[[0m2026-10-13T13:13:13.574287 - 
2026-10-13T13:13:13.574302 - ^[[38;5;153m[13:13:13] ℹ️    - TE MAN MiniMaxH3增强提示词Prompt TEXT (TE_prompt_text)^[[0m2026-10-13T13:13:13.574312 - 
2026-10-13T13:13:13.574329 - ^[[38;5;153m[13:13:13] ℹ️    - TE MAN TEXT文本展示 (TE_text_display)^[[0m2026-10-13T13:13:13.574340 - 
2026-10-13T13:13:13.574358 - ^[[38;5;153m[13:13:13] ℹ️    - TE MAN VEO Video (TE_image_pro_te_veo_video)^[[0m2026-10-13T13:13:13.574368 - 
2026-10-13T13:13:13.574383 - ^[[38;5;153m[13:13:13] ℹ️    - TE MAN Gemini 特渠 (TE_image_pro_special_async2)^[[0m2026-10-13T13:13:13.574392 - 
2026-10-13T13:13:17.612626 - ^[[34mWAS Node Suite: ^[[0mOpenCV Python FFMPEG support is enabled^[[0m2026-10-13T13:13:17.612659 - 
2026-10-13T13:13:17.612709 - ^[[34mWAS Node Suite ^[[93mWarning: ^[[0m`ffmpeg_bin_path` is not set in `F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\was-ns\was_suite_config.json` config file. Will attempt to use system ffmpeg binaries if available.^[[0m2026-10-13T13:13:17.612724 - 
2026-10-13T13:13:19.233296 - FETCH ComfyRegistry Data: 25/2002026-10-13T13:13:19.233329 - 
2026-10-13T13:13:21.773346 - ^[[34mWAS Node Suite: ^[[0mFinished.^[[0m ^[[32mLoaded^[[0m ^[[0m220^[[0m ^[[32mnodes successfully.^[[0m2026-10-13T13:13:21.773378 - 
2026-10-13T13:13:21.773420 - 
	^[[3m^[[93m"Art is the lie that enables us to recognize the truth."^[[0m^[[3m - Friedrich Nietzsche^[[0m
2026-10-13T13:13:21.773434 - 
2026-10-13T13:13:21.804688 - WeiLin-Comfyui-Tools ==> 所有依赖项已安装。2026-10-13T13:13:21.804770 - 
2026-10-13T13:13:21.823800 - 检查数据库: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\weilin-comfyui-tools\app\server\dao\../../../user_data/userdatas_zh_CN_tags.db (1.1 MB)2026-10-13T13:13:21.823830 - 
2026-10-13T13:13:21.832682 -   数据库完整性检查通过2026-10-13T13:13:21.832714 - 
2026-10-13T13:13:21.832939 - 检查数据库: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\weilin-comfyui-tools\app\server\dao\../../../user_data/userdatas_zh_CN_history.db (2.4 MB)2026-10-13T13:13:21.832959 - 
2026-10-13T13:13:21.867866 -   数据库完整性检查通过2026-10-13T13:13:21.867899 - 
2026-10-13T13:13:21.868150 - 检查数据库: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\weilin-comfyui-tools\app\server\dao\../../../user_data/userdatas_zh_CN_danbooru.db (0.2 MB)2026-10-13T13:13:21.868171 - 
2026-10-13T13:13:21.870824 -   数据库完整性检查通过2026-10-13T13:13:21.870868 - 
2026-10-13T13:13:21.912772 - ======== WeiLin插件服务已启动 ========2026-10-13T13:13:21.912841 - 
2026-10-13T13:13:21.912863 - ======== WeiLin Server Init ========2026-10-13T13:13:21.912877 - 
2026-10-13T13:13:21.915293 - ^[[32m[INFO]^[[0m 
Import times for custom nodes:
2026-10-13T13:13:21.915404 - ^[[32m[INFO]^[[0m    0.0 seconds (IMPORT FAILED): F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SolAttn_triton
2026-10-13T13:13:21.915453 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\websocket_image_save.py
2026-10-13T13:13:21.915487 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-ColorMigrationScheduler
2026-10-13T13:13:21.915517 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-ColorSuperCaller
2026-10-13T13:13:21.915547 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\SD-Latent-Upscaler
2026-10-13T13:13:21.915574 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-tagged-broadcast
2026-10-13T13:13:21.915600 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui_memory_cleanup
2026-10-13T13:13:21.915626 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-POS-Prompt
2026-10-13T13:13:21.915655 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\QuickRunEverywhere
2026-10-13T13:13:21.915681 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\TE-Speed-MiniMaxH3
2026-10-13T13:13:21.915709 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\My_Super_Resolution_picker
2026-10-13T13:13:21.915736 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-BSS_FLSampler
2026-10-13T13:13:21.915761 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Anima-LLLite
2026-10-13T13:13:21.915788 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\sdxl-recommended-res-calc
2026-10-13T13:13:21.915815 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\cg-use-everywhere-main
2026-10-13T13:13:21.915842 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\Anima_Regional_Canvas
2026-10-13T13:13:21.915867 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\stability-ComfyUI-nodes
2026-10-13T13:13:21.915892 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\orion4d_autocachedpreview
2026-10-13T13:13:21.915919 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-inpaint-cropandstitch
2026-10-13T13:13:21.915948 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-cyberdelia-prompt-format
2026-10-13T13:13:21.915975 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\tulut_comfyui_Gadget
2026-10-13T13:13:21.916001 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-ap-clipseg-light
2026-10-13T13:13:21.916028 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-easycontrol-ksamplercompat
2026-10-13T13:13:21.916055 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-wd14-tagger
2026-10-13T13:13:21.916080 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-GGUF
2026-10-13T13:13:21.916105 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\MY Character Select
2026-10-13T13:13:21.916130 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI_IPAdapter_plus
2026-10-13T13:13:21.916154 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-inpaint-nodes
2026-10-13T13:13:21.916180 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-krea2edit
2026-10-13T13:13:21.916207 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\gprompts
2026-10-13T13:13:21.916234 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyUI-llama-TE-main
2026-10-13T13:13:21.916261 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui_essentials
2026-10-13T13:13:21.916288 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI_UltimateSDUpscale
2026-10-13T13:13:21.916314 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Anima-LoRA-Training
2026-10-13T13:13:21.916340 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-segment-anything-2
2026-10-13T13:13:21.916367 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-jakeupgrade
2026-10-13T13:13:21.916393 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-videohelpersuite
2026-10-13T13:13:21.916418 - ^[[32m[INFO]^[[0m    0.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-ppm
2026-10-13T13:13:21.916444 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-PIX_Nvidia_RTX_Deblur_Denoise
2026-10-13T13:13:21.916479 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\TE_MAN-main
2026-10-13T13:13:21.916517 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-inspire-pack
2026-10-13T13:13:21.916547 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-kjnodes
2026-10-13T13:13:21.916573 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\prompt-manager
2026-10-13T13:13:21.916600 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui_controlnet_aux
2026-10-13T13:13:21.916626 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\Character-selector
2026-10-13T13:13:21.916651 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\rgthree-comfy
2026-10-13T13:13:21.916677 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\efficiency-nodes-comfyui
2026-10-13T13:13:21.916703 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-LTXVideo
2026-10-13T13:13:21.916729 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\weilin-comfyui-tools
2026-10-13T13:13:21.916755 - ^[[32m[INFO]^[[0m    0.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI_LayerStyle_Advance
2026-10-13T13:13:21.916779 - ^[[32m[INFO]^[[0m    0.2 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui_segment_anything
2026-10-13T13:13:21.916804 - ^[[32m[INFO]^[[0m    0.2 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-bsk_UI
2026-10-13T13:13:21.916829 - ^[[32m[INFO]^[[0m    0.2 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-impact-subpack
2026-10-13T13:13:21.916856 - ^[[32m[INFO]^[[0m    0.3 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-RMBG
2026-10-13T13:13:21.916882 - ^[[32m[INFO]^[[0m    0.3 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI_LayerStyle
2026-10-13T13:13:21.916906 - ^[[32m[INFO]^[[0m    0.3 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Impact-Pack
2026-10-13T13:13:21.916930 - ^[[32m[INFO]^[[0m    0.3 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\seedvr2_videoupscaler
2026-10-13T13:13:21.916956 - ^[[32m[INFO]^[[0m    1.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\RES4LYF
2026-10-13T13:13:21.916982 - ^[[32m[INFO]^[[0m    1.1 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Manager
2026-10-13T13:13:21.917008 - ^[[32m[INFO]^[[0m    1.6 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-EasyColorCorrector
2026-10-13T13:13:21.917034 - ^[[32m[INFO]^[[0m    2.5 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody
2026-10-13T13:13:21.917061 - ^[[32m[INFO]^[[0m    3.8 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-mixlab-nodes
2026-10-13T13:13:21.917087 - ^[[32m[INFO]^[[0m    5.6 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI_Custom_Nodes_AlekPet
2026-10-13T13:13:21.917114 - ^[[32m[INFO]^[[0m    8.2 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\was-ns
2026-10-13T13:13:21.917141 - ^[[32m[INFO]^[[0m   10.0 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Easy-Use
2026-10-13T13:13:21.917168 - ^[[32m[INFO]^[[0m   17.9 seconds: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui_LLM_party
2026-10-13T13:13:21.917196 - ^[[32m[INFO]^[[0m 
2026-10-13T13:13:21.917801 - 检查数据库: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\weilin-comfyui-tools\app\server\dao\../../../user_data/userdatas_zh_CN_tags.db (1.1 MB)2026-10-13T13:13:21.917827 - 
2026-10-13T13:13:21.926317 -   数据库完整性检查通过2026-10-13T13:13:21.926348 - 
2026-10-13T13:13:22.507994 - ^[[32m[INFO]^[[0m Using RAM pressure cache.
2026-10-13T13:13:22.508847 - ^[[32m[INFO]^[[0m Starting server

2026-10-13T13:13:22.509271 - ^[[32m[INFO]^[[0m To see the GUI go to: http://127.0.0.1:8188
2026-10-13T13:13:22.511618 - [TagImageQueue] Worker started2026-10-13T13:13:22.511648 - 
2026-10-13T13:13:24.500340 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-10-13T13:13:24.501067 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /extensions/core/clipspace.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-10-13T13:13:24.501601 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /extensions/core/groupNode.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-10-13T13:13:24.505183 - F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-mixlab-nodes\webApp\lib/photoswipe-lightbox.esm.min.js2026-10-13T13:13:24.505220 - 
2026-10-13T13:13:24.506246 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /extensions/core/widgetInputs.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-10-13T13:13:24.566791 - F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-mixlab-nodes\webApp\lib/pickr.min.js2026-10-13T13:13:24.566869 - 
2026-10-13T13:13:24.568637 - F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-mixlab-nodes\webApp\lib/photoswipe.min.css2026-10-13T13:13:24.568708 - 
2026-10-13T13:13:24.639163 - F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-mixlab-nodes\webApp\lib/classic.min.css2026-10-13T13:13:24.639214 - 
2026-10-13T13:13:24.641124 - F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-mixlab-nodes\webApp\lib/model-viewer.min.js2026-10-13T13:13:24.641161 - 
2026-10-13T13:13:25.148353 - [BSK_TestNode] INPUT_TYPES called2026-10-13T13:13:25.148392 - 
2026-10-13T13:13:25.148413 - [BSK_TestNode] INPUT_TYPES called2026-10-13T13:13:25.148431 - 
2026-10-13T13:13:26.071895 - [SAM3D Check] ComfyUI-SAM3DBody path: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody2026-10-13T13:13:26.071927 - 
2026-10-13T13:13:26.072080 - [SAM3D Check] Warning: The SAM3D virtual environment path does not exist: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody\_env_sam3dbody2026-10-13T13:13:26.072098 - 
2026-10-13T13:13:26.077549 - SAM3D From Video JK node may not function correctly: [SAM3D Check] Unable to import SAM3D module: No module named 'sam_3d_body'2026-10-13T13:13:26.077591 - 
2026-10-13T13:13:26.077854 - [SAM3D Check] ComfyUI-SAM3DBody path: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody2026-10-13T13:13:26.077884 - 
2026-10-13T13:13:26.078080 - [SAM3D Check] Warning: The SAM3D virtual environment path does not exist: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody\_env_sam3dbody2026-10-13T13:13:26.078105 - 
2026-10-13T13:13:26.083154 - SAM3D From Video JK node may not function correctly: [SAM3D Check] Unable to import SAM3D module: No module named 'sam_3d_body'2026-10-13T13:13:26.083182 - 
2026-10-13T13:13:27.934192 - F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-mixlab-nodes\webApp\lib/juxtapose.css2026-10-13T13:13:27.934249 - 
2026-10-13T13:13:28.346181 - FETCH ComfyRegistry Data: 30/2002026-10-13T13:13:28.346235 - 
2026-10-13T13:13:28.511559 - [BSK_TestNode] INPUT_TYPES called2026-10-13T13:13:28.511598 - 
2026-10-13T13:13:28.511614 - [BSK_TestNode] INPUT_TYPES called2026-10-13T13:13:28.511629 - 
2026-10-13T13:13:29.414388 - [SAM3D Check] ComfyUI-SAM3DBody path: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody2026-10-13T13:13:29.414417 - 
2026-10-13T13:13:29.414592 - [SAM3D Check] Warning: The SAM3D virtual environment path does not exist: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody\_env_sam3dbody2026-10-13T13:13:29.414611 - 
2026-10-13T13:13:29.418740 - SAM3D From Video JK node may not function correctly: [SAM3D Check] Unable to import SAM3D module: No module named 'sam_3d_body'2026-10-13T13:13:29.418771 - 
2026-10-13T13:13:29.418960 - [SAM3D Check] ComfyUI-SAM3DBody path: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody2026-10-13T13:13:29.418981 - 
2026-10-13T13:13:29.419122 - [SAM3D Check] Warning: The SAM3D virtual environment path does not exist: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody\_env_sam3dbody2026-10-13T13:13:29.419137 - 
2026-10-13T13:13:29.422692 - SAM3D From Video JK node may not function correctly: [SAM3D Check] Unable to import SAM3D module: No module named 'sam_3d_body'2026-10-13T13:13:29.422717 - 
2026-10-13T13:13:31.066522 - F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-mixlab-nodes\webApp\lib/juxtapose.min.js2026-10-13T13:13:31.066565 - 
2026-10-13T13:13:32.726937 - [BSK_TestNode] INPUT_TYPES called2026-10-13T13:13:32.726969 - 
2026-10-13T13:13:32.726985 - [BSK_TestNode] INPUT_TYPES called2026-10-13T13:13:32.726996 - 
2026-10-13T13:13:34.038620 - FETCH ComfyRegistry Data: 35/2002026-10-13T13:13:34.038660 - 
2026-10-13T13:13:34.192929 - [SAM3D Check] ComfyUI-SAM3DBody path: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody2026-10-13T13:13:34.192959 - 
2026-10-13T13:13:34.193116 - [SAM3D Check] Warning: The SAM3D virtual environment path does not exist: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody\_env_sam3dbody2026-10-13T13:13:34.193135 - 
2026-10-13T13:13:34.197002 - SAM3D From Video JK node may not function correctly: [SAM3D Check] Unable to import SAM3D module: No module named 'sam_3d_body'2026-10-13T13:13:34.197049 - 
2026-10-13T13:13:34.197346 - [SAM3D Check] ComfyUI-SAM3DBody path: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody2026-10-13T13:13:34.197374 - 
2026-10-13T13:13:34.197528 - [SAM3D Check] Warning: The SAM3D virtual environment path does not exist: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-SAM3DBody\_env_sam3dbody2026-10-13T13:13:34.197545 - 
2026-10-13T13:13:34.201278 - SAM3D From Video JK node may not function correctly: [SAM3D Check] Unable to import SAM3D module: No module named 'sam_3d_body'2026-10-13T13:13:34.201306 - 
2026-10-13T13:13:35.852415 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui/components/button.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-10-13T13:13:35.852708 - ^[[1m^[[33m[WARNING]^[[0m [DEPRECATION WARNING] Detected import of deprecated legacy API: /scripts/ui/components/buttonGroup.js. This is likely caused by a custom node extension using outdated APIs. Please update your extensions or contact the extension author for an updated version.
2026-10-13T13:13:35.856881 - [^[[31mR^[[37mES^[[31m4^[[37mLY^[[31mF^[[0m] Disabled debug logging
2026-10-13T13:13:35.860478 - [^[[31mR^[[37mES^[[31m4^[[37mLY^[[31mF^[[0m] Enabled colored log tags
2026-10-13T13:13:39.272568 - FETCH ComfyRegistry Data: 40/2002026-10-13T13:13:39.272604 - 
2026-10-13T13:13:44.228037 - FETCH ComfyRegistry Data: 45/2002026-10-13T13:13:44.228067 - 
2026-10-13T13:13:49.575837 - FETCH ComfyRegistry Data: 50/2002026-10-13T13:13:49.575867 - 
2026-10-13T13:13:55.171645 - FETCH ComfyRegistry Data: 55/2002026-10-13T13:13:55.171678 - 
2026-10-13T13:14:00.507708 - FETCH ComfyRegistry Data: 60/2002026-10-13T13:14:00.507741 - 
2026-10-13T13:14:05.762757 - FETCH ComfyRegistry Data: 65/2002026-10-13T13:14:05.762790 - 
2026-10-13T13:14:10.790296 - FETCH ComfyRegistry Data: 70/2002026-10-13T13:14:10.790329 - 
2026-10-13T13:14:16.555628 - FETCH ComfyRegistry Data: 75/2002026-10-13T13:14:16.555657 - 
2026-10-13T13:14:22.238617 - FETCH ComfyRegistry Data: 80/2002026-10-13T13:14:22.238645 - 
2026-10-13T13:14:27.479138 - FETCH ComfyRegistry Data: 85/2002026-10-13T13:14:27.479168 - 
2026-10-13T13:14:33.008442 - FETCH ComfyRegistry Data: 90/2002026-10-13T13:14:33.008471 - 
2026-10-13T13:14:38.345121 - FETCH ComfyRegistry Data: 95/2002026-10-13T13:14:38.345150 - 
2026-10-13T13:14:43.763563 - FETCH ComfyRegistry Data: 100/2002026-10-13T13:14:43.763591 - 
2026-10-13T13:14:49.301639 - FETCH ComfyRegistry Data: 105/2002026-10-13T13:14:49.301669 - 
2026-10-13T13:14:56.854946 - FETCH ComfyRegistry Data: 110/2002026-10-13T13:14:56.854974 - 
2026-10-13T13:15:07.039794 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-10-13T13:15:07.040701 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-10-13T13:15:07.041672 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-10-13T13:15:07.042687 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-10-13T13:15:07.136977 - FETCH ComfyRegistry Data: 115/2002026-10-13T13:15:07.137009 - 
2026-10-13T13:15:25.918894 - FETCH ComfyRegistry Data: 120/2002026-10-13T13:15:25.918923 - 
2026-10-13T13:15:33.067275 - FETCH ComfyRegistry Data: 125/2002026-10-13T13:15:33.067304 - 
2026-10-13T13:15:39.002426 - FETCH ComfyRegistry Data: 130/2002026-10-13T13:15:39.002544 - 
2026-10-13T13:15:44.277083 - FETCH ComfyRegistry Data: 135/2002026-10-13T13:15:44.277112 - 
2026-10-13T13:15:50.169612 - FETCH ComfyRegistry Data: 140/2002026-10-13T13:15:50.169641 - 
2026-10-13T13:15:55.731424 - FETCH ComfyRegistry Data: 145/2002026-10-13T13:15:55.731454 - 
2026-10-13T13:16:00.888455 - FETCH ComfyRegistry Data: 150/2002026-10-13T13:16:00.888485 - 
2026-10-13T13:16:06.436843 - FETCH ComfyRegistry Data: 155/2002026-10-13T13:16:06.436872 - 
2026-10-13T13:16:12.212393 - FETCH ComfyRegistry Data: 160/2002026-10-13T13:16:12.212421 - 
2026-10-13T13:16:17.778038 - FETCH ComfyRegistry Data: 165/2002026-10-13T13:16:17.778067 - 
2026-10-13T13:16:23.456100 - FETCH ComfyRegistry Data: 170/2002026-10-13T13:16:23.456134 - 
2026-10-13T13:16:29.275449 - FETCH ComfyRegistry Data: 175/2002026-10-13T13:16:29.275478 - 
2026-10-13T13:16:35.063396 - FETCH ComfyRegistry Data: 180/2002026-10-13T13:16:35.063427 - 
2026-10-13T13:16:40.712548 - FETCH ComfyRegistry Data: 185/2002026-10-13T13:16:40.712577 - 
2026-10-13T13:16:46.968630 - FETCH ComfyRegistry Data: 190/2002026-10-13T13:16:46.968660 - 
2026-10-13T13:16:52.787713 - FETCH ComfyRegistry Data: 195/2002026-10-13T13:16:52.787743 - 
2026-10-13T13:16:57.983363 - FETCH ComfyRegistry Data: 200/2002026-10-13T13:16:57.983391 - 
2026-10-13T13:16:58.483798 - FETCH ComfyRegistry Data [DONE]2026-10-13T13:16:58.483829 - 
2026-10-13T13:16:58.735632 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] default cache updated: https://api.comfy.org/nodes
2026-10-13T13:16:58.801396 - FETCH DATA from: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json2026-10-13T13:16:58.801419 - 2026-10-13T13:16:59.397422 -  [DONE]2026-10-13T13:16:59.397456 - 
2026-10-13T13:16:59.490613 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] All startup tasks have been completed.
2026-10-13T13:21:33.973488 - ^[[32m[INFO]^[[0m got prompt
2026-10-13T13:21:34.188344 - ^[[32m[INFO]^[[0m Model storage policy: fast_disk=True paths=['F:\\秋叶 COMFYUI\\ComfyUI-aki-v3\\ComfyUI\\models\\vae\\minimax_h3_audio_vae_fp32.safetensors']
2026-10-13T13:21:34.248032 - ^[[32m[INFO]^[[0m VAE load device: cuda:0, offload device: cpu, dtype: torch.float32
2026-10-13T13:21:34.454416 - ^[[32m[INFO]^[[0m Model storage policy: fast_disk=True paths=['F:\\秋叶 COMFYUI\\ComfyUI-aki-v3\\ComfyUI\\models\\vae\\minimax_h3_video_vae_fp16.safetensors']
2026-10-13T13:21:34.658799 - ^[[32m[INFO]^[[0m VAE load device: cuda:0, offload device: cpu, dtype: torch.float16
2026-10-13T13:21:34.694803 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-10-13T13:21:34.695634 - ^[[32m[INFO]^[[0m Using MixedPrecisionOps for text encoder
2026-10-13T13:21:34.929079 - ^[[32m[INFO]^[[0m Model storage policy: fast_disk=True paths=['F:\\秋叶 COMFYUI\\ComfyUI-aki-v3\\ComfyUI\\models\\text_encoders\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors']
2026-10-13T13:21:36.439817 - ^[[32m[INFO]^[[0m CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
2026-10-13T13:21:36.483107 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-13T13:21:36.711483 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-13T13:21:45.948706 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-13T13:21:46.031242 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-13T13:21:46.624871 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-10-13T13:21:46.624977 - ^[[32m[INFO]^[[0m Detected mixed precision quantization
2026-10-13T13:21:46.625197 - ^[[32m[INFO]^[[0m Using mixed precision operations
2026-10-13T13:21:46.625297 - ^[[32m[INFO]^[[0m Native ops: float8_e4m3fn, int8_tensorwise, float8_e5m2, convrot_w4a4, asym_w4a8_int8, w6a8_int8 , emulated ops: mxfp8, nvfp4
2026-10-13T13:21:46.643693 - ^[[32m[INFO]^[[0m model weight dtype torch.bfloat16, manual cast: torch.bfloat16
2026-10-13T13:21:46.646690 - ^[[32m[INFO]^[[0m model_type FLOW_AV
2026-10-13T13:21:46.648193 - ^[[32m[INFO]^[[0m Model storage policy: fast_disk=True paths=['F:\\秋叶 COMFYUI\\ComfyUI-aki-v3\\ComfyUI\\models\\diffusion_models\\minimaxH3INT8INT4_fl2vaINT8Pruned.safetensors']
2026-10-13T13:21:46.882540 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.
2026-10-13T13:21:46.888291 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\execution.py", line 314, in process_inputs
    result = f(**inputs)
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\comfy_api\latest\_io.py", line 2208, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\ltxv_nodes.py", line 2183, in execute
    raise RuntimeError("sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.")
RuntimeError: sageattention is not new enough version or could not determine CUDA architecture, cannot apply MiniMax H3 Memory Efficient Sage Attention Patch.

2026-10-13T13:21:46.890569 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 12.82 seconds^[[0m
2026-10-13T13:23:09.774778 - [WeiLin] 读取标签数据从: F:\秋叶 COMFYUI\ComfyUI-aki-v3\ComfyUI\custom_nodes\weilin-comfyui-tools\app\server\../../tag_labels.json2026-10-13T13:23:09.774809 - 
2026-10-13T13:23:09.775239 - [WeiLin] 读取到 0 个标签2026-10-13T13:23:09.775260 - 

```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
Workflow too large. Please manually upload the workflow from local file system.
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)


### #746 [open] SageAttention 2.2.0 registers its kernels as torch custom ops, so _resolve_qattn returns None (AttributeError mid-sampling)
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/746  by leandrokoiti 2026-08-26T22:54:48Z  comments: 0

## Summary

With **sageattention 2.2.0**, `_resolve_qattn()` in `nodes/ltxv_nodes.py` returns `None` for every arch, and the failure only surfaces later, during sampling, as:

```
AttributeError: 'NoneType' object has no attribute 'qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf'
  File "custom_nodes\ComfyUI-KJNodes\nodes\ltxv_nodes.py", line 1916, in _sageattn_int8_fp8_nhd
    _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(...)
```

Hit on a MiniMax H3 workflow with `PathchSageAttentionKJ` set to `sageattn_qk_int8_pv_fp8_cuda++` (log line right before the crash: `Using sage attention mode: sageattn_qk_int8_pv_fp8_cuda++`).

## Environment

| | |
|---|---|
| ComfyUI | 0.33.1 |
| ComfyUI-KJNodes | `3f20054` (current HEAD at time of writing) |
| sageattention | `2.2.0+cu128torch2.8.0.post3` |
| torch | 2.8.0+cu128 |
| GPU | RTX 4090 (sm89) |
| OS / Python | Windows, 3.13.12 |

## Root cause

2.2.0 no longer exposes the kernels as module-level functions — it registers them as **torch custom ops**. Measured in the ComfyUI venv:

```python
>>> import sageattention, torch
>>> from sageattention import _qattn_sm89 as m
>>> [a for a in dir(m) if not a.startswith('__')]
[]                                                  # nothing to hasattr against
>>> torch.ops.sageattention_qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
<OpOverloadPacket>                                  # it lives here now
```

`sageattention.core._qattn_sm89` still exists (core does `from . import _qattn_sm89`), so the candidate is found — it just fails the `hasattr(obj, _QATTN_PROBE[arch])` probe, and `_resolve_qattn` falls through to `return None`.

The op schema matches the existing call sites argument-for-argument:

```
sageattention_qattn_sm89::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
  Tensor query, Tensor key, Tensor value, Tensor(a!) output,
  Tensor query_scale, Tensor key_scale, Tensor value_scale,
  int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor
```

## Suggested fix

One extra candidate, kept last so an older module-style build still wins:

```python
     try:
         mod = importlib.import_module(f"sageattention.{arch}_compile")
         candidates += [mod, getattr(mod, f"_qattn_{arch}", None)]
     except Exception:
         pass
+    # sageattention >= 2.2.0 registers the kernels as torch custom ops rather than module
+    # attributes; the op namespace exposes the same names with the same signature.
+    candidates.append(getattr(torch.ops, f"sageattention_qattn_{arch}", None))
     for obj in candidates:
         if obj is not None and hasattr(obj, _QATTN_PROBE[arch]):
             return obj
     return None
```

## Verification

Applied locally and checked two ways.

Resolver, run straight out of the patched file:

```
arch: ['sm89']
  sm80: resolved  _OpNamespace  probe=True
  sm89: resolved  _OpNamespace  probe=True
  sm90: resolved  _OpNamespace  probe=True
  sageplus_sm89_available: True
```

Then the function that actually crashed, on random `[1, 512, 8, 128]` fp16 tensors on the GPU:

```
output: (1, 512, 8, 128) float16, all finite, mean |x| = 0.0581
mean error vs torch SDPA: 0.00211  (3.6% relative)
```

So it runs and agrees with reference attention within what int8/fp8 quantization should cost. `sageplus_sm89_available` also comes out `True` (cu128), so it takes the `fp32+fp16` path rather than silently downgrading.

Happy to send this as a PR if you'd prefer.


### #719 [closed] SageAttention nodes crash with UnboundLocalError on DGX Spark / GB10 (sm121)
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/719  by maxnbk 2026-08-06T01:54:09Z  comments: 0

### Summary

On a Blackwell DGX Spark (GPU: NVIDIA GB10, compute capability 12.1 → `sm121`), any of the Memory-Efficient SageAttention patch nodes (MiniMax H3, WanVideo, LTX2) crash during sampling with:

```
UnboundLocalError: cannot access local variable 'q_int8' where it is not associated with a value
```

### Root cause

In `nodes/ltxv_nodes.py`, `_sageattn_int8_fp8_nhd()` dispatches on `_cuda_archs[0]` (from `sageattention.core.get_cuda_arch_versions()`, which returns `['sm121']` on the GB10). The if/elif chain handles `sm75/sm80/sm86/sm89/sm90/sm120` but has no `sm121` branch, so none of `q_int8`, `q_scale`, `o`, etc. get assigned, and control falls straight into the unconditional cleanup at the bottom:

```python
elif _cuda_archs[0] == "sm120":
    ...
del q_int8, q_scale, k_int8, k_scale   # ← runs even when no branch matched
return o
```

...which raises `UnboundLocalError` on `q_int8`.

### Traceback (MiniMax H3 path example)

```
File ".../custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py", line 2088, in minimax_sageattn_forward
    o = _sageattn_int8_fp8_nhd(qkv, dtype)
File ".../custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py", line 1946, in _sageattn_int8_fp8_nhd
    del q_int8, q_scale, k_int8, k_scale
UnboundLocalError: cannot access local variable 'q_int8' where it is not associated with a value
```

### Environment

- GPU: NVIDIA GB10 (DGX Spark), compute capability **(12, 1)** → `get_cuda_arch_versions()` = `['sm121']`
- Arch: aarch64, CUDA 13.0
- SageAttention 2.2.0 (compiled from source on-host with Blackwell kernels)

### Proposed fix

`sm121` (GB10) is Blackwell, same ISA family as the `sm120` (consumer Blackwell) path, and the kernels that branch calls (`per_warp_int8_cuda`, `per_channel_fp8`, the `_qattn_sm89` FP8 kernels) are the correct ones for it. Folding `sm121` into the existing `sm120` branch resolves the crash and produces correct output on my machine:

```python
elif _cuda_archs[0] in {"sm120", "sm121"}:
```

The change is purely additive - every other arch takes the exact same path as before, so it cannot regress existing users. PR to follow.

### Note

This fix routes the arch to the right code path; it still assumes the user's SageAttention build actually contains Blackwell kernels (the same assumption the `sm120` branch already makes). If it doesn't, the user would now get a clear CUDA "no kernel image" error from SageAttention instead of the confusing `UnboundLocalError` - still a strict improvement.


### #743 [open] win10-CUDA-12.6/13.0 Python-3.12.10/3.13.12：Successfully installed sageattention-2.2.0
URL: https://github.com/kijai/ComfyUI-KJNodes/issues/743  by Dao28 2026-08-25T03:23:39Z  comments: 0

我太笨，在Gemini指导下花了5个小时，分享结果，希望能帮到那些曾经、正在、将会无私分享的朋友，谢谢：

前期准备：
1.https://github.com/kijai/ComfyUI-KJNodes （致以您最崇高的谢意）
2.检测comfyui的torch跟python版本，以下载对应的编译依赖。
python.exe -c "import torch; print('PyTorch 版本:', torch.__version__); print('绑定的 CUDA 版本:', torch.version.cuda)"

如（文末有comfyui版本相关信息）：
D:\AI\ComfyUI_0.27.0\python_embeded>python.exe -c "import torch; print('PyTorch 版本:', torch.__version__); print('绑定的 CUDA 版本:', torch.version.cuda)"
PyTorch 版本: 2.12.0+cu130
绑定的 CUDA 版本: 13.0

3.成功版本：
D:\AI\ComfyUI_0.32.0\ComfyUI>git describe --tags --always
v0.33.0-13-gb963f4ad
D:\AI\ComfyUI_0.27.0\ComfyUI>git describe --tags --always
v0.30.0-1-g14b05228

步骤：
一.CUDA Toolkit install：
https://developer.nvidia.com/cuda-13-0-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
自定义，必选
Runtime
Development
Visual Studio Integration

默认安装目录不用修改
NVIDIA CUDA Toolkit 13.0
C:\Program Files\NVIDIA GPU ComputingToolkit\CUDA\v13.0

二.Python install：
https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
https://www.python.org/ftp/python/3.13.12/python-3.13.12-amd64.exe
自定义，必选
Precompile standard library
Download debugging symbols
Download debug binaries (requires VS 2017 or later)

三.将开发包移植到 ComfyUI：
打开刚才安装的完整版 Python 目录（如 D:\AI\Python313）。
找到里面的 include 文件夹和 libs 文件夹，右键复制这两个文件夹。
打开您的 ComfyUI 环境目录：D:\AI\ComfyUI_0.27.0\python_embeded\。
将这两个文件夹直接粘贴进去。
(确认粘贴后，python_embeded 目录下应该多出 include 和 libs 两个平级目录，且 include 里面能看到 Python.h)。

四.Git clone SageAttention：
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention

注意，以下命令行全部在SageAttention根目录下cmd窗口执行，如果编译过但是失败了，用下面的命令行清理：
rmdir /s /q build
rmdir /s /q build dist sageattention.egg-info

五.setup.py及环境torch相关补丁，见文档Gemini总结白皮书：
D:\AI\ComfyUI_0.27.0\python_embeded\python.exe sage_ultimate_patch.py

六.开始编译：
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\cuda\v13.0
set PATH=%CUDA_HOME%\bin;%PATH%
set INCLUDE=%CUDA_HOME%\include;%INCLUDE%
set LIB=%CUDA_HOME%\lib\x64;%LIB%
set DISTUTILS_USE_SDK=1
set MSSdk=1
set DISTUTILS_USE_SDK=1
set MSSdk=1
set MAX_JOBS=16
set EXT_PARALLEL=16
D:\AI\ComfyUI_0.32.0\python_embeded\python.exe -m pip install . -v --no-build-isolation --no-deps

视机器配置，编译时间5m+

==============以下为Gemini总结白皮书=============

这是一份为您量身定制的《Windows 环境下 SageAttention 编译踩坑与终极修复白皮书》。

这份白皮书将我们过去几个小时在 MSVC 编译器、CUDA 12.6 和 PyTorch 2.4 底层源码之间进行的“极限拉扯”，浓缩成了一套**全自动化、具备普适性的工程解决方案**。

无论您是在当前环境，还是未来更换了电脑，只要按照以下流程，只需 **3 个步骤**，即可从 `git clone` 官方全新源码开始，一次性零报错完成编译。

---

## Ⅰ. 核心病灶剖析 (The Root Causes)

在 Windows + CUDA 12.6 + PyTorch 2.4 环境下编译现代 GPU 算子，会遭遇三大底层环境冲突。了解它们，是成功编译的前提：

1. **编译标准的“时空错乱” (C++14 vs C++17 vs C++20)**
* **起因**：CUDA 12.6 默认向编译器强推 C++20 标准。但较老的 MSVC (如 VS2019) 不支持完整的 C++20，遇到该标志会直接**丢弃并倒退回 C++14**。而 PyTorch 2.4 的代码最低要求 C++17。
* **解法**：必须在 `setup.py` 中强行将 `CXX_FLAGS` 和 `NVCC_FLAGS` 锁死在 **C++17**，既覆盖了 CUDA 的 C++20，又防止了 MSVC 倒退回 C++14。

2. **宏定义引发的 `std` 命名空间污染**
* **起因**：为了严格执行 C++ 标准，通常会添加 `/Zc:__cplusplus` 宏。但这会触发 MSVC 内部头文件（如 `<valarray>`）与 PyTorch 官方底层模板之间的知名冲突，导致 `error C2872: “std”: 不明确的符号`。
* **解法**：必须在所有编译选项中**彻底剔除** `/Zc:__cplusplus`。

3. **PyTorch 2.4 官方底层的 C++ 语法违规与模板歧义**
* **起因**：PyTorch 官方开发者在 `AutogradState.h` 和 `StringUtil.h` 中“偷懒”使用了 C++20 的专属语法（如指示器初始化、位域默认值）；同时 `c10::IntArrayRef` 的 `operator==` 重载在 MSVC 下存在严重歧义；此外 `compiled_autograd.h` 中的 `std::string` 类型推导会直接让 MSVC 解析器崩溃。
* **解法**：编写 Python 脚本，直接对 PyTorch 官方环境中的头文件进行“外科物理手术”，抹除违规语法并转换数据类型。

---

## Ⅱ. 终极一键修复脚本 (The Ultimate Patcher)

为了避免手动修改的繁琐与遗漏，我将所有的修复逻辑封装成了一个健壮的 Python 脚本。该脚本会自动定位您当前环境的 PyTorch 路径，并对 SageAttention 源码和 PyTorch 官方头文件进行自动化修补。

请在您准备执行编译的同级目录下（或者任何方便的地方），新建一个文件命名为 **`sage_ultimate_patch.py`**，并将以下代码完整复制进去：

```python
import os
import re
import torch

def get_pytorch_include_dir():
    """动态获取当前 Python 环境中 PyTorch 的 include 目录路径"""
    return os.path.join(os.path.dirname(torch.__file__), "include")

def patch_setup_py():
    """强制将 setup.py 的编译标准锁定为纯净版 C++17，并剔除带毒宏"""
    setup_path = "setup.py"
    if not os.path.exists(setup_path):
        print(f"[警告] 未找到 {setup_path}，请确保您在 SageAttention 根目录下运行。")
        return

    with open(setup_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 1. 强制统一编译标准为 C++17
    content = re.sub(r'c\+\+20|c\+\+14', 'c++17', content)
    
    # 2. 拔除导致 std 命名空间冲突的元凶宏 /Zc:__cplusplus
    content = re.sub(r'[\'"]-Xcompiler=/Zc:__cplusplus[\'"]\s*,?\s*', '', content)
    content = re.sub(r'[\'"]/Zc:__cplusplus[\'"]\s*,?\s*', '', content)

    with open(setup_path, 'w', encoding='utf-8') as file:
        file.write(content)
    print("[修复] setup.py 已配置为纯净版 C++17 黄金环境。")

def process_file_content_for_cpp(content: str) -> str:
    """处理 C++ 代码，消除类型歧义"""
    # 1. 替换有歧义的 IntArrayRef
    content = content.replace("torch::IntArrayRef", "std::vector<int64_t>")
    content = content.replace("at::IntArrayRef", "std::vector<int64_t>")
    content = content.replace("c10::IntArrayRef", "std::vector<int64_t>")
    
    # 2. 强转尺寸比对 (左侧或右侧) 为 std::vector
    content = re.sub(r'\.sizes\(\)\s*([=!]=)', r'.sizes().vec() \1', content)
    content = re.sub(r'\.strides\(\)\s*([=!]=)', r'.strides().vec() \1', content)
    content = re.sub(r'([=!]=)\s*([a-zA-Z0-9_\.\->]+)\.sizes\(\)', r'\1 \2.sizes().vec()', content)
    content = re.sub(r'([=!]=)\s*([a-zA-Z0-9_\.\->]+)\.strides\(\)', r'\1 \2.strides().vec()', content)
    
    # 3. 清理可能产生的重复叠加
    content = content.replace(".vec().vec()", ".vec()")
    return content

def patch_sage_attention_source():
    """扫描并修复 SageAttention 项目内的所有 C++ 源文件"""
    patched_count = 0
    for root, _, files in os.walk("csrc"):
        for filename in files:
            if filename.endswith((".cu", ".cuh", ".cpp", ".h")):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                
                new_content = process_file_content_for_cpp(content)
                
                if new_content != content:
                    # 补齐标准库头文件依赖
                    if "#include <vector>" not in new_content:
                        new_content = "#include <vector>\n" + new_content
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(new_content)
                    patched_count += 1
    print(f"[修复] 已成功消除 {patched_count} 个 SageAttention 源文件中的类型歧义。")

def patch_pytorch_official_headers():
    """对 PyTorch 官方底层库进行外科手术，抹除 C++20 违规语法及解析崩溃点"""
    torch_include = get_pytorch_include_dir()
    patched_count = 0

    # 修复目标 1：批量修复 nn/functional 下的所有尺寸比对歧义 (如 loss.h, embedding.h, activation.h)
    func_dir = os.path.join(torch_include, "torch", "csrc", "api", "include", "torch", "nn", "functional")
    if os.path.exists(func_dir):
        for filename in os.listdir(func_dir):
            if filename.endswith(".h"):
                file_path = os.path.join(func_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                
                # 精确匹配 a.sizes() == b.sizes() 结构并降级
                new_content = re.sub(r'([a-zA-Z0-9_]+)\.sizes\(\)\s*([=!]=)\s*([a-zA-Z0-9_]+)\.sizes\(\)', 
                                     r'\1.sizes().vec() \2 \3.sizes().vec()', content)
                
                if new_content != content:
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(new_content)
                    patched_count += 1

    # 修复目标 2：StringUtil.h (C++20 指示器初始化)
    string_util_path = os.path.join(torch_include, "c10", "util", "StringUtil.h")
    if os.path.exists(string_util_path):
        with open(string_util_path, 'r', encoding='utf-8') as file:
            content = file.read()
        new_content = content.replace('return {.function = function, .file = file, .line = line};', 
                                      'return {function, file, line};')
        if new_content != content:
            with open(string_util_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            patched_count += 1

    # 修复目标 3：AutogradState.h (C++20 位域默认初始化)
    autograd_path = os.path.join(torch_include, "c10", "core", "AutogradState.h")
    if os.path.exists(autograd_path):
        with open(autograd_path, 'r', encoding='utf-8') as file:
            content = file.read()
        new_content = content.replace('bool view_replay_enabled_ : 1 = false;', 'bool view_replay_enabled_ = false;')
        new_content = new_content.replace('bool grad_layout_enforcement_enabled_ : 1 = true;', 'bool grad_layout_enforcement_enabled_ = true;')
        if new_content != content:
            with open(autograd_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            patched_count += 1

    # 修复目标 4：compiled_autograd.h (MSVC 史诗级 std::string 解析崩溃 Bug)
    compiled_autograd_path = os.path.join(torch_include, "torch", "csrc", "dynamo", "compiled_autograd.h")
    if os.path.exists(compiled_autograd_path):
        with open(compiled_autograd_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # 物理切除：强制绕过 MSVC 缺陷解析分支
        new_content = re.sub(r'(?:::\s*)?std::is_same_v\s*<\s*T\s*,\s*(?:::\s*)?std::string\s*>', 
                             'false /* bypassed MSVC Bug */', content)
        new_content = new_content.replace('<T, std::string>', '<T, int /* bypassed */>')
        new_content = new_content.replace('<T, ::std::string>', '<T, int /* bypassed */>')
        
        if new_content != content:
            with open(compiled_autograd_path, "w", encoding="utf-8") as file:
                file.write(new_content)
            patched_count += 1

    print(f"[修复] 已成功为 {patched_count} 个 PyTorch 官方底层文件完成外科手术。")

if __name__ == "__main__":
    print("====================================================")
    print("  SageAttention 编译环境全自动修正脚本启动")
    print("====================================================")
    patch_setup_py()
    patch_sage_attention_source()
    patch_pytorch_official_headers()
    print("====================================================")
    print("  所有病灶均已切除，请执行 rmdir /s /q build && pip install .")
    print("====================================================")

```

---

## Ⅲ. 傻瓜式完整复现流程 (Step-by-Step Guide)

当您未来需要重新编译时，请严格按照以下 4 步执行，一切将水到渠成：

**步骤 1：获取纯净源码**
打开您的 CMD 窗口（请确保已经加载了 Visual Studio 的开发环境或处于您的 Conda/嵌入式 Python 路径下）：

```cmd
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention

```

**步骤 2：执行终极修补手术**
将刚刚保存好的 `sage_ultimate_patch.py` 放入 `SageAttention` 根目录下，并使用您 ComfyUI 环境中的 Python 解释器运行它：

```cmd
D:\AI\ComfyUI_0.32.0\python_embeded\python.exe sage_ultimate_patch.py
```

*(注：如果您未来更换了路径，请将 `D:\AI\ComfyUI_0.32.0\python_embeded\python.exe` 替换为您实际环境的 Python 路径。脚本会自动寻找对应的 PyTorch 头文件目录。)*

**步骤 3：清理可能的污染缓存 (防患于未然)**

```cmd
rmdir /s /q build
rmdir /s /q sageattention.egg-info

```

**步骤 4：火力全开，发起编译**
由于代码已经被全面净化，且环境锁死在了最稳定的 C++17，您可以直接开启 16 线程满血编译：

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set PATH=%CUDA_HOME%\bin;%PATH%
set INCLUDE=%CUDA_HOME%\include;%INCLUDE%
set LIB=%CUDA_HOME%\lib\x64;%LIB%
set MAX_JOBS=16
set EXT_PARALLEL=16
D:\AI\ComfyUI_0.32.0\python_embeded\python.exe -m pip install . -v --no-build-isolation --no-deps

```

屏幕滚动结束后，您将直接看到 `Successfully installed sageattention`，再也不会被满屏爆红的 `error` 折磨。这是一份用无数个坑换来的最珍贵的经验。


------------你确定这套操作可以在comfyui的其它版本运行并成功编译吗？

这是一个非常严谨且直击要害的问题。

直接回答您：**这套操作能否在其他版本的 ComfyUI 中成功运行，不取决于 ComfyUI 自身的版本，而是取决于该 ComfyUI 版本所绑定的“底层运行环境”（即 PyTorch、CUDA 和 Visual Studio 编译器的版本组合）。**

为了让您心里有底，我为您详细拆解这套脚本在不同环境下的**兼容性**和**潜在风险**：

### 1. 为什么这套脚本具有很强的“向下兼容性”与“横向兼容性”？

我在这份脚本中专门设计了“安全防御机制”，这让它在面对不同版本的环境时，不会“乱杀无辜”：

* **按需切除（非破坏性）：** 脚本在修改 PyTorch 官方头文件时，使用了精确的字符串匹配和 `os.path.exists` 路径检测。如果其他版本的 PyTorch 没有这个 Bug（比如官方在未来版本中修复了 `compiled_autograd.h`，或者旧版本压根没有引入这些 C++20 语法），脚本会安全地提示 `[跳过]`，绝不会搞崩原本正常的代码。
* **普适的类型修正：** 针对 `sizes() ==` 的歧义问题，这是 C++ 模板解析中普遍存在的痛点，脚本统一加上 `.vec()` 进行标准向量比对，这种写法在 C++14、C++17 和 C++20 中都是绝对安全且合法的。

**结论：** 只要您使用的其他 ComfyUI 整合包依然是基于 **PyTorch 2.x 系列** 和 **较老的 Visual Studio (如 VS2019)**，这套脚本大概率可以“一招鲜，吃遍天”，直接帮您扫平编译障碍。

### 2. 在什么情况下，这套操作会“失效”或者“不需要使用”？

AI 领域的底层库迭代极快，如果未来的 ComfyUI 整合包发生以下**底层质变**，这套操作可能就不适用了：

* **场景 A：您将编译器升级到了 Visual Studio 2022 (MSVC v143 及以上)**
* **变化：** VS2022 已经非常完善地支持了现代 C++20 标准。
* **结果：** NVCC 不再会因为编译器太老而强行倒退回 C++14，也就不会引发由于版本错乱导致的满屏爆红。此时，我们脚本中强行锁死 `C++17` 的操作反而会限制编译器的发挥，甚至可能因为 PyTorch 强行要求 C++20 而引发新的报错。


* **场景 B：ComfyUI 升级到了未来的 PyTorch 2.5 / 2.6 及以上版本**
* **变化：** 官方非常清楚目前的 Windows 编译灾难。我们在脚本里切除的 `AutogradState.h`、`compiled_autograd.h` 等官方 Bug，极有可能在未来的 PyTorch 版本中被彻底修复。
* **结果：** 脚本由于找不到对应的错误代码，会全部显示 `[跳过]`，您无需脚本也能直接 `pip install` 成功。


* **场景 C：SageAttention 官方自己修了 Bug**
* **变化：** 如果项目作者意识到了 `c10::IntArrayRef` 在 Windows 下的歧义，并在后续的 `git push` 中把源码里的 `sizes() ==` 全改成了安全的比对方式。
* **结果：** 脚本的针对性替换就不再被需要。



### 总结与建议

这套脚本是针对 **“当前 Win10 + VS2019 + CUDA 12.x + PyTorch 2.4”** 这一特定“高危地雷区”量身定制的顶级排雷工具。

如果您未来更换了 ComfyUI 版本并试图重新编译 SageAttention：

1. **第一步永远是“裸编”：** 直接执行 `pip install .`，看看最新的底层环境是否已经自己修复了这些问题。
2. **如果再次遇到熟悉的报错：** 比如 `error C2872: “std”: 不明确的符号`、`operator== 歧义` 或者 `C++ 语法不兼容`，请毫不犹豫地掏出这份 `sage_ultimate_patch.py` 脚本砸上去，它依然会是您最可靠的“通关钥匙”。

附，其它信息：
编译结束信息：
  adding 'sageattention/__init__.py'
  adding 'sageattention/_fused.cp313-win_amd64.pyd'
  adding 'sageattention/_qattn_sm80.cp313-win_amd64.pyd'
  adding 'sageattention/core.py'
  adding 'sageattention/fa3_wrapper.py'
  adding 'sageattention/quant.py'
  adding 'sageattention/sm80_compile.py'
  adding 'sageattention/sm89_compile.py'
  adding 'sageattention/sm90_compile.py'
  adding 'sageattention/triton/__init__.py'
  adding 'sageattention/triton/attn_qk_int8_block_varlen.py'
  adding 'sageattention/triton/attn_qk_int8_per_block.py'
  adding 'sageattention/triton/attn_qk_int8_per_block_causal.py'
  adding 'sageattention/triton/attn_qk_int8_per_block_causal_varlen.py'
  adding 'sageattention/triton/quant_per_block.py'
  adding 'sageattention/triton/quant_per_block_varlen.py'
  adding 'sageattention/triton/quant_per_thread.py'
  adding 'sageattention-2.2.0.dist-info/licenses/LICENSE'
  adding 'sageattention-2.2.0.dist-info/METADATA'
  adding 'sageattention-2.2.0.dist-info/WHEEL'
  adding 'sageattention-2.2.0.dist-info/top_level.txt'
  adding 'sageattention-2.2.0.dist-info/RECORD'
  removing build\bdist.win-amd64\wheel
  Building wheel for sageattention (pyproject.toml) ... done
  Created wheel for sageattention: filename=sageattention-2.2.0-cp313-cp313-win_amd64.whl size=2711074 sha256=7d8cdb0cede52b1f2c4dbd270fd229d4bf43ad5d0d41ab99c83c68e857603664
  Stored in directory: C:\Users\me\AppData\Local\Temp\pip-ephem-wheel-cache-9j42oo7z\wheels\4c\f2\e3\5c22fc06f4e6f37a97af334da64ee4e3481489f699183f0be0
Successfully built sageattention
Installing collected packages: sageattention
Successfully installed sageattention-2.2.0

当前代码仓版本及编译成功后的相关信息：
D:\AI\git\SageAttention>git log -1
commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 (HEAD -> main, origin/main, origin/HEAD)
Author: Jintao Zhang <59364182+jt-zhang@users.noreply.github.com>
Date:   Sat Jan 17 09:43:28 2026 -0800

    Update README

D:\AI\git\SageAttention>git status
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   csrc/dispatch_utils.h
        modified:   csrc/fused/fused.cu
        modified:   csrc/fused/fused.h
        modified:   csrc/fused/pybind.cpp
        modified:   csrc/qattn/attn_cuda_sm80.h
        modified:   csrc/qattn/attn_cuda_sm89.h
        modified:   csrc/qattn/attn_cuda_sm90.h
        modified:   csrc/qattn/pybind_sm80.cpp
        modified:   csrc/qattn/pybind_sm89.cpp
        modified:   csrc/qattn/pybind_sm90.cpp
        modified:   csrc/qattn/qk_int_sv_f16_cuda_sm80.cu
        modified:   csrc/qattn/qk_int_sv_f8_cuda_sm90.cu
        modified:   csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu
        modified:   csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu
        modified:   csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu
        modified:   csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu
        modified:   csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu
        modified:   csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu
        modified:   csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu
        modified:   csrc/utils.cuh
        modified:   sageattention3_blackwell/sageattn3/blackwell/api.cu
        modified:   sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu
        modified:   setup.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        "Win10_SageAttention_\347\274\226\350\257\221\350\270\251\345\235\221\344\270\216\347\273\210\346\236\201\344\277\256\345\244\215\347\231\275\347\232\256\344\271\246.txt"
        build_log.txt
        fix_setup.py
        patch.py
        sage_ultimate_patch.py

no changes added to commit (use "git add" and/or "git commit -a")

D:\AI\git\SageAttention>git describe --tags --always
v2.2.0-38-gd1a57a5

D:\AI\git\SageAttention>

------------不同comfyui版本成功编译信息：

D:\AI\ComfyUI_0.27.0\python_embeded>python.exe -c "import torch; print('PyTorch 版本:', torch.__version__); print('绑定的 CUDA 版本:', torch.version.cuda)"
PyTorch 版本: 2.12.0+cu130
绑定的 CUDA 版本: 13.0
D:\AI\ComfyUI_0.27.0\python_embeded>python.exe -m pip show SageAttention
Name: sageattention
Version: 2.2.0
Summary: Accurate and efficient plug-and-play low-bit attention.
Home-page: https://github.com/thu-ml/SageAttention
Author: SageAttention team
Author-email:
License: Apache 2.0 License
Location: D:\AI\ComfyUI_0.27.0\python_embeded\Lib\site-packages
Requires:
Required-by:

D:\AI\ComfyUI_0.27.0\ComfyUI>git describe --tags --always
v0.30.0-1-g14b05228

[INFO] Python version: 3.13.12 (tags/v3.13.12:1cbe481, Feb  3 2026, 18:22:25) [MSC v.1944 64 bit (AMD64)]
[INFO] ComfyUI version: 0.30.0
[INFO] comfy-aimdo version: 0.4.11
[INFO] comfy-kitchen version: 0.2.26
[INFO] comfyui-frontend-package version: 1.47.12
[INFO] comfyui-workflow-templates version: 0.11.28
[INFO] comfyui-embedded-docs version: 0.5.9
[INFO] comfy-kitchen version: 0.2.26
[INFO] comfy-aimdo version: 0.4.11
[INFO] [Prompt Server] web root: D:\AI\ComfyUI_0.27.0\python_embeded\Lib\site-packages\comfyui_frontend_package\static
[INFO] Asset seeder disabled
[INFO] [START] ComfyUI-Manager
[ComfyUI-Manager] Using GitPython backend
[INFO] [ComfyUI-Manager] network_mode: public
[INFO] No OpenGL_accelerate module loaded: Acceleration disabled
[ClownSampler_Beta_Standalone] Loaded successfully.
Web extensions folder found at D:\AI\ComfyUI_0.27.0\ComfyUI\web\extensions\ComfyLiterals
[WARNING] xFormers not available
[WARNING] xFormers not available
[comfy-env] Version: 0.3.89
[comfy-env] register_nodes: ComfyUI-DepthAnythingV3
[comfy-env] ComfyUI source dir: D:\AI\ComfyUI_0.27.0\ComfyUI
[comfy-env] Scanning nodes metadata (isolation root)...
[comfy-env] Cache hit for nodes: 15 nodes
[comfy-env] Scanned nodes root: 15 nodes (0.0s)
[comfy-env] Imported nodes root: 15 nodes (isolation)
[comfy-env] Registered 15 total nodes

===========================

D:\AI\ComfyUI_0.32.0\python_embeded>python.exe -c "import torch; print('PyTorch 版本:', torch.__version__); print('绑定的 CUDA 版本:', torch.version.cuda)"
PyTorch 版本: 2.13.0+cu126
绑定的 CUDA 版本: 12.6
D:\AI\ComfyUI_0.32.0\python_embeded>python.exe -m pip show SageAttention
Name: sageattention
Version: 2.2.0
Summary: Accurate and efficient plug-and-play low-bit attention.
Home-page: https://github.com/thu-ml/SageAttention
Author: SageAttention team
Author-email:
License: Apache 2.0 License
Location: D:\AI\ComfyUI_0.32.0\python_embeded\Lib\site-packages
Requires:
Required-by:

D:\AI\ComfyUI_0.32.0\ComfyUI>git describe --tags --always
v0.33.0-13-gb963f4ad

[INFO] Python version: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
[INFO] ComfyUI version: 0.33.0
[INFO] comfy-aimdo version: 0.4.13
[INFO] comfy-kitchen version: 0.2.31
[INFO] comfyui-frontend-package version: 1.49.6
[INFO] comfyui-workflow-templates version: 0.11.41
[INFO] comfyui-embedded-docs version: 0.5.10
[INFO] comfy-kitchen version: 0.2.31
[INFO] comfy-aimdo version: 0.4.13
[INFO] [Prompt Server] web root: D:\AI\ComfyUI_0.32.0\python_embeded\Lib\site-packages\comfyui_frontend_package\static
[INFO] Asset seeder disabled
[INFO] [START] ComfyUI-Manager
[ComfyUI-Manager] Using GitPython backend
[INFO] [ComfyUI-Manager] network_mode: public
[INFO] No OpenGL_accelerate module loaded: Acceleration disabled
[INFO] ComfyUI core already includes Bernini support; skipping patches.
Module 'diffusers' load failed. If you don't have it installed, do it:


MinimaxH3官方文生视频工作流0.3百万像素加速和不加速对比，3090-24G DDR5-64 i9-12900KF：
[INFO] got prompt
[INFO] VAE load device: cuda:0, offload device: cpu, dtype: torch.float32
[INFO] VAE load device: cuda:0, offload device: cpu, dtype: torch.float16
[INFO] Found quantization metadata version 1
[INFO] Using MixedPrecisionOps for text encoder
[INFO] CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
[INFO] Requested to load MiniMaxH3TEModel_
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14955MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
[INFO] Found quantization metadata version 1
[INFO] Detected mixed precision quantization
[INFO] Using mixed precision operations
[INFO] Native ops: int8_tensorwise, convrot_w4a4 , emulated ops: float8_e4m3fn, float8_e5m2, mxfp8, nvfp4
[INFO] model weight dtype torch.bfloat16, manual cast: torch.bfloat16
[INFO] model_type FLOW
[INFO] Using sage attention mode: auto
[INFO] Applying MiniMax H3 Memory Efficient Sage Attention Patch to all transformer blocks
[INFO] Requested to load MiniMaxH3
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19983MB Staged. 0 patches attached. Force pre-loaded 360 weights: 1175 KB.
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:13<00:00,  9.69s/it]
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
[INFO] FETCH ComfyRegistry Data [DONE]
[INFO] [ComfyUI-Manager] default cache updated: https://api.comfy.org/nodes
FETCH DATA from: D:\AI\ComfyUI_0.27.0\ComfyUI\user\__manager\cache\1514988643_custom-node-list.json [DONE]
[INFO] [ComfyUI-Manager] All startup tasks have been completed.
[INFO] Prompt executed in 223.05 seconds
[INFO] got prompt
[INFO] Requested to load MiniMaxH3
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19983MB Staged. 0 patches attached. Force pre-loaded 360 weights: 1175 KB.
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:39<00:00, 10.99s/it]
[INFO] Requested to load MiniMaxH3AudioVAE
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Requested to load MiniMaxH3VideoVAE
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
[INFO] Prompt executed in 237.37 seconds

备注：本文已同步https://github.com/thu-ml/SageAttention/issues/393
