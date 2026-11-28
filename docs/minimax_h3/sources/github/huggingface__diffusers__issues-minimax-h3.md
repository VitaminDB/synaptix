# huggingface__diffusers__issues-minimax-h3

Source: https://github.com/huggingface/diffusers/issues?q=minimax+h3
Fetched: 2026-11-28



---

# ISSUES in huggingface/diffusers matching "H3" (top 25 by comments)


### #11816 [closed] set_adapters performance degrades with the number of inactive adapters
URL: https://github.com/huggingface/diffusers/issues/11816  by hrazjan 2025-06-26T22:27:54Z  comments: 27

### Describe the bug

### Goal
Build an image-generation service with `StableDiffusionXLPipeline` that:

1. Keeps ~50 LoRA adapters resident in GPU VRAM.
2. For each request:
   • activate **≤ 5** specific LoRAs via `pipeline.set_adapters(...)`  
   • run inference  
   • deactivate them (ready for the next request).

### Issue
`pipeline.set_adapters()` becomes progressively slower the more unique LoRAs have ever been loaded,
even though each call still enables only up to five adapters.

| # LoRAs ever loaded | `set_adapters()` time (s) | 
|---------------------|---------------------------|
| 3                   | ~ 0.1031                      | 
| 6                  | ~ 0.1843                     | 
| 9                  | ~ 0.2614                     |
| 12                  | ~ 0.3522                     | 
| 45                  | ~ 1.2470                     | 
| 57                  | ~ 1.5435                     |

### What I’ve tried
1. **Load LoRAs from disk for every request** ~ 0.8 s/LoRA, too slow.  
2. **Keep LoRAs in RAM (`SpooledTemporaryFile`) + `pipeline.delete_adapter()`** – roughly as slow as (1).  
3. **Keep all 50 LoRAs on the GPU** and just switch with `set_adapters()` – fastest so far, but still shows the O(N)-style growth above.

### Question
Is this increasing latency expected?  
Is there a recommended pattern for caching many LoRAs on the GPU and switching between small subsets without paying an O(total LoRAs) cost every time?

Any guidance (or confirmation it’s a current limitation) would be greatly appreciated!

### Reproduction

<details>
<summary>Code</summary>

``` Minimal example
import os
import time
from typing import List
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
import torch
from diffusers.utils import logging
logging.disable_progress_bar()
logging.set_verbosity_error() 

pipeline = None

class Lora(BaseModel):
    name: str
    strength: float

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        print(f"{func.__name__} executed in {duration:.4f} seconds")
        return result
    return wrapper

@timeit
def load_model():
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        vae=AutoencoderTiny.from_pretrained(
            'madebyollin/taesdxl',
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    return pipeline

@timeit
def set_adapters(pipeline, adapter_names, adapter_weights):
    pipeline.set_adapters(
        adapter_names=adapter_names,
        adapter_weights=adapter_weights,
    )

@timeit
def fuse_lora(pipeline):
    pipeline.fuse_lora()

@timeit
def inference(pipeline, req, generator=None):
    return pipeline(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        generator=generator,
    ).images

def apply_loras(pipeline, loras: list[Lora]) -> str:
    if not loras or len(loras) == 0:
        pipeline.disable_lora()
        return
    
    pipeline.enable_lora()
    for lora in loras:
        try:
            pipeline.load_lora_weights(
                "ostris/super-cereal-sdxl-lora",
                weight_name="cereal_box_sdxl_v1.safetensors",
                adapter_name=lora.name,
                token=os.getenv("HUGGINGFACE_HUB_TOKEN", None),
            )
        except ValueError:
            continue # LoRA already loaded, skip
        except Exception as e:
            print(f"Failed to load LoRA {lora}: {e}")
            continue
    set_adapters(
        pipeline,
        adapter_names=[lora.name for lora in loras],
        adapter_weights=[lora.strength for lora in loras],
    )
    fuse_lora(pipeline)
    

    return

def generate_images(req, pipeline):
    generator = torch.Generator(device="cuda").manual_seed(42)

    apply_loras(pipeline, req.loras)

    images = inference(
        pipeline,
        req,
        generator=generator,
    )

    pipeline.unfuse_lora()
    
    return images

class GenerationRequest(BaseModel):
    prompt: str
    loras: List[Lora] = []
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7

def test_lora_group(pipeline, lora_group: List[Lora], group_number: int):    
    test_req = GenerationRequest(
        prompt="a simple test image",
        loras=[Lora(name=lora_name, strength=0.8) for lora_name in lora_group],
        width=256,
        height=256,
        steps=10,
    )
    
    try:
        generate_images(test_req, pipeline)
        return True, lora_group
    except Exception as e:
        return False, lora_group

def chunk_loras(lora_list: List[Lora], chunk_size: int = 5) -> List[List[Lora]]:
    """Split LoRAs into groups of specified size"""
    chunks = []
    for i in range(0, len(lora_list), chunk_size):
        chunks.append(lora_list[i:i + chunk_size])
    return chunks

def test():
    global pipeline
    # Load the pipeline if not already loaded
    if not pipeline:
        print("Loading pipeline for LoRA tests...")
        pipeline = load_model()
    
    # Split LoRAs into groups of 3
    all_loras = [f"cereal_{i}" for i in range(1, 50)]  # Example LoRAs
    lora_groups = chunk_loras(all_loras, 5)
    
    successful_groups = 0
    failed_loras = []
    
    print(f"Testing {len(all_loras)} LoRAs in {len(lora_groups)} groups of up to 5...")
    
    for i, lora_group in enumerate(lora_groups, 1):
        print(f"Testing group {i}/{len(lora_groups)}: {lora_group}")
        success, group = test_lora_group(pipeline, lora_group, i)
        if success:
            successful_groups += 1
        else:
            failed_loras.extend(group)
    
    if failed_loras:
        failed_names = [lora.value for lora in failed_loras]
        print(f"✗ Failed LoRAs: {', '.join(failed_names)}")

def test_all_loras():
    test()
    # run second test when all loras are loaded
    test()
    

if __name__ == "__main__":
    test_all_loras()
```

</details>

<details>
<summary><h3>Logs</h3></summary>

Above example gave me following results on RTX5090

```shell
Loading pipeline for LoRA tests...
load_model executed in 3.3787 seconds
Testing 49 LoRAs in 10 groups of up to 5...
Testing group 1/10: ['cereal_1', 'cereal_2', 'cereal_3', 'cereal_4', 'cereal_5']
/aibabe-diffusers/worker-env/lib/python3.11/site-packages/peft/tuners/tuners_utils.py:167: UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
  warnings.warn(
set_adapters executed in 0.1883 seconds
fuse_lora executed in 0.1787 seconds
inference executed in 0.4306 seconds
Testing group 2/10: ['cereal_6', 'cereal_7', 'cereal_8', 'cereal_9', 'cereal_10']
set_adapters executed in 0.6675 seconds
fuse_lora executed in 0.1478 seconds
inference executed in 0.2558 seconds
Testing group 3/10: ['cereal_11', 'cereal_12', 'cereal_13', 'cereal_14', 'cereal_15']
set_adapters executed in 0.6900 seconds
fuse_lora executed in 0.2076 seconds
inference executed in 0.3285 seconds
Testing group 4/10: ['cereal_16', 'cereal_17', 'cereal_18', 'cereal_19', 'cereal_20']
set_adapters executed in 0.7407 seconds
fuse_lora executed in 0.1937 seconds
inference executed in 0.2802 seconds
Testing group 5/10: ['cereal_21', 'cereal_22', 'cereal_23', 'cereal_24', 'cereal_25']
set_adapters executed in 0.9166 seconds
fuse_lora executed in 0.2169 seconds
inference executed in 0.2973 seconds
Testing group 6/10: ['cereal_26', 'cereal_27', 'cereal_28', 'cereal_29', 'cereal_30']
set_adapters executed in 1.7071 seconds
fuse_lora executed in 0.2461 seconds
inference executed in 0.3163 seconds
Testing group 7/10: ['cereal_31', 'cereal_32', 'cereal_33', 'cereal_34', 'cereal_35']
set_adapters executed in 1.8331 seconds
fuse_lora executed in 0.2640 seconds
inference executed in 0.3335 seconds
Testing group 8/10: ['cereal_36', 'cereal_37', 'cereal_38', 'cereal_39', 'cereal_40']
set_adapters executed in 2.0735 seconds
fuse_lora executed in 0.2882 seconds
inference executed in 0.3425 seconds
Testing group 9/10: ['cereal_41', 'cereal_42', 'cereal_43', 'cereal_44', 'cereal_45']
set_adapters executed in 2.3036 seconds
fuse_lora executed in 0.3111 seconds
inference executed in 0.3606 seconds
Testing group 10/10: ['cereal_46', 'cereal_47', 'cereal_48', 'cereal_49']
set_adapters executed in 2.2296 seconds
fuse_lora executed in 0.3228 seconds
inference executed in 0.3789 seconds
Testing 49 LoRAs in 10 groups of up to 5...
Testing group 1/10: ['cereal_1', 'cereal_2', 'cereal_3', 'cereal_4', 'cereal_5']
set_adapters executed in 1.7317 seconds
fuse_lora executed in 0.3396 seconds
inference executed in 0.3807 seconds
Testing group 2/10: ['cereal_6', 'cereal_7', 'cereal_8', 'cereal_9', 'cereal_10']
set_adapters executed in 2.5145 seconds
fuse_lora executed in 0.3403 seconds
inference executed in 0.3808 seconds
Testing group 3/10: ['cereal_11', 'cereal_12', 'cereal_13', 'cereal_14', 'cereal_15']
set_adapters executed in 1.7341 seconds
fuse_lora executed in 0.3392 seconds
inference executed in 0.3820 seconds
Testing group 4/10: ['cereal_16', 'cereal_17', 'cereal_18', 'cereal_19', 'cereal_20']
set_adapters executed in 2.5293 seconds
fuse_lora executed in 0.3392 seconds
inference executed in 0.3761 seconds
Testing group 5/10: ['cereal_21', 'cereal_22', 'cereal_23', 'cereal_24', 'cereal_25']
set_adapters executed in 1.7418 seconds
fuse_lora executed in 0.3438 seconds
inference executed in 0.3790 seconds
Testing group 6/10: ['cereal_26', 'cereal_27', 'cereal_28', 'cereal_29', 'cereal_30']
set_adapters executed in 2.5821 seconds
fuse_lora executed in 0.3522 seconds
inference executed in 0.3876 seconds
Testing group 7/10: ['cereal_31', 'cereal_32', 'cereal_33', 'cereal_34', 'cereal_35']
set_adapters executed in 2.5645 seconds
fuse_lora executed in 0.3373 seconds
inference executed in 0.3804 seconds
Testing group 8/10: ['cereal_36', 'cereal_37', 'cereal_38', 'cereal_39', 'cereal_40']
set_adapters executed in 1.8011 seconds
fuse_lora executed in 0.3375 seconds
inference executed in 0.3843 seconds
Testing group 9/10: ['cereal_41', 'cereal_42', 'cereal_43', 'cereal_44', 'cereal_45']
set_adapters executed in 2.5058 seconds
fuse_lora executed in 0.3371 seconds
inference executed in 0.3810 seconds
Testing group 10/10: ['cereal_46', 'cereal_47', 'cereal_48', 'cereal_49']
set_adapters executed in 1.4389 seconds
fuse_lora executed in 0.3215 seconds
inference executed in 0.3849 seconds
```

</details>

### System Info

- 🤗 Diffusers version: 0.34.0
- Platform: Linux-6.8.0-58-generic-x86_64-with-glibc2.35
- Running on Google Colab?: No
- Python version: 3.11.11
- PyTorch version (GPU?): 2.7.1+cu128 (True)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Huggingface_hub version: 0.33.1
- Transformers version: 4.53.0
- Accelerate version: 1.8.1
- PEFT version: 0.15.2
- Bitsandbytes version: not installed
- Safetensors version: 0.5.3
- xFormers version: not installed
- Accelerator: NVIDIA GeForce RTX 5090, 32607 MiB
- Using GPU in script?: yes
- Using distributed or parallel set-up in script?: no

### Who can help?

@sayakpaul 

> **Comment by sayakpaul (2025-06-27T04:05:03Z):**

Cc: @BenjaminBossan too.

Maybe we could keep a small number of LoRAs on RAM by calling https://github.com/huggingface/diffusers/blob/d7dd924ece56cddf261cd8b9dd901cbfa594c62c/src/diffusers/loaders/lora_base.py#L919
?

Additionally, if you could provide us with a minimal reproducible snippet so that we could obtain the timings that would be helpful. Sometimes, hardware, disk bandwidth, etc. do creep into these things.


> **Comment by BenjaminBossan (2025-06-27T11:57:09Z):**

I think we first need to identify where the time is spent before jumping to solutions. A small reproducer would be ideal for that.


> **Comment by hrazjan (2025-06-27T13:14:48Z):**

Hi @sayakpaul and @BenjaminBossan, I've updated the bug description with a better example and logs it produced on my setup. Let me know it this is good enough. Thanks for looking into this.


> **Comment by BenjaminBossan (2025-06-27T14:25:39Z):**

Thanks for that @hrazjan. Here are my results:

```
set_adapters executed in 0.5061 seconds
fuse_lora executed in 0.2195 seconds
inference executed in 0.5384 seconds
Testing group 2/10: ['cereal_6', 'cereal_7', 'cereal_8', 'cereal_9', 'cereal_10']
set_adapters executed in 0.4466 seconds
fuse_lora executed in 0.1894 seconds
inference executed in 0.3834 seconds
Testing group 3/10: ['cereal_11', 'cereal_12', 'cereal_13', 'cereal_14', 'cereal_15']
set_adapters executed in 0.6302 seconds
fuse_lora executed in 0.2163 seconds
inference executed in 0.4036 seconds
Testing group 4/10: ['cereal_16', 'cereal_17', 'cereal_18', 'cereal_19', 'cereal_20']
set_adapters executed in 0.8395 seconds
fuse_lora executed in 0.2492 seconds
inference executed in 0.4246 seconds
Testing group 5/10: ['cereal_21', 'cereal_22', 'cereal_23', 'cereal_24', 'cereal_25']
set_adapters executed in 1.4510 seconds
fuse_lora executed in 0.2683 seconds
inference executed in 0.4354 seconds
Testing group 6/10: ['cereal_26', 'cereal_27', 'cereal_28', 'cereal_29', 'cereal_30']
set_adapters executed in 1.6897 seconds
fuse_lora executed in 0.2958 seconds
inference executed in 0.4524 seconds
Testing group 7/10: ['cereal_31', 'cereal_32', 'cereal_33', 'cereal_34', 'cereal_35']
set_adapters executed in 1.4007 seconds
fuse_lora executed in 0.3228 seconds
inference executed in 0.4780 seconds
Testing group 8/10: ['cereal_36', 'cereal_37', 'cereal_38', 'cereal_39', 'cereal_40']
set_adapters executed in 1.6028 seconds
fuse_lora executed in 0.3525 seconds
inference executed in 0.4877 seconds
Testing group 9/10: ['cereal_41', 'cereal_42', 'cereal_43', 'cereal_44', 'cereal_45']
set_adapters executed in 1.8075 seconds
fuse_lora executed in 0.3738 seconds
inference executed in 0.5127 seconds
Testing group 10/10: ['cereal_46', 'cereal_47', 'cereal_48', 'cereal_49']
set_adapters executed in 2.2938 seconds
fuse_lora executed in 0.3811 seconds
inference executed in 0.5186 seconds
Testing 49 LoRAs in 10 groups of up to 5...
Testing group 1/10: ['cereal_1', 'cereal_2', 'cereal_3', 'cereal_4', 'cereal_5']
set_adapters executed in 1.9787 seconds
fuse_lora executed in 0.4013 seconds
inference executed in 0.5190 seconds
Testing group 2/10: ['cereal_6', 'cereal_7', 'cereal_8', 'cereal_9', 'cereal_10']
set_adapters executed in 2.6414 seconds
fuse_lora executed in 0.4009 seconds
inference executed in 0.5124 seconds
Testing group 3/10: ['cereal_11', 'cereal_12', 'cereal_13', 'cereal_14', 'cereal_15']
set_adapters executed in 1.9644 seconds
fuse_lora executed in 0.3925 seconds
inference executed in 0.5151 seconds
Testing group 4/10: ['cereal_16', 'cereal_17', 'cereal_18', 'cereal_19', 'cereal_20']
set_adapters executed in 2.6315 seconds
fuse_lora executed in 0.3968 seconds
inference executed in 0.5186 seconds
Testing group 5/10: ['cereal_21', 'cereal_22', 'cereal_23', 'cereal_24', 'cereal_25']
set_adapters executed in 1.9647 seconds
fuse_lora executed in 0.4023 seconds
inference executed in 0.5193 seconds
Testing group 6/10: ['cereal_26', 'cereal_27', 'cereal_28', 'cereal_29', 'cereal_30']
set_adapters executed in 2.6223 seconds
fuse_lora executed in 0.3962 seconds
inference executed in 0.5048 seconds
Testing group 7/10: ['cereal_31', 'cereal_32', 'cereal_33', 'cereal_34', 'cereal_35']
set_adapters executed in 1.9642 seconds
fuse_lora executed in 0.4016 seconds
inference executed in 0.5168 seconds
Testing group 8/10: ['cereal_36', 'cereal_37', 'cereal_38', 'cereal_39', 'cereal_40']
set_adapters executed in 2.6285 seconds
fuse_lora executed in 0.3991 seconds
inference executed in 0.5124 seconds
Testing group 9/10: ['cereal_41', 'cereal_42', 'cereal_43', 'cereal_44', 'cereal_45']
set_adapters executed in 1.9712 seconds
fuse_lora executed in 0.3981 seconds
inference executed in 0.5147 seconds
Testing group 10/10: ['cereal_46', 'cereal_47', 'cereal_48', 'cereal_49']
set_adapters executed in 1.6291 seconds
fuse_lora executed in 0.3822 seconds
inference executed in 0.5167 seconds
```

Is this consistent with what you got? And to be clear, the issue is the increase in `fuse_lora` time that we see in the first run as we progressively load more LoRAs, right?


> **Comment by sayakpaul (2025-06-27T14:45:22Z):**

I think it's the `set_adapters()` method. `fuse_lora()` timing seems expected to me.


> **Comment by hrazjan (2025-06-27T14:52:36Z):**

Yes, it's consistent. The `fuse_lora` is increasing, but only slightly. The main problem is `set_adapters`. As you can see the first run takes only `0.1883 seconds`, but the last ones can take up to `2.6285 seconds`

Alternative solution that came to my mind is something like this:

```python
# preload all state_dicts to GPU
def create_lora_registry(pipeline: StableDiffusionXLPipeline):
    for name, path in lora_paths.items():
        if name in lora_registry:
            continue
        lora_registry[name], _ = pipeline.lora_state_dict(path)
create_lora_registry(pipeline)

...
## generate image
#load lora from dict
entry = lora_registry[lora.name]
pipeline.load_lora_weights(
    pretrained_model_name_or_path_or_dict=entry,
    adapter_name=lora.name,
)
# set_adapters
# inference
# delete_adapters
```

This would keep the loras in VRAM (i think), but the number of total adapters would be small and `set_adapters` should be fast. Unfortunately I wasn't able to make it run.

```
Traceback (most recent call last):
  File "/aibabe-diffusers/aibabe_diffusers/main.py", line 89, in load_loras
    pipeline.load_lora_weights(
  File "/aibabe-diffusers/worker-env/lib/python3.11/site-packages/diffusers/loaders/lora_pipeline.py", line 686, in load_lora_weights
    self.load_lora_into_unet(
  File "/aibabe-diffusers/worker-env/lib/python3.11/site-packages/diffusers/loaders/lora_pipeline.py", line 889, in load_lora_into_unet
    unet.load_lora_adapter(
  File "/aibabe-diffusers/worker-env/lib/python3.11/site-packages/diffusers/loaders/peft.py", line 311, in load_lora_adapter
    inject_adapter_in_model(lora_config, self, adapter_name=adapter_name, **peft_kwargs)
  File "/aibabe-diffusers/worker-env/lib/python3.11/site-packages/peft/mapping.py", line 76, in inject_adapter_in_model
    peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/aibabe-diffusers/worker-env/lib/python3.11/site-packages/peft/tuners/lora/model.py", line 142, in __init__
    super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
  File "/aibabe-diffusers/worker-env/lib/python3.11/site-packages/peft/tuners/tuners_utils.py", line 180, in __init__
    self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
  File "/aibabe-diffusers/worker-env/lib/python3.11/site-packages/peft/tuners/tuners_utils.py", line 527, in inject_adapter
    raise ValueError(error_msg)
ValueError: Target modules {'0.1.transformer_blocks.0.attn1.to_v', '7.1.transformer_blocks.0.attn2.to_out.0', ... } not found in the base model. Please check the target modules and try again.
```


> **Comment by BenjaminBossan (2025-06-27T15:08:08Z):**

I did some quick profiling, if `line_profiler` is not lying to me, the majority of time (95% of time for 50 LoRAs) is spent on here:

https://github.com/huggingface/diffusers/blob/cdaf84a708eadf17d731657f4be3fa39d09a12c0/src/diffusers/loaders/peft.py#L482

There, the majority comes from:

https://github.com/huggingface/diffusers/blob/cdaf84a708eadf17d731657f4be3fa39d09a12c0/src/diffusers/loaders/unet_loader_utils.py#L55

Probably there is room for improvement here.

@hrazjan If you comment out the `weights = scale_expansion_fn(self, weights)` line, do you see improved results?


> **Comment by BenjaminBossan (2025-06-27T15:18:13Z):**

I re-ran the benchmark script with weight scaling commented out. There is still an increase in the `set_adapters` time, but overall it's 10x faster, thus overall, this should not be a bottle neck.

```
set_adapters executed in 0.0373 seconds
fuse_lora executed in 0.2179 seconds
inference executed in 0.5062 seconds
Testing group 2/10: ['cereal_6', 'cereal_7', 'cereal_8', 'cereal_9', 'cereal_10']
set_adapters executed in 0.0628 seconds
fuse_lora executed in 0.1841 seconds
inference executed in 0.3688 seconds
Testing group 3/10: ['cereal_11', 'cereal_12', 'cereal_13', 'cereal_14', 'cereal_15']
set_adapters executed in 0.0900 seconds
fuse_lora executed in 0.2104 seconds
inference executed in 0.3897 seconds
Testing group 4/10: ['cereal_16', 'cereal_17', 'cereal_18', 'cereal_19', 'cereal_20']
set_adapters executed in 0.1173 seconds
fuse_lora executed in 0.2417 seconds
inference executed in 0.4012 seconds
Testing group 5/10: ['cereal_21', 'cereal_22', 'cereal_23', 'cereal_24', 'cereal_25']
set_adapters executed in 0.1441 seconds
fuse_lora executed in 0.2650 seconds
inference executed in 0.4212 seconds
Testing group 6/10: ['cereal_26', 'cereal_27', 'cereal_28', 'cereal_29', 'cereal_30']
set_adapters executed in 0.1750 seconds
fuse_lora executed in 0.2900 seconds
inference executed in 0.4554 seconds
Testing group 7/10: ['cereal_31', 'cereal_32', 'cereal_33', 'cereal_34', 'cereal_35']
set_adapters executed in 0.2063 seconds
fuse_lora executed in 0.3178 seconds
inference executed in 0.4499 seconds
Testing group 8/10: ['cereal_36', 'cereal_37', 'cereal_38', 'cereal_39', 'cereal_40']
set_adapters executed in 0.2324 seconds
fuse_lora executed in 0.3466 seconds
inference executed in 0.4743 seconds
Testing group 9/10: ['cereal_41', 'cereal_42', 'cereal_43', 'cereal_44', 'cereal_45']
set_adapters executed in 0.2589 seconds
fuse_lora executed in 0.3733 seconds
inference executed in 0.4896 seconds
Testing group 10/10: ['cereal_46', 'cereal_47', 'cereal_48', 'cereal_49']
set_adapters executed in 0.2826 seconds
fuse_lora executed in 0.3714 seconds
inference executed in 0.5036 seconds
Testing 49 LoRAs in 10 groups of up to 5...
Testing group 1/10: ['cereal_1', 'cereal_2', 'cereal_3', 'cereal_4', 'cereal_5']
set_adapters executed in 0.2788 seconds
fuse_lora executed in 0.3942 seconds
inference executed in 0.4996 seconds
Testing group 2/10: ['cereal_6', 'cereal_7', 'cereal_8', 'cereal_9', 'cereal_10']
set_adapters executed in 0.2786 seconds
fuse_lora executed in 0.3940 seconds
inference executed in 0.5099 seconds
Testing group 3/10: ['cereal_11', 'cereal_12', 'cereal_13', 'cereal_14', 'cereal_15']
set_adapters executed in 0.2814 seconds
fuse_lora executed in 0.3980 seconds
inference executed in 0.5006 seconds
Testing group 4/10: ['cereal_16', 'cereal_17', 'cereal_18', 'cereal_19', 'cereal_20']
set_adapters executed in 0.2784 seconds
fuse_lora executed in 0.3960 seconds
inference executed in 0.5058 seconds
Testing group 5/10: ['cereal_21', 'cereal_22', 'cereal_23', 'cereal_24', 'cereal_25']
set_adapters executed in 0.2825 seconds
fuse_lora executed in 0.4038 seconds
inference executed in 0.5029 seconds
Testing group 6/10: ['cereal_26', 'cereal_27', 'cereal_28', 'cereal_29', 'cereal_30']
set_adapters executed in 0.2840 seconds
fuse_lora executed in 0.3912 seconds
inference executed in 0.4997 seconds
Testing group 7/10: ['cereal_31', 'cereal_32', 'cereal_33', 'cereal_34', 'cereal_35']
set_adapters executed in 0.2862 seconds
fuse_lora executed in 0.4066 seconds
inference executed in 0.5199 seconds
Testing group 8/10: ['cereal_36', 'cereal_37', 'cereal_38', 'cereal_39', 'cereal_40']
set_adapters executed in 0.2856 seconds
fuse_lora executed in 0.4041 seconds
inference executed in 0.5159 seconds
Testing group 9/10: ['cereal_41', 'cereal_42', 'cereal_43', 'cereal_44', 'cereal_45']
set_adapters executed in 0.2786 seconds
fuse_lora executed in 0.3998 seconds
inference executed in 0.5047 seconds
Testing group 10/10: ['cereal_46', 'cereal_47', 'cereal_48', 'cereal_49']
set_adapters executed in 0.2794 seconds
fuse_lora executed in 0.3740 seconds
inference executed in 0.5033 seconds
```



> **Comment by hrazjan (2025-06-27T15:36:28Z):**

@BenjaminBossan I commented out that line and the results match yours. Thanks a lot. I will try this on my whole project. Hopefully, it should not have any effect on the image quality. I tried with debugger and the `scale_expansion_fn` just returned whatever I set in adapter_weights.


> **Comment by BenjaminBossan (2025-06-27T15:52:01Z):**

I'm sure we can optimize the function to make it fast without outright skipping it :) But yeah, for the time being, please test this workaround and let us know if it helps.


> **Comment by sayakpaul (2025-06-27T16:54:48Z):**

I can look into optimizing it but for the scale expansion stuff please refer to:
https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference?weight-scale=finer+control#hotswapping

Since in this case we're not supplying any custom scaling the `scale_expansion_fn()` just returns the weights set through `adapter_weights()`.


> **Comment by BenjaminBossan (2025-06-30T10:40:34Z):**

There is a very simple optimization, which is to only call `unet.state_dict()` if necessary instead of calling it early, as it's expensive and often not needed. The patch is:

```patch
index 274665204..1856292fd 100644
--- a/src/diffusers/loaders/unet_loader_utils.py
+++ b/src/diffusers/loaders/unet_loader_utils.py
@@ -14,6 +14,8 @@
 import copy
 from typing import TYPE_CHECKING, Dict, List, Union
 
+from torch import nn
+
 from ..utils import logging
 
 
@@ -52,7 +54,7 @@ def _maybe_expand_lora_scales(
             weight_for_adapter,
             blocks_with_transformer,
             transformer_per_block,
-            unet.state_dict(),
+            model=unet,
             default_scale=default_scale,
         )
         for weight_for_adapter in weight_scales
@@ -65,7 +67,7 @@ def _maybe_expand_lora_scales_for_one_adapter(
     scales: Union[float, Dict],
     blocks_with_transformer: Dict[str, int],
     transformer_per_block: Dict[str, int],
-    state_dict: None,
+    model: nn.Module,
     default_scale: float = 1.0,
 ):
     """
@@ -155,7 +157,7 @@ def _maybe_expand_lora_scales_for_one_adapter(
         del scales[updown]
 
     for layer in scales.keys():
-        if not any(_translate_into_actual_layer_name(layer) in module for module in state_dict.keys()):
+        if not any(_translate_into_actual_layer_name(layer) in module for module in model.state_dict().keys()):
             raise ValueError(
                 f"Can't set lora scale for layer {layer}. It either doesn't exist in this unet or it has no attentions."
             )
```

As in this case, we early return from `_maybe_expand_lora_scales_for_one_adapter`, the `state_dict()` call is never performed, keeping the code quick. When I re-ran the script, I got the roughly same results as above with the code commented out.

Also, iterating through `model.named_modules()` should be faster than `model.state_dict()`, at least in my testing that's what I found. As we only need the module names, we could change that too.

@sayakpaul WDYT?


> **Comment by sayakpaul (2025-06-30T13:43:54Z):**

@BenjaminBossan thanks a lot for looking into this! Would you be able to make a PR for this and we can take it from there?


> **Comment by asomoza (2025-06-30T15:47:58Z):**

since this was resolved, I just want to comment (for future reference) that loras for SDXL aren't the same as LLMs, the impact of the PR that added the lora scales is huge for me, what layers do you set the weight have a HUGE beneficial impact on the result, pretty much the same as PAG and IP Adapters.

What I'm trying to say is that even if you can speed up the `set_adapters` by skipping the specific layer scales, loras for SDXL shouldn't be loaded with just a normal scale for all of them.

As an example, with this lora of [pikachu](https://huggingface.co/TheLastBen/Pikachu_SDXL), if you load it with a scale of 1.0, you will get always the same angle and a much lower quality image than the newer finetunes, this is because the lora was probably trained with close-ups and with lower quality images, you can easily overcome that problem by just loading the specific layers for the character and skip everything else like pose, composition and the higher end details that add pixelation and noise.

|with 1.0|with layer scaling|
|---|---|
|![Image](https://github.com/user-attachments/assets/7f35c1fb-24b2-4f27-93f1-53d3dff2201b)|![Image](https://github.com/user-attachments/assets/5a1d5bd1-40e6-445c-a4bb-30a604bd9cc9)|

If you lower the global scale of the lora you can sometimes overcome the problem but at the cost of losing the similarity of the character


> **Comment by hrazjan (2025-06-30T16:08:54Z):**

@asomoza Interesting, I didn't know different layers add diferent features to the generated image. But how do you know which layers effect pose, composition, details etc.? 


> **Comment by asomoza (2025-06-30T16:26:14Z):**

the fist time the layer scaling comes to mind to me was when @cubiq started using them for IP Adapters, later then there was a paper called [InstantStyle](https://huggingface.co/papers/2404.02733). For lora I think there is [B-Lora](https://huggingface.co/papers/2403.14572), probably there are more papers that I don't know.

To learn which ones affect what I just started experimenting, this works only for SDXL and the Unet where each block has a different clearl effect.

For example, for the characters I usually use:

```python
scales = {
    "text_encoder": 1.0,
    "text_encoder_2": 1.0,
    "unet": {
        "down": 0.0,
        "mid": 0.0,
        "up": {
            "block_0": [1.0, 1.0, 0.0],
            "block_1": 0.0,
        },
    },
}
```

but if the lora is really good with a good training I usually add more blocks so it works better, and this is just the tip of the iceberg, you can probably get a lot better results if you do a step scheduling with the scales using callbacks.

If you want I can add to my TODOs a blog post about this with a more detailed explanation, code and images. Sadly this doesn't work that well with DiT models which is one of the main reasons I still love using SDXL.


> **Comment by cubiq (2025-06-30T16:31:18Z):**

I vote for the blog post! 😄 

PS: I also wished cascade received more love


> **Comment by BenjaminBossan (2025-07-01T09:15:22Z):**

> What I'm trying to say is that even if you can speed up the `set_adapters` by skipping the specific layer scales, loras for SDXL shouldn't be loaded with just a normal scale for all of them.

Just to clarify, as it might not be quite clear to everyone, the PR #11834 does not make any functional changes, layer scales are still applied as they were before. It's purely about avoiding an expensive `unet.state_dict()` call for cases where it's not needed. For cases where it is needed, there won't be any speed ups.


> **Comment by asomoza (2025-07-01T09:52:00Z):**

yeah, sorry if what I said was misleading too, I didn't mean in any way that the PR does something wrong, just giving advice on how to get the best results with SDXL loras since OP was using them, the PR doesn't change anything that affects the results with or without blockwise scales.


> **Comment by tuan3w (2025-09-11T15:04:53Z):**

We have faced same problem with loading a lot of loras (~10). As we profiling the code, we see that problem is related to loading weights with peft. Typically, at a time, we only need to use one adapter only. Any solution to this. 

![Image](https://github.com/user-attachments/assets/cde37f40-e848-4403-8ebd-779ebc2491b4)

[profile.html](https://github.com/user-attachments/files/22278160/profile.html)


> **Comment by BenjaminBossan (2025-09-12T12:46:11Z):**

@tuan3w This is difficult to answer with the given information. Do you have a script to reproduce these results? Could you please also provide the package versions you use (PEFT, diffusers, torch, etc.)?

At a glimpse, it looks like a significant amount of time is being spent on requests to the HF hub, are the adapters available locally? Do you get similar results when you run with `HF_HUB_OFFLINE=1`?


> **Comment by tuan3w (2025-09-12T13:43:26Z):**

Hi @BenjaminBossan 
I use latest version: PEFT 0.17.1 and diffusers 0.35.1.
Here is the script to reproduce the issue:

```python
import time

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from loguru import logger
# from pyinstrument import Profiler

controlnet = ControlNetModel.from_pretrained(
    "brad-twinkl/controlnet-union-sdxl-1.0-promax",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    controlnet=controlnet,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

profiler = Profiler()
loras = [
    "k1.safetensors",
] * 6
# profiler.start()
for i, lora in enumerate(loras):
    # path = hf_hub_download("johass/lr", lora)
    adapter_name = lora.replace(".", "_")
    adapter_name = f"{i}_{adapter_name}"
    logger.info(f"Loading {adapter_name}")
    ti = time.time()
    pipeline.load_lora_weights(
        "johass/lr", weight_name=lora, adapter_name=adapter_name, low_cpu_mem_usage=True
    )
    dt = time.time() - ti
    pipeline.set_adapters(adapter_name, adapter_weights=0.7)
    logger.info(f"Loaded in {dt * 1000}ms")

# profiler.stop()
# profiler.write_html("profile2.html")
```

Here is the log of output:

```
2025-09-12 20:38:57.143 | INFO     | __main__:<module>:30 - Loading 0_k1_safetensors
2025-09-12 20:38:57.935 | INFO     | __main__:<module>:37 - Loaded in 776.038646697998ms
2025-09-12 20:38:57.935 | INFO     | __main__:<module>:30 - Loading 1_k1_safetensors
/media/fatman/data/workspace/gen-api/.venv/lib/python3.12/site-packages/peft/tuners/tuners_utils.py:196: UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
  warnings.warn(
2025-09-12 20:38:59.657 | INFO     | __main__:<module>:37 - Loaded in 1705.993413925171ms
2025-09-12 20:38:59.657 | INFO     | __main__:<module>:30 - Loading 2_k1_safetensors
2025-09-12 20:39:01.497 | INFO     | __main__:<module>:37 - Loaded in 1820.7652568817139ms
2025-09-12 20:39:01.497 | INFO     | __main__:<module>:30 - Loading 3_k1_safetensors
2025-09-12 20:39:03.758 | INFO     | __main__:<module>:37 - Loaded in 2238.1811141967773ms
2025-09-12 20:39:03.758 | INFO     | __main__:<module>:30 - Loading 4_k1_safetensors
2025-09-12 20:39:06.537 | INFO     | __main__:<module>:37 - Loaded in 2751.4865398406982ms
2025-09-12 20:39:06.537 | INFO     | __main__:<module>:30 - Loading 5_k1_safetensors
2025-09-12 20:39:09.570 | INFO     | __main__:<module>:37 - Loaded in 3001.9733905792236ms
```




> **Comment by tuan3w (2025-09-12T15:14:29Z):**

I do further investigation in the issue. The problem is that there a lot of checking with parameters. When a new lora is loaded, all keys must be checked with all weight params including base model and loaded loras. Is this behavior expected ?

<img width="1046" height="407" alt="Image" src="https://github.com/user-attachments/assets/2299295a-586d-42af-9890-dda238a821f6" />

Here is script to see what's happened:

```python
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from loguru import logger

# Create pipeline
controlnet = ControlNetModel.from_pretrained(
    "brad-twinkl/controlnet-union-sdxl-1.0-promax",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    controlnet=controlnet,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

print("=== BEFORE LOADING ANY LORA ===")
base_modules = [key for key, _ in pipeline.unet.named_modules()]
print(f"Total base modules: {len(base_modules)}")
print("Sample modules:")
for i, key in enumerate(base_modules[:10]):
    print(f"  {key}")
print("...")

print("\n=== AFTER LOADING 1st LORA ===")
pipeline.load_lora_weights("johass/lr", weight_name="k1.safetensors", adapter_name="lora1")
modules_after_1 = [key for key, _ in pipeline.unet.named_modules()]
print(f"Total modules after 1st LoRA: {len(modules_after_1)}")

# Find new modules added by LoRA
new_modules_1 = set(modules_after_1) - set(base_modules)
print(f"New modules added by 1st LoRA: {len(new_modules_1)}")
for key in sorted(list(new_modules_1)[:10]):
    print(f"  {key}")
if len(new_modules_1) > 10:
    print("...")

print("\n=== AFTER LOADING 2nd LORA ===")
pipeline.load_lora_weights("johass/lr", weight_name="k1.safetensors", adapter_name="lora2")
modules_after_2 = [key for key, _ in pipeline.unet.named_modules()]
print(f"Total modules after 2nd LoRA: {len(modules_after_2)}")

# Find modules added by 2nd LoRA
new_modules_2 = set(modules_after_2) - set(modules_after_1)
print(f"New modules added by 2nd LoRA: {len(new_modules_2)}")
for key in sorted(list(new_modules_2)[:10]):
    print(f"  {key}")
if len(new_modules_2) > 10:
    print("...")

print("\n=== MODULE GROWTH ANALYSIS ===")
print(f"Base model: {len(base_modules)} modules")
print(f"After 1st LoRA: {len(modules_after_1)} modules (+{len(modules_after_1) - len(base_modules)})")
print(f"After 2nd LoRA: {len(modules_after_2)} modules (+{len(modules_after_2) - len(modules_after_1)})")
print(f"Total growth: {len(modules_after_2) - len(base_modules)} new modules")
```

Here is output:


```
The config attributes {'num_control_type': 8} were passed to ControlNetModel, but are not expected and will be ignored. Please verify your config.json configuration file.
Some weights of the model checkpoint at brad-twinkl/controlnet-union-sdxl-1.0-promax were not used when initializing ControlNetModel: 
 ['transformer_layes.0.attn.out_proj.bias, transformer_layes.0.ln_1.bias, transformer_layes.0.attn.in_proj_bias, spatial_ch_projs.weight, transformer_layes.0.mlp.c_proj.bias, transformer_layes.0.mlp.c_proj.weight, transformer_layes.0.attn.in_proj_weight, control_add_embedding.linear_1.weight, transformer_layes.0.ln_2.weight, transformer_layes.0.attn.out_proj.weight, control_add_embedding.linear_2.bias, spatial_ch_projs.bias, transformer_layes.0.mlp.c_fc.weight, transformer_layes.0.ln_2.bias, control_add_embedding.linear_2.weight, transformer_layes.0.mlp.c_fc.bias, task_embedding, transformer_layes.0.ln_1.weight, control_add_embedding.linear_1.bias']
Loading pipeline components...: 100%|█████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 18.46it/s]
You have disabled the safety checker for <class 'diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .
=== BEFORE LOADING ANY LORA ===
Total base modules: 1930
Sample modules:
  
  conv_in
  time_proj
  time_embedding
  time_embedding.linear_1
  time_embedding.act
  time_embedding.linear_2
  add_time_proj
  add_embedding
  add_embedding.linear_1
...

=== AFTER LOADING 1st LORA ===
Total modules after 1st LoRA: 9320
New modules added by 1st LoRA: 7390
  down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.lora_magnitude_vector
  down_blocks.2.attentions.0.transformer_blocks.9.ff.net.2.lora_dropout.lora1
  down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_k.lora_dropout.lora1
  mid_block.attentions.0.transformer_blocks.1.ff.net.2.lora_A
  up_blocks.0.attentions.0.transformer_blocks.3.ff.net.2.lora_B.lora1
  up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_v.lora_magnitude_vector
  up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_k.lora_dropout.lora1
  up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_v.lora_dropout
  up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_out.0.lora_A.lora1
  up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_v.lora_B
...

=== AFTER LOADING 2nd LORA ===
/media/fatman/data/workspace/gen-api/.venv/lib/python3.12/site-packages/peft/tuners/tuners_utils.py:196: UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
  warnings.warn(
Total modules after 2nd LoRA: 11537
New modules added by 2nd LoRA: 2217
  down_blocks.2.attentions.0.transformer_blocks.5.ff.net.2.lora_dropout.lora2
  down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_v.lora_dropout.lora2
  mid_block.attentions.0.transformer_blocks.6.attn2.to_q.lora_A.lora2
  up_blocks.0.attentions.0.proj_in.lora_B.lora2
  up_blocks.0.attentions.0.transformer_blocks.2.ff.net.0.proj.lora_dropout.lora2
  up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_q.lora_A.lora2
  up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_out.0.lora_B.lora2
  up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_v.lora_A.lora2
  up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_q.lora_dropout.lora2
  up_blocks.0.attentions.2.transformer_blocks.1.ff.net.0.proj.lora_B.lora2
...

=== MODULE GROWTH ANALYSIS ===
Base model: 1930 modules
After 1st LoRA: 9320 modules (+7390)
After 2nd LoRA: 11537 modules (+2217)
Total growth: 9607 new modules
````


> **Comment by BenjaminBossan (2025-09-15T15:56:00Z):**

Thanks for the further details and analyses. I can confirm that there is a slowdown due to module checks, as there are thousands of modules involved here (more than 18k at one point). There is most likely room for improvement but I don't think there are any quick wins here.

However, be aware that the profiler can skew the results: In my local testing, running with the profiler turned on leads to the script running 44% longer. I'm not sure if the sampling of pyinstrument could introduce a bias in the results it shows.

I did some of my own profiling and saw that a significant portion of time seems to be spent on an optimization that is supposed to speed up exactly this kind of situation. It seems that the application of this optimization doesn't amortize for the LoRA adapter you used here. You can try disabling it by increasing the threshold:

```python
# disable PEFT optimization by setting very high threshold
import peft
peft.tuners.tuners_utils.MIN_TARGET_MODULES_FOR_OPTIMIZATION = 9999999
```

For me, this led to a ~20% speedup, which isn't much but helps a little bit.

You could also check if [hotswapping](https://huggingface.co/docs/diffusers/v0.35.1/en/tutorials/using_peft_for_inference#hotswapping) is an option for you but if your real use case is similar to your examples, it wouldn't help you here.



> **Comment by BenjaminBossan (2025-09-16T14:05:57Z):**

I created a PR with some small speedups: https://github.com/huggingface/peft/pull/2785. It's not much, roughly 24% overall, but notably there is no longer the need to increase `MIN_TARGET_MODULES_FOR_OPTIMIZATION`.

As mentioned earlier, profiling can significantly skew the results, so for the final measurements, please always turn off the profiler.

With the new optimizations, I get ~15 sec runtime for the script. Of those, roughly 6 sec are spent on adapter injection. That's still a lot but at this point, I see no easy way to add further optimizations. There are just some operations that cannot be avoided, and as we're talking about thousands of modules here, those quickly add up. 


> **Comment by tuan3w (2025-09-17T04:34:55Z):**

I hope there is better solution. I haven't check peft library for checking logic. Do we need to check against all parameters by both loras and base models ? Can we cache these. I will investigate more when I have time. 


> **Comment by BenjaminBossan (2025-09-29T14:33:13Z):**

> I hope there is better solution. I haven't check peft library for checking logic. Do we need to check against all parameters by both loras and base models ? Can we cache these.

Yes, eventually we need to check for each module on the model if a LoRA adapter has to be added, and if it does, we have to create it (which takes time, even when using meta device). As we're dealing with thousands of modules in this example, the time adds up. I don't see any room for caching either.

Btw https://github.com/huggingface/peft/pull/2785 is merged, so if you install PEFT from source, you should see an improvement in loading times.



### #13295 [open] Modular Diffusers 🧨
URL: https://github.com/huggingface/diffusers/issues/13295  by sayakpaul 2026-03-20T05:01:29Z  comments: 20

Hey folks 👋 

We recently released [Modular Diffusers](https://huggingface.co/blog/modular-diffusers), which gives developers the flexibility to reuse existing pipeline blocks in different workflows and also easily implement custom "modular" blocks.

While the `DiffusionPipeline` class has helped establish a standard abstraction layer, it has also been a bit rigid when it comes to extensions. With Modular Diffusers, we want to give our users the utmost flexibility while also complementing our standard pipeline interfaces.

To improve modular, we are soliciting contributions from the community. Contributions for modular can be manifold:

* Implement a modular version of an existing pipeline
* Implement a cool custom modular pipeline and share it with us
* Identify a feature that could be improved in modular and improve it

We're looking forward to hearing from you!

> **Comment by AlanPonnachan (2026-03-21T11:19:19Z):**

I can take a stab at SD3. Since it follows a similar MMDiT and flow-matching architecture as Flux, it seems like a natural next step for the modular ecosystem. I’ll focus on T2I and I2I first.



> **Comment by gasparyanartur (2026-03-21T13:58:05Z):**

Hi! I would like to implement a modular version of the PixArt pipeline family, starting with `PixArtAlphaPipeline`  under [src/diffusers/pipelines/pixart_alpha/](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/pixart_alpha).

The initial scope would focus on the core text-to-image workflow, with reusable blocks aligned to the current pipeline:
- `encoders` for prompt encoding
- `before_denoise` for input / latent prep
- `denoise` for the denoising loop
- `decoders` for VAE decoding and post-processing

I would start with PixArt-alpha, but would structure it such that Sigma-support can be added later. 

Let me know if this looks good, or if something else should be added! 


> **Comment by sayakpaul (2026-03-21T15:33:06Z):**

All good! Let's do this. We also added some skills to help bootstrap AI-agentic implementations for this if you want to leverage those.


> **Comment by gasparyanartur (2026-03-21T18:50:34Z):**

Great! I’ll open a dedicated tracking issue for the PixArtAlphaPipeline, and link it here, then follow up with a draft PR once I have an outline in place.


> **Comment by akshan-main (2026-03-23T10:38:25Z):**

Hi! I'd like to contribute an Ultimate SD Upscale-style Modular Diffusers pipeline for SDXL.

**Initial scope:**
- Built from the standard SDXL img2img modular workflow, with per-tile latent preparation / conditioning / denoising moved into an outer tile loop
- Real `LoopSequentialPipelineBlocks` tile loop with 3 leaf sub-blocks
- Tile-aware SDXL micro-conditioning (`crops_coords_top_left` per tile)
- Linear tile traversal
- Padded crop context for denoising & non-overlapping core paste reconstruction
- Configurable `tile_size`, `tile_padding`, and `strength`

This first pass is intentionally scoped to a minimal mergeable implementation. Follow-up work would cover chess traversal, seam-fixing passes, and alternative blending strategies.


> **Comment by asomoza (2026-03-23T17:46:51Z):**

@akshan-main I posted in your PR, but as a rule of thumb, before starting a PR, you should wait for a maintainer to approve it so you don't waste your time. As I wrote in your PR, Ultimate SD Upscale is a very good candidate for a custom hub block and not something that should go into the core.


> **Comment by akshan-main (2026-03-23T20:11:35Z):**

@asomoza Quick question, I saw the Diffusers MVP contribution call here, and was wondering, would a published custom modular block like this count toward Diffusers MVP contributions?


> **Comment by asomoza (2026-03-23T20:15:04Z):**

it should be, according to the second point:

- Implement a cool custom modular pipeline and share it with us

but @sayakpaul can give you the definitive answer


> **Comment by devanshi00 (2026-03-26T14:28:16Z):**

Hey 👋 this looks super exciting!
I’d like to contribute a modular extension around image-conditioned / first-frame-conditioned workflows, especially for video pipelines.
Instead of implementing a single pipeline directly, I’m thinking of structuring this as:
1. Reusable modular blocks
image preprocessing / conditioning block
VAE encode + latent initialization block
strength-aware latent noise injection block
timestep trimming / remapping block
These would be designed to be reusable across workflows (video, img2img, ControlNet-style pipelines, etc.).
2. Reference custom pipeline
a modular image-to-image / first-frame-conditioned video workflow built using the above blocks
demonstrates reuse of existing denoise / decode blocks with only conditioning swapped

I think this aligns well with the modular philosophy of separating workflows into reusable primitives.
@asomoza @sayakpaul 


> **Comment by sayakpaul (2026-03-26T14:35:50Z):**

@akshan-main sure I think it would be depending on its usefulness. 

@devanshi00 thanks! We have a couple of reference implementations in the core codebase I think. 

I think that sounds very useful. Would you like to get started implementing them as custom blocks, first? Then we can test and determine if they should be made a part of the core codebase.


> **Comment by akshan-main (2026-03-31T01:27:42Z):**

Hey @asomoza @sayakpaul , following up on my earlier comment.

I went ahead and built a tiled SDXL upscaler as a custom Hub block using Modular Diffusers. Used MultiDiffusion for latent-space blending across tiles, optional ControlNet Tile to keep it faithful to the input, progressive 2x passes for 4x and above upscaling, and auto-strength scaling per pass.

It's working well! If you want to test it out https://colab.research.google.com/gist/akshan-main/e9bfd39568bbf7b5795053a27ee89ef6/modular_sdxl_upscale_demo.ipynb

Hub block: https://huggingface.co/akshan-main/modular-sdxl-upscale

would love you guys' review and feedback, and wondering if this would count toward the Diffusers MVP contributions.


> **Comment by akshan-main (2026-03-31T01:42:13Z):**

The per-tile img2img I'd proposed earlier had an issue. Each tile's UNet pass produced independent noise predictions, so adjacent tiles drifted apart even with overlap blending and seam-fix passes. Pixel-space blending couldn't fix predictions that were already divergent. Since the full image evolves from one shared latent in MultiDiffusion, there's no tile drift in my newer, better approach and looks proper.


> **Comment by akshan-main (2026-03-31T23:00:12Z):**

I also went ahead and built a Flux version of the upscaler as a custom Hub block.

Same MultiDiffusion approach but adapted for the Flux transformer architecture; packed latent sequences (crop in 4D, pack per-tile, unpack for blending), RoPE position IDs with global spatial offsets per tile, and guidance via embedding instead of CFG. Uses jasperai/Flux.1-dev-Controlnet-Upscaler for structure preservation.

Hub block: https://huggingface.co/akshan-main/modular-flux-upscale
Demo: https://colab.research.google.com/gist/akshan-main/e7a96626058b5a2065d70b25a2e35c37/modular_flux_upscale_demo.ipynb


> **Comment by akshan-main (2026-04-03T02:14:44Z):**

@sayakpaul Also implemented modular version of LTX and HunyuanVideo 1.5. Do appreciate your review!


> **Comment by akshan-main (2026-04-06T04:30:27Z):**

I also built a Z-Image version of the upscaler as a custom Hub block.

Same MultiDiffusion approach adapted for Z-Image architecture; with optional [ControlNet Union](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union). Z-Image Turbo runs in just 4-8 steps so upscaling is fast.

Hub block: https://huggingface.co/akshan-main/modular-zimage-upscale
Colab demo: https://colab.research.google.com/gist/akshan-main/a2fa5c3dc2af23124749966513a7f3c7/modular-zimage-upscale_demo.ipynb


> **Comment by akshan-main (2026-04-27T17:19:46Z):**

Ernie-Image modular pipeline at #13498 


> **Comment by charchit7 (2026-08-08T18:10:21Z):**

Added Chroma modular Pipeline (T2I) : #14417 


> **Comment by Om-singhaI (2026-08-24T00:16:09Z):**

Hi @sayakpaul @yiyixuxu, I would like to take one of the modular pipeline ports. Following @asomoza's advice earlier in this thread I would rather have the go ahead before writing anything than after.

**The family I would like to claim is Sana.** `src/diffusers/modular_pipelines/` currently covers anima, cosmos, ernie_image, flux, flux2, helios, hunyuan_video1_5, ideogram4, krea2, ltx, ltx2, minimax_h3, minimax_music3, qwenimage, stable_diffusion_3, stable_diffusion_xl, wan, wan_animate_2 and z_image. Sana is absent, and there are four standard pipelines to port: `pipeline_sana.py`, `pipeline_sana_sprint.py`, `pipeline_sana_controlnet.py` and `pipeline_sana_sprint_img2img.py`.

Two reasons I picked it over the other gaps. It is the one mainstream text to image family in the repo that still runs at real weights on a laptop, so I can diff outputs against the standard pipelines rather than only asserting on tiny fixtures. And the four pipelines share enough of the denoise loop that a single blockset covers all of them, which keeps the diff proportionate to the coverage.

Shape would mirror z_image and krea2: `encoders.py`, `before_denoise.py`, `denoise.py`, `decoders.py`, `modular_blocks.py` and `modular_pipeline.py` under `src/diffusers/modular_pipelines/sana/`, a matching package under `tests/modular_pipelines/sana/`, and docs. Staged as text to image first, then Sprint, then ControlNet, then img2img, so each PR stands on its own.

Five questions before I start, since each one changes the first PR:

1. Is Sana the family you would want next, or is there one further up your list? Happy to be redirected.
2. Should PR 1 be text to image only, or text to image and Sprint together? Sprint changes the timestep and scheduler path, so it is either a clean second PR or a fair amount of branching inside the first.
3. @lawrence-cj has #13881 open, which adds another Sana family transformer. Is that likely to land in a shape that affects the blockset, and would you rather I wait for it or work alongside it?
4. ControlNet: a separate blockset in the same package, or blocks inside the main one selected by config? qwenimage and flux differ here and I could not tell which is the current preference.
5. CONTRIBUTING mentions starting a Hub repo alongside the feature request. Is that aimed at custom blocks rather than in repo ports like this, or would you like one either way?

One last thing, meant as planning rather than a nudge: a number of community modular PRs are currently waiting on a first review, some of them two or three weeks old. If the queue is tight right now I would genuinely rather time this to suit you than add to it. I am in no hurry and the work is the same whenever it starts.



> **Comment by ViktoriiaRomanova (2026-09-03T13:48:35Z):**

Hi @sayakpaul, I’m interested in implementing a modular pipeline for Kandinsky 3. Would this contribution still be useful?


> **Comment by yiyixuxu (2026-09-04T22:59:16Z):**

  Hey, for everyone interested in implementing modular pipelines 
  
 As you've noticed, we're currently very behind on reviewing the community modular PRs that are already open. So please consider implementing a remote hub version of any pipeline you're interested in first, and share it back here so that other users don't have to wait for the review process, and when we have more bandwidth, we'll invite you to open a PR to upstream it — and that review will be easier since the code will already have users
  
The custom blocks guide covers how to publish one: https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks



### #14700 [open] Add VideoDeltaNet-H3 Model
URL: https://github.com/huggingface/diffusers/issues/14700  by haochengxi 2026-09-04T02:57:45Z  comments: 13

### Model/Pipeline/Scheduler description

VideoDeltaNet (VDN-H3) is a hybrid-attention variant of MiniMax H3 designed for faster video generation with near-lossless quality. It replaces dense video-video attention with a combination of local sliding-window softmax attention and long-range linear attention, while keeping the original H3 backbone largely unchanged. 

All weights and training and inference code are available.

### Open source status

- [x] The model implementation is available.
- [x] The model weights are available (Only relevant if addition is not a scheduler).

### Provide useful links for the implementation

Weight: https://huggingface.co/OpenVDN/vdn-minimax-h3
Code: https://github.com/OpenVDN/vdn-minimax-h3
I'm the author :)

> **Comment by github-actions[bot] (2026-09-04T02:58:00Z):**

Thanks for the request!

**How new model support works in Diffusers**

We're a small team, and our review queue shouldn't be what decides whether a model is usable in Diffusers. With [Modular Diffusers](https://huggingface.co/docs/diffusers/modular_diffusers/overview), a pipeline can live as remote code in any Hub repo and load straight from there with `from_pretrained`.

🛠️ **Want to bring this model to Diffusers?**

Please start with a Hub repo — you don't need anything from us to do that, and people can use it immediately. From there we decide how to support it: we might work with the authors, upstream an existing community version, or just point people at the one on the Hub. The pipelines we integrate are usually the ones people are already running.

Tag `@asomoza` when you have something to share — we'll give feedback on the implementation, help get it in front of people, and add the ones we like to our hand-picked [Modular Pipelines](https://huggingface.co/collections/diffusers/modular-pipelines) collection. Tell us where you hit friction along the way, too: confusing APIs, missing docs, bugs. That feedback is worth as much to us as the pipeline.

👋 **Are you an author of the model?** We'd love to hear from you — comment here and we'll help you pick the path that fits.

📚 [Quickstart](https://huggingface.co/docs/diffusers/modular_diffusers/quickstart) · [Building custom blocks](https://huggingface.co/docs/diffusers/modular_diffusers/custom_blocks) — template repo, and how to publish to the Hub · [Modular Pipelines](https://huggingface.co/collections/diffusers/modular-pipelines) and [Custom Blocks](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks) — examples to crib from

*This is an automated message.*



> **Comment by haochengxi (2026-09-10T00:39:25Z):**

> Thanks for the request!
> 
> **How new model support works in Diffusers**
> 
> We're a small team, and our review queue shouldn't be what decides whether a model is usable in Diffusers. With [Modular Diffusers](https://huggingface.co/docs/diffusers/modular_diffusers/overview), a pipeline can live as remote code in any Hub repo and load straight from there with `from_pretrained`.
> 
> 🛠️ **Want to bring this model to Diffusers?**
> 
> Please start with a Hub repo — you don't need anything from us to do that, and people can use it immediately. From there we decide how to support it: we might work with the authors, upstream an existing community version, or just point people at the one on the Hub. The pipelines we integrate are usually the ones people are already running.
> 
> Tag `@asomoza` when you have something to share — we'll give feedback on the implementation, help get it in front of people, and add the ones we like to our hand-picked [Modular Pipelines](https://huggingface.co/collections/diffusers/modular-pipelines) collection. Tell us where you hit friction along the way, too: confusing APIs, missing docs, bugs. That feedback is worth as much to us as the pipeline.
> 
> 👋 **Are you an author of the model?** We'd love to hear from you — comment here and we'll help you pick the path that fits.
> 
> 📚 [Quickstart](https://huggingface.co/docs/diffusers/modular_diffusers/quickstart) · [Building custom blocks](https://huggingface.co/docs/diffusers/modular_diffusers/custom_blocks) — template repo, and how to publish to the Hub · [Modular Pipelines](https://huggingface.co/collections/diffusers/modular-pipelines) and [Custom Blocks](https://huggingface.co/collections/diffusers/modular-diffusers-custom-blocks) — examples to crib from
> 
> *This is an automated message.*
> 


@asomoza 
The implementation is on the Hub and loads as a Modular Diffusers pipeline, no fork and no patched diffusers:
 https://huggingface.co/OpenVDN/vdn-minimax-h3

It reuses MiniMaxH3Blocks unchanged. The only thing we ship is the transformer, a hybrid-attention variant of MiniMaxH3Transformer3DModel. Every other component still points at MiniMaxAI/MiniMax-H3. Can be used through:

```
pipe = ModularPipeline.from_pretrained("OpenVDN/vdn-minimax-h3", workflow="t2va")
pipe.load_components(trust_remote_code=True, torch_dtype=torch.bfloat16)
```

We would like to see if we can merge it into Diffusers repo. Thanks so much!


> **Comment by yiyixuxu (2026-09-10T23:59:04Z):**

cd @asomoza 
can you add to our collection?


> **Comment by haochengxi (2026-09-11T23:44:10Z):**

By the way, while trying to implement offloading for VDN-H3, we found this PR really helpful. However, I noticed that it’s being closed.

Would you consider fixing the bug, or would you prefer that I submit a PR?



> **Comment by asomoza (2026-09-12T00:04:08Z):**

HI, seems you didn't put the link to the PR, but if it helps, I got it to run on a 40GB GPU with this:

```python
manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained(
    "OpenVDN/vdn-minimax-h3", workflow="t2va", components_manager=manager, collection="vdn"
)
pipe.load_components(names="transformer", trust_remote_code=True, dtype=torch.bfloat16, fp8={"transformer": True})
pipe.load_components(trust_remote_code=True, dtype=torch.bfloat16)

offload = {
    "onload_device": torch.device("cuda"),
    "offload_device": torch.device("cpu"),
    "use_stream": True,
    "low_cpu_mem_usage": True,
}

pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
pipe.vae.to("cuda")
pipe.audio_vae.to("cuda")

out = pipe(
    prompt=prompt,
    num_frames=192,
    num_inference_steps=9,
    generator=torch.Generator().manual_seed(0),
    output=["videos", "audio", "sampling_rate"],
)
```


> **Comment by haochengxi (2026-09-12T02:42:34Z):**

https://github.com/huggingface/diffusers/pull/13238
The link to the PR ↑.

The problem I encountered is that when I quantize the linear layers into FP8 and register the weights as buffers, it can not be offloaded correctly when trying to stream the offloading process. By adding the fix above, it works. So maybe adding that fix back will be helpful!


> **Comment by DN6 (2026-09-14T08:48:13Z):**

@haochengxi what library/method are you using to quantize the weights? Is it custom? I've asked the author of #13238  to reconsider opening the PR in the meantime. 


> **Comment by haochengxi (2026-09-14T08:56:32Z):**

It's a custom quantized method (FP8 Linear Quantization). I think that the PR's fix is quite general, not limited to custom FP8 Quantization. Thanks so much!


> **Comment by vladmandic (2026-09-14T09:26:34Z):**

@haochengxi btw, from what i can tell you allow either your custom fp8 quant or original bf16. but if user passes quantization_config, it is ignored. afaik, there is no reason to skip user-selected quantization and force bf16?


> **Comment by haochengxi (2026-09-15T04:24:44Z):**

> [@haochengxi](https://github.com/haochengxi) btw, from what i can tell you allow either your custom fp8 quant or original bf16. but if user passes quantization_config, it is ignored. afaik, there is no reason to skip user-selected quantization and force bf16?

Thanks let me check for more details about what's the best practice.


> **Comment by haochengxi (2026-09-15T23:05:12Z):**

@vladmandic The quantization is torchao now. The weights are ordinary parameters,
so `apply_group_offloading(..., use_stream=True)` on stock diffusers streams them as they
are, and we no longer need the fix from #13238 on our side. (but still, I think that PR makes sense)

For quantization config, it needs to be more cautious: the loader
quantizes each weight as it loads, and this component merges LoRA adapters into
`Linear.weight` after loading, which a quantized weight cannot take (that is where a
passed-through config failed before, rather than being dropped). `from_pretrained` now
catches it, loads bf16, merges, then applies it. It accepts a `TorchAoConfig` and
raises on any other backend; `fp8=True` is just a preset `TorchAoConfig`.

```python
from diffusers import TorchAoConfig
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow

pipe.load_components(
    trust_remote_code=True, torch_dtype=torch.bfloat16,
    quantization_config={"transformer": TorchAoConfig(
        Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))},
)
```

@asomoza on memory: with the transformer offloaded per block (`offload_type="block_level"`,
`num_blocks_per_group=1`, `use_stream=True`) and the other components offloaded, the
full 345 frames now peak at 22 GB in bf16 and 20 GB in fp8, so the 14 s clip runs on a
24 GB card.


> **Comment by haochengxi (2026-09-17T07:32:27Z):**

Hi @asomoza, thanks for testing this!

Just following up on upstreaming VDN-H3. Since the Modular Pipeline is working now and most components are reused directly from MiniMax H3, would it make sense to upstream the VDN transformer into Diffusers? I'm happy to open a PR and add the implementation/tests/docs if that's a direction you'd be open to.

Also, I don't think the Hub repo has made it into the Modular Pipelines collection yet. Happy to make any changes needed on our side.



> **Comment by asomoza (2026-09-17T13:09:01Z):**

Hi @haochengxi it is here: https://huggingface.co/collections/diffusers-modular/community-modular-diffusers-blocks but we have a small list  for now and still deciding where we will put it so it is more visible, this will take some time.

About adding the VDN transformer, we usually do it only for models that are very popular, but lets wait for @yiyixuxu decision about this.



### #15218 [closed] `_tp_plan` / `_cp_plan`: allow entries for modules that are not always present
URL: https://github.com/huggingface/diffusers/issues/15218  by yiyixuxu 2026-11-09T05:37:22Z  comments: 8

Follow-up to #14609. This is a discussion, not a request for a PR: I want to hear opinions on whether we
need this and which shape it should take before anyone starts on it.

## Problem

`_tp_plan` and `_cp_plan` are class attributes, and both are resolved by attribute lookup:
`resolve_tp_shard_specs` / `_resolve_tp_plan` in `src/diffusers/hooks/tensor_parallel.py` walk each
key with `getattr`, and `_find_submodule_by_name` in `src/diffusers/hooks/context_parallel.py`
raises `ValueError` for an atom that is not a submodule. So every key must resolve on the instance,
or applying the plan fails before anything is sharded.

That means a plan can only list modules that exist on every instance of the class. #14609 hit this
with fused projections and had to leave them out:

```python
# `attn.to_qkv` is deliberately not listed: it only exists after `fuse_projections()`, and the plan is
# resolved by attribute lookup, so an unconditional entry would break the ordinary unfused model.
```

The same limit applies to any module that depends on the config: a model with optional layers
(a guidance embedder, an extra projection enabled by a flag, a block variant chosen per
checkpoint) cannot have one plan that covers all of its configurations.

## Options

One way is to let a plan mark some entries as optional, so they are skipped when the module is absent
and still raise for a typo in a required entry. Two possible shapes:

1. an explicit wrapper, e.g. `"transformer_blocks.*.attn.to_qkv": Optional(PackedColwiseParallel([1, 1, 1]))`
2. match plan keys against the modules that exist, like `transformers` does, and warn for an unmatched key

(1) keeps the plan strict by default; (2) is less code in each model but a typo goes unnoticed. A third
option is to leave the plans as they are and accept that a model with config-dependent layers needs
separate handling. Whatever we pick should apply to both `_tp_plan` and `_cp_plan`.

Opinions welcome, especially from anyone who has written a plan for a model with optional layers.

## Related

Nothing stops a user from calling `fuse_qkv_projections()` on a model that is already sharded with TP,
and it cannot work: the sharded `to_q`/`to_k`/`to_v` weights are `DTensor`s, and `fuse_projections`
concatenates them into a new plain `nn.Linear` that has no TP hooks and no plan entry. We should
either raise there, like we already do for offloading, saving and LoRA on a sharded model, or let the
plan cover the fused Linear, which needs the optional entry discussed above.

---

_Posted by Claude Code on behalf of @yiyixuxu; the text was reviewed and approved by them._


> **Comment by yiyixuxu (2026-11-09T05:39:46Z):**

cc @sayakpaul @JingyaHuang 
let me know if we need to make _tp_plan/_cp_plan work with optional layers 


> **Comment by sayakpaul (2026-11-09T06:56:42Z):**

Thanks for opening the thread!

I think the benefits of fusion start to diminish with parallelism. The benefits start showing up with specific quantization schemes, depending on the model parameters and input sizes. But we don't support TP + quant yet. To get more ideas on this, I ran a small benchmark with H3 and TP:

<img width="783" height="356" alt="Image" src="https://github.com/user-attachments/assets/50621a1c-a72b-402d-bc60-3114711ded2f" />

Full artifact: https://claude.ai/artifact/CNbMdEuNJFuiZuoJZFkrNW

(it's possible that the implementation is suboptimal but that is a different issue IMO)

So, based on the numbers, I think instead of working on the first option or any other more intrusive changes, we should simply warn the users (i.e., the second option). WDYT?


> **Comment by yiyixuxu (2026-11-09T07:13:32Z):**

@sayakpaul 

thanks for looking into this! so it's feature is not just for fusing use case in particular - although that's what inspired it 

more generally, currently we don't really know how to create a plan involve optional layers in models, i.e. if a model class support 2 different checkpoint with slightly different layout, e.g. add_k_proj/add_v_proj are None in one checkpoint but not in another etc 
 


> **Comment by sayakpaul (2026-11-09T07:15:25Z):**

Ah yes, that is true. Let think about it a bit and get back to you.


> **Comment by sayakpaul (2026-11-09T08:12:46Z):**

Okay some further thoughts. 

I think https://github.com/huggingface/diffusers/pull/15197 is quite related to this discussion because it's linked to https://github.com/huggingface/diffusers/pull/15137 where the Cosmos checkpoints have optional layers. 

#15197 basically exposes a model-level `_get_tp_plan()` which model classes can override when there is conditional logic to be implemented. For example (from https://github.com/huggingface/diffusers/pull/15137/):

```py
def _get_tp_plan(self):
    tp_plan = super()._get_tp_plan().copy()
    if self.config.hidden_act == "silu":
        tp_plan.update(
            {
                "layers.*.mlp.gate_proj": "colwise",
                "layers.*.mlp_moe_gen.gate_proj": "colwise",
            }
        )
    return tp_plan
```

The idea is that the TP-related code would read from what is returned from `_get_tp_plan()` instead of just the class-level `_tp_plan` attribute.

I think marking with `Optional` like you suggested could also work. IMO, the `_get_tp_plan()` approach is more explicit.


> **Comment by yiyixuxu (2026-11-09T18:45:20Z):**

ohh i see, indeedd it's related cc @DN6 into the discussion too 

I don't think `_get_tp_plan`, as how it is currently implemented in the PR, is the right design for this though. 

it's basically to express "this layer is optional" with bunch of if ... else in an override. , i.e. compare this
```python
_tp_plan = {
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
    ...
}

def _get_tp_plan(self):
    tp_plan = super()._get_tp_plan().copy()
    if self.config.hidden_act == "silu":
        tp_plan.update(
            {
                "layers.*.mlp.gate_proj": "colwise",
                "layers.*.mlp_moe_gen.gate_proj": "colwise",
            }
        )
    return tp_plan
```

with something like this

```python
_tp_plan = {
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
    # only in the SiLU checkpoints
    "layers.*.mlp.gate_proj": Optional("colwise"),
    "layers.*.mlp_moe_gen.gate_proj": Optional("colwise"),
    ...
}
```

I think it's expressing the exact same amount of info, but the plan lives in two places instead of one so that's not very nice/ easy to read and I think better to keep it declarative like either one of options proposed in the issue 

However, if we want the plan to be more customizable (not sure if there is a use case), we can allow the override from the checkpoint level (not the model level like in the PR ) -> so that would be something user config in config.json 


> **Comment by sayakpaul (2026-11-10T02:02:48Z):**

Then Option 1 (`Optional`) seems like a better choice to me. I think we can also warn for unmatched modules just for comprehensive logging, too. 

Let's see what @DN6 has to say and I can open a PR soon because this blocks TP for latest Cosmos checkpoints (and I think this will be a recurring case). Cc: @yzhautouskay 


> **Comment by DN6 (2026-11-10T07:39:10Z):**

Option 1 looks cleanest to me 👍🏽. Just have to make sure we have no clashes with `typing.Optional` (although I think we removed all of these when we bumped the min Python version)



### #15267 [open] [tests] A tester mixin for running modular blocks on their own
URL: https://github.com/huggingface/diffusers/issues/15267  by yiyixuxu 2026-11-12T18:32:02Z  comments: 7

A modular pipeline is meant to be used in pieces: you can take the text encoder block, turn it into its own pipeline, and run it separately from the rest. We want that to keep working for every pipeline, and we want it to be intuitive to use that way. Today each test file checks the first by hand, if at all, and nothing checks the second.

### What we want to test

For each top-level block of a pipeline (text encoder, VAE encoder, denoise, decode, ...):

- it builds into its own pipeline with `init_pipeline()`
- that pipeline loads only the components the block declares, e.g. just `vae` for a VAE encoder
- it runs on the inputs it declares and returns the outputs it declares
- running the blocks one after another, each fed the outputs of the ones before it, gives the same result as the assembled pipeline
- it is intuitive to use on its own: what you pass in and what you get back make sense without knowing the rest of the pipeline

### Is it intuitive to use this way?

Running image -> encoder -> denoise -> decoder by hand shows interface problems that the assembled pipeline hides. Patterns we have run into so far:

- **Components it does not own.** It would be weird for a VAE encoder block to require the transformer just because it reads a value from its config. A block run on its own should load only what it actually runs.
- **Too much crossing a boundary.** A decoder that takes packed latents also needs `latent_height` and `latent_width` to unpack them, so the user has to carry those over from the denoise step. We fixed this for flux, krea2 and ltx by making unpack part of the denoise group (#14730, #14737). What one block hands to the next should be a short, canonical list (`prompt_embeds`, `image_latents`, `latents`).
- **Optional in, something out.** A text encoder block should not require the guider: the user should not have to configure guidance in two separate places, once for the text encoder and once for the denoiser. Run on its own, it gives back negative embeddings when you pass a `negative_prompt` and none when you don't. Getting a `negative_prompt=None` encoded anyway is unintuitive (#14469).
- **Redundant inputs.** A denoise block that takes `image_latents` should not also need `height` and `width`, unless the pipeline supports an output size different from the reference's.

In this test we want to check for these specific patterns. We catch new ones over time during reviews, so the mixin should make adding one cheap.

### What it looks like today

Fifteen test files under `tests/modular_pipelines/` build a one-block pipeline by hand, each with its own setup. For example, from the MiniMax H3 Image tests:

```py
class TestMiniMaxH3ImageModularBlocks(MiniMaxH3ImageModularPipelineTesterConfig):
    def test_vae_encoder_block_standalone(self):
        pipe = MiniMaxH3ImageEditVaeEncoderStep().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)

        reference_latents = pipe(
            reference_images=[Image.new("RGB", (64, 32))],
            generator=self.get_generator(0),
            output="reference_latents",
        )

        assert set(pipe.components) == {"vae"}
        assert len(reference_latents) == 1
        assert reference_latents[0].shape == (1, 4, 8, 16)
```

The build, the load, the component check and the "does it return its output" check are the same for every block in every pipeline. Only the last assertion is specific to this model.

### Proposal

A tester mixin next to the existing ones in `tests/modular_pipelines/testing_utils/`, composed with the same config class as the others. It walks the top-level blocks of each workflow and runs the checks above, so a new pipeline gets them without writing anything. Model-specific assertions stay as methods on the pipeline's own test class.

### How to contribute

This issue carries the `idea-contribution-welcome` label. We are not currently looking for a PR here, we are looking for your feedback, especially if you use modular blocks on their own:

- other patterns you think this test should check for
- places where running a block standalone is unintuitive today: an input you did not expect to need, an output you did not expect to get, a value you had to carry from one block to the next

we may invite you to work with us on this once we agree on the implementation. If we decide to write the PR ourselves, anyone whose feedback shaped it will be added as a co-author.

---
*Posted by Claude on behalf of @yiyixuxu. The text was drafted together and approved by yiyixuxu.*


> **Comment by yiyixuxu (2026-11-12T18:53:43Z):**

cc @sayakpaul let me know what you think


> **Comment by sayakpaul (2026-11-16T10:14:39Z):**

Good idea!

> A tester mixin next to the existing ones in tests/modular_pipelines/testing_utils/, composed with the same config class as the others. It walks the top-level blocks of each workflow and runs the checks above, so a new pipeline gets them without writing anything. Model-specific assertions stay as methods on the pipeline's own test class.

I think we should allow defining the list of these modular blocks we want to test for each pipeline under `src/diffusers/modular_pipelines`. We usually define these blocks under a `modular_blocks_*.py` file, pertaining to a pipeline. For example:

```py
AUTO_BLOCKS = InsertableDict(
    [
        ("prompt_upsampler", Flux3AutoPromptUpsampleStep()),
        ("text_encoder", Flux3TextEncoderStep()),
        ("vae_encoder", Flux3AutoVaeEncoderStep()),
        ("denoise", Flux3AutoCoreDenoiseStep()),
        ("decode", Flux3DecodeStep()),
    ]
)
```

Do we want to stick to these blocks only? I think these should suffice. Each of these blocks has different inputs, different outputs, and shapes to check against. I think these should be the atomic tests the mixin would have. Then on the point of

> Redundant inputs. A denoise block that takes image_latents should not also need height and width, unless the pipeline supports an output size different from the reference's.

We can define properties at the mixin level (so that they can be overridden) to run these negations. So, a rough skeleton for this mixin could be like so:

```py
class ModularBlocksTesterMixin:
    # The only values one block may hand to the next. Anything else a block reads has to come from the user.
    handoff_names = frozenset(["prompt_embeds", "negative_prompt_embeds", "image_latents", "latents"])
    # Components a block must not load when it runs on its own.
    unowned_components = {
        "text_encoder": {"guider", "transformer", "unet", "vae"},
        "vae_encoder": {"guider", "transformer", "unet"},
        "decode": {"guider", "transformer", "unet"},
    }
    # Inputs a block must run without, even when the user passes them to the assembled pipeline.
    redundant_inputs = {"decode": {"height", "width"}}
    # Optional input -> (a value for it, the output that must be None when the input is not passed).
    optional_outputs = {"negative_prompt": ("blurry", "negative_prompt_embeds")}

    def get_block_pipeline(self, block_name):
        block = self.pipeline_blocks_class().sub_blocks[block_name]
        pipe = block.init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        return pipe.to(torch_device)

    def get_inputs(self, pipe, handoffs=None, skip=()):
        """The dummy inputs `pipe` declares, every optional input given, plus what earlier blocks handed
over."""
        optional = {name: value for name, (value, _) in self.optional_outputs.items()}
        inputs = {**self.get_dummy_inputs(), **optional, **(handoffs or {})}
        return {k: v for k, v in inputs.items() if k in pipe.blocks.input_names and k not in skip}

    def test_blocks_load_only_their_components(self):
        for block_name in self.pipeline_blocks_class().sub_blocks:
            pipe = self.get_block_pipeline(block_name)
            unowned = set(pipe.components) & self.unowned_components.get(block_name, set())
            assert not unowned, f"Block '{block_name}' loads components it does not own: {sorted(unowned)}"

    def test_blocks_chained_match_pipeline(self):
        pipe = self.pipeline_blocks_class().init_pipeline(self.pretrained_model_name_or_path)
        pipe.load_components(dtype=torch.float32)
        pipe.to(torch_device)
        expected = pipe(**self.get_inputs(pipe), output=self.output_name)

        # one generator for the whole chain, as in the assembled pipeline
        handoffs = {"generator": self.get_generator(0)}
        for block_name in self.pipeline_blocks_class().sub_blocks:
            block_pipe = self.get_block_pipeline(block_name)
            outputs = block_pipe(
                **self.get_inputs(block_pipe, handoffs, skip=self.redundant_inputs.get(block_name, ())),
                output=block_pipe.blocks.intermediate_output_names,
            )
            handed_over = self.handoff_names | {self.output_name}
            handoffs.update({k: v for k, v in outputs.items() if k in handed_over and v is not None})

        assert_tensors_close(handoffs[self.output_name], expected, atol=1e-3, rtol=1e-3)

    def test_optional_input_gates_its_output(self):
        for block_name, block in self.pipeline_blocks_class().sub_blocks.items():
            for input_name, (value, output_name) in self.optional_outputs.items():
                if input_name not in block.input_names:
                    continue
                pipe = self.get_block_pipeline(block_name)
                inputs = self.get_inputs(pipe, skip={input_name})
                assert pipe(**inputs, output=output_name) is None, (
                    f"Block '{block_name}' returns `{output_name}` without `{input_name}`"
                )
                assert pipe(**inputs, **{input_name: value}, output=output_name) is not None
```

Then each block-level test suite could subclass from the mixin and override as needed.


> **Comment by charchit7 (2026-11-17T12:38:47Z):**



I have been tinkering with qwen-image for a use-case, so I ran the above sketch as written against the merged QwenImage blocks (`hf-internal-testing/tiny-qwenimage-modular`, CPU).  


It catches the guider / negative-prompt pattern in the text encoder as intended. Two things came up in `test_blocks_chained_match_pipeline`:

**1. Legitimate hand-offs outside the canonical list.** It fails with `Required input 'prompt_embeds_mask' is missing`. QwenImage's text encoder hands `prompt_embeds_mask` and `negative_prompt_embeds_mask` to denoise, because the transformer needs them, and they aren't in `handoff_names`. The error names the input but not that the mixin filtered it out. Asserting up front that everything a block reads from earlier blocks is in `handoff_names`, plus a per-pipeline `extra_handoff_names`, would give a message like "`denoise` reads `prompt_embeds_mask` from `text_encoder`, which is not a declared hand-off", and each exception becomes a reviewable line in the test file.

**2. A pattern that isn't on the list yet: config from a component the block doesn't own, with a silent default.** With the masks added to `handoff_names`, the test still fails: `Shape mismatch: actual [1, 3, 16, 16] vs expected [1, 3, 32, 32]`. The standalone denoise pipeline has no `vae`, so `vae_scale_factor` falls back to 8, while the tiny VAE's factor is 4. The latents come out as `(1, 4, 1, 4, 4)` where the assembled pipeline gives `(1, 4, 1, 8, 8)`. With the real VAE the factor is 8, so it's invisible outside the tiny repo.

I understand the fallback was required in #14943 added it for Anima so the denoise block runs without a VAE. The cost is that it's only right when the VAE matches the default. With the real QwenImage VAE the factor is 8, so it's invisible outside the tiny repo, but a finetune or a different VAE with another factor would get the wrong latent size with no error.

So, for the mixin, this is the flip side of "components it does not own": the block doesn't require the VAE, but it guesses a value from it. The chained test catches it only because the tiny model happens to differ from the default. A direct check could compare the derived properties a block uses (`vae_scale_factor`, `num_channels_latents`, ...) between the block pipeline and the assembled one. How to resolve a mismatch is a separate question, e.g. letting the user pass the value, or handing it over with the latents.

It isn't specific to QwenImage. Running the same two steps on other tiny repos:

| Pipeline | Extra hand-off | Canonical hand-offs only | `vae_scale_factor` standalone / assembled | Latents standalone / assembled |
|---|---|---|---|---|
| Krea 2 | `prompt_embeds_mask`, `negative_prompt_embeds_mask` | `Required input 'prompt_embeds_mask' is missing` | 8 / 4 | `(1, 4, 1, 4, 4)` / `(1, 4, 1, 8, 8)` |
| Flux | `pooled_prompt_embeds` | `AttributeError: 'NoneType' object has no attribute 'repeat'` | 8 / 1 | `(1, 1, 0, 0)` / `(1, 1, 8, 8)` |
| Z-Image | none | OK | 8 / 2 | `(1, 16, 4, 4)` / `(1, 16, 16, 16)` |


Repro for both (tiny repo, CPU):

<details>
<summary>script</summary>

```python
import torch

from diffusers.modular_pipelines import QwenImageAutoBlocks


repo = "hf-internal-testing/tiny-qwenimage-modular"
blocks = QwenImageAutoBlocks()


def build(block):
    pipe = block.init_pipeline(repo)
    pipe.load_components(dtype=torch.float32)
    return pipe


full = build(blocks)
text_encoder = build(blocks.sub_blocks["text_encoder"])
denoise = build(blocks.sub_blocks["denoise"])

text_outputs = ["prompt_embeds", "negative_prompt_embeds", "prompt_embeds_mask", "negative_p
embeds = text_encoder(prompt="dance monkey", negative_prompt="bad quality", max_sequence_length=16, output=text_outputs)
denoise_inputs = {"num_inference_steps": 2, "height": 32, "width": 32}

# 1. Only the canonical hand-offs: the masks are missing.
canonical = {k: v for k, v in embeds.items() if k in ("prompt_embeds", "negative_prompt_embeds")}
try:
    denoise(**canonical, **denoise_inputs, generator=torch.manual_seed(0), output="latents")
except ValueError as e:
    print("canonical hand-offs only:", e)

# 2. With the masks handed over: the standalone block has no VAE and falls back to the default scale factor.
latents = denoise(**embeds, **denoise_inputs, generator=torch.manual_seed(0), output="latent
expected = full(
    prompt="dance monkey",
    negative_prompt="bad quality",
    max_sequence_length=16,
    **denoise_inputs,
    generator=torch.manual_seed(0),
    output="latents",
)
print("vae_scale_factor  standalone denoise:", denoise.vae_scale_factor, "| assembled:", full.vae_scale_factor)
print("latents           standalone denoise:", tuple(latents.shape), "| assembled:", tuple(e
```

</details>

```
canonical hand-offs only: Required input 'prompt_embeds_mask' is missing
vae_scale_factor  standalone denoise: 8 | assembled: 4
latents           standalone denoise: (1, 4, 1, 4, 4) | assembled: (1, 4, 1, 8, 8)
```




> **Comment by sayakpaul (2026-11-17T14:31:03Z):**

Can you simplify the explanation and try to write it in your own words? The extra slop creates a bit of fatigue.


> **Comment by charchit7 (2026-11-17T15:52:52Z):**

Yeah, sorry my bad,

So, I ran your sketch on QwenImage with the tiny repo and two things came up in the chained test.

1. The test doesn't pass every output forward, it keeps only the names in handoff_names. In QwenImage the text encoder also gives `prompt_embeds_mask` and `negative_prompt_embeds_mask` to denoise. So the test drops the two masks and denoise fails with "Required input 'prompt_embeds_mask' is missing". To get past it I added the two names in the test subclass:

```python
handoff_names = ModularBlocksTesterMixin.handoff_names | {"prompt_embeds_mask", "negative_prompt_embeds_mask"}
```

Is that the intended way for pipelines that need extra handoffs? Please correct me if I am missing something. 

2. After I added the masks, the output was 16x16 instead of 32x32. The denoise block decides the latent size from vae_scale_factor so when it runs alone it has no vae, it uses the default 8. The tiny vae is 4. So the latents come out 4x4 instead of 8x8, and the decoded image is half the size. With the real model the vae is 8, so it doesn't show up there. A fallback like this was added on purpose for Anima in #14943  so the block can run without a vae, but it gives the wrong size when the vae is not 8.


> **Comment by charchit7 (2026-11-19T08:50:49Z):**

@sayakpaul what do you think, Please correct me if I am missing something.


> **Comment by mahmoudabdalhamid10-alt (2026-11-23T12:49:06Z):**

Hi @yiyixuxu @sayakpaul, I'd like to pick this up if you haven't decided who takes it yet.

My plan: add a tester mixin in tests/modular_pipelines/testing_utils/ that walks the top-level blocks of each workflow and checks that each block builds with init_pipeline(), loads only the components it declares, runs on its declared inputs and returns its declared outputs, and that running the blocks in sequence matches the assembled pipeline. I'd start by wiring it into one or two existing pipeline test classes, so you can review the shape before I move the rest over.

Disclosure: I'm using AI assistance for the code and I will review and test everything myself before opening a PR. I'll wait for your go-ahead before starting the PR.



### #14379 [closed] MiniMax-H3: `references` argument ignored and documented multi-GPU example fails with CUDA device mismatch
URL: https://github.com/huggingface/diffusers/issues/14379  by jyx13121802323 2026-08-04T08:03:56Z  comments: 7

### Describe the bug

## Environment

- OS: CentOS 7
- GPUs: 8 × NVIDIA A100 80GB
- Python: 3.11
- PyTorch: 2.6.0+cu124
- CUDA Driver: 550.xx (CUDA 12.4)
- Diffusers branches tested:
  - `minimax-h3`
  - `minimax-h3-refactor`

The same issues occur on both branches.

---

## Description

I tried to reproduce the MiniMax-H3 Reference-to-Video example from the documentation:

https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md

I encountered two independent issues.

---

## Issue 1: `references` argument is ignored

Calling the pipeline with

```python
state = pipe(
    prompt="...",
    references=[
        MiniMaxH3Reference(image=subject),
    ],
)
```

produces

```text
UserWarning:
Unexpected input 'dict_keys(['references'])' provided.
This input will be ignored.
```

As a result, all reference inputs are ignored by the pipeline.

This happens on both `minimax-h3` and `minimax-h3-refactor`.

---

## Issue 2: Official multi-GPU example fails with device mismatch

Following the official documentation, I place the Qwen3-VL text encoder on GPU1 while keeping the transformer on GPU0.

```python
import torch

from diffusers import ComponentsManager, ModularPipeline
from transformers import Qwen3VLForConditionalGeneration

manager = ComponentsManager()

pipe = ModularPipeline.from_pretrained(
    "MiniMaxAI/MiniMax-H3",
    components_manager=manager,
)

text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    "MiniMaxAI/MiniMax-H3",
    subfolder="text_encoder",
    dtype=torch.bfloat16,
    device_map={"": "cuda:1"},
)

pipe.update_components(
    text_encoder=text_encoder,
)

pipe.load_components(dtype=torch.bfloat16)

pipe.transformer.to("cuda:0")
pipe.vae.to("cuda:0")
pipe.audio_vae.to("cuda:0")
```

The pipeline initializes successfully, but fails during denoising with

```text
RuntimeError:
Expected all tensors to be on the same device,
but found at least two devices, cuda:1 and cuda:0
```

The traceback points to

```python
transformer_minimax_h3.py

rotary_emb = self.rope(position_ids)

freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(...)
```

Apparently,

- `position_ids` is located on `cuda:1`
- `self.inv_freq` is located on `cuda:0`

No automatic activation transfer happens between the text encoder and the transformer.

The full traceback ends with

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
```

---

## Expected behavior

According to the documentation,

1. `references` should be accepted by the pipeline.
2. The documented two-GPU loading strategy should work without additional user modifications.
3. Intermediate tensors produced by the text encoder should be transferred automatically (or otherwise handled) before entering the transformer.

---

## Questions

1. Is the current documentation ahead of the implementation?
2. Is the `references` argument currently supported in `ModularPipeline`?
3. Is the documented multi-GPU example expected to work, or is this feature still under development?
4. Is there an officially recommended way to run MiniMax-H3 on two 80GB GPUs without CPU offloading?

Thanks for the great work on MiniMax-H3 support. I'd be happy to test any proposed fixes or provide additional logs if needed.

### Reproduction

import torch

from diffusers import ComponentsManager, ModularPipeline
from transformers import Qwen3VLForConditionalGeneration

from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
from diffusers.utils import load_image, load_video
from diffusers.utils.export_utils import encode_video


# -------------------------
# Pipeline
# -------------------------

manager = ComponentsManager()

pipe = ModularPipeline.from_pretrained(
    "MiniMaxAI/MiniMax-H3",
    components_manager=manager,
)

print("Loading Qwen3-VL on GPU1...")

text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    "MiniMaxAI/MiniMax-H3",
    subfolder="text_encoder",
    dtype=torch.bfloat16,
    #device_map={"": "cuda:1"},
)

pipe.load_components(dtype=torch.bfloat16)

manager.enable_auto_cpu_offload(
    device="cuda",
    memory_reserve_margin="12GB",
)

pipe.update_components(
    text_encoder=text_encoder,
)

print("Loading remaining components...")

pipe.load_components(dtype=torch.bfloat16)

print("Moving transformer to GPU0...")

pipe.transformer.to("cuda:0")

pipe.vae.to("cuda:0")

pipe.audio_vae.to("cuda:0")

print("Pipeline Ready.")

subject = load_image(
    "extracted_images/img_row8_col3_12.png"
)

product = load_image(
    "extracted_images/img_row23_col3_48.png"
)


state = pipe(
    prompt="subject_definitions: <Subject 1> is the elderly female host in <Picture 1>, sitting in a cozy streaming setup with a dignified and wise appearance. <Subject 2> is the health supplement in <Picture 2>, a premium wellness product with trustworthy packaging and clear health-focused branding. summary: [reference generation] The target video shows <Subject 1> at her streaming desk in a fixed camera livestream, carefully picking up and presenting <Subject 2> to the audience with a caring and authoritative manner, emphasizing the product's health benefits. retention_analysis: <Subject 1>: fully_preserved - the host's dignified elderly appearance and wise demeanor are retained throughout. <Subject 2>: fully_preserved - the health product's trustworthy packaging and branding are retained. detailed_description: The target video is a fixed-camera livestream shot in a warm and authoritative health style with soft, warm lighting and a comfortable, home-like background. The camera remains stationary throughout. A medium shot shows <Subject 1>, the elderly woman, sitting at her streaming desk with a composed and dignified expression. She carefully picks up <Subject 2> from the display area. She holds the product steadily at a clear viewing height, presenting its label side to the camera. Her hands cradle the product with care, and her expression conveys deep personal endorsement. She nods gently, reinforcing the trustworthiness of her recommendation. overall_soundscape: Warm, quiet room tone with a gentle, calming atmosphere. non_diegetic_music: N/A",
    references=[
        MiniMaxH3Reference(image=subject),
        MiniMaxH3Reference(image=product),
    ],
    num_frames=124,
)

encode_video(
    state["videos"][0],
    fps=24,
    output_path="result.mp4",
    audio=state["audio"][0],
    audio_sample_rate=state["sampling_rate"],
)

### Logs

```shell

```

### System Info

- 🤗 Diffusers version: 0.40.0.dev0
- Platform: Linux-4.18.0-2.6.8.kwai.x86_64-x86_64-with-glibc2.17
- Running on Google Colab?: No
- Python version: 3.11.0
- PyTorch version (GPU?): 2.6.0+cu124 (True)
- Huggingface_hub version: 1.26.0
- Transformers version: 5.14.1
- Accelerate version: 1.14.0
- PEFT version: not installed
- Safetensors version: 0.8.0
- xFormers version: not installed
- Accelerator: NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
NVIDIA A100-SXM4-80GB, 81920 MiB
- Using GPU in script?: yes
- Using distributed or parallel set-up in script?: yes

### Who can help?

_No response_

> **Comment by Saad-Mallebhari (2026-08-06T17:58:38Z):**

Hi @jyx13121802323

Thanks for the detailed issue report and the exact reproduction snippet.

Root Cause

1. **References Kwarg Ignored:** In `src/diffusers/modular_pipelines/minimax_h3/pipeline_minimax_h3.py`, `references` is omitted from `ModularPipeline`'s accepted `__call__` schema whitelist, triggering a fallback warning that discards `references` inputs.
2. **Multi-GPU Device Mismatch:** In `src/diffusers/models/transformers/transformer_minimax_h3.py`, the RoPE rotary embedding forward pass computes `position_ids.unsqueeze(-1) * self.inv_freq` without aligning `position_ids.device` to `self.inv_freq.device`. When the text encoder resides on `cuda:1` and the transformer on `cuda:0`, cross-device arithmetic throws a PyTorch `RuntimeError`.

Proposed Solution

1. Add `references` to the allowed input arguments schema in `src/diffusers/modular_pipelines/minimax_h3/pipeline_minimax_h3.py`.
2. In `src/diffusers/models/transformers/transformer_minimax_h3.py`, ensure `position_ids` is moved to `self.inv_freq.device` (`position_ids = position_ids.to(self.inv_freq.device)`) prior to multiplying with `self.inv_freq`.
3. Add regression unit tests in `tests/pipelines/minimax_h3/test_minimax_h3.py` verifying device placement and `references` input processing.

Would you like me to open a Pull Request with this MiniMax-H3 multi-GPU device alignment and references fix with accompanying unit tests?


> **Comment by jyx13121802323 (2026-08-07T12:02:41Z):**

> 你好[@jyx13121802323](https://github.com/jyx13121802323)
> 
> 感谢您提供的详细问题报告和确切的复现代码片段。
> 
> 根本原因
> 
> 1. **References Kwarg Ignored:**在 中`src/diffusers/modular_pipelines/minimax_h3/pipeline_minimax_h3.py`，被从的接受模式白名单`references`中省略，从而触发了丢弃输入的回退警告。`ModularPipeline``__call__``references`
> 2. **多 GPU 设备不匹配：**`src/diffusers/models/transformers/transformer_minimax_h3.py`在RoPE 旋转嵌入前向传递过程中，计算时未`position_ids.unsqueeze(-1) * self.inv_freq`与目标对齐`position_ids.device`。`self.inv_freq.device`当文本编码器位于目标设备上，`cuda:1`而转换器位于目标设备上时`cuda:0`，跨设备运算会引发 PyTorch 异常`RuntimeError`。
> 
> 建议的解决方案
> 
> 1. `references`在允许的输入参数架构中添加`src/diffusers/modular_pipelines/minimax_h3/pipeline_minimax_h3.py`。
> 2. 在 中`src/diffusers/models/transformers/transformer_minimax_h3.py`，确保在与 相乘之前`position_ids`移到`self.inv_freq.device`( ) 。`position_ids = position_ids.to(self.inv_freq.device)``self.inv_freq`
> 3. `tests/pipelines/minimax_h3/test_minimax_h3.py`在验证设备放置和`references`输入处理的过程中，增加回归单元测试。
> 
> 您是否希望我提交一个 Pull Request，其中包含 MiniMax-H3 多 GPU 设备对齐和引用修复以及相应的单元测试？

Thanks for looking into this issue.

I did some additional investigation and confirmed that the two problems appear to be independent.

For the `references` issue, the warning

```text
Unexpected input 'dict_keys(['references'])' provided.
This input will be ignored.
```

is reproducible on both `minimax-h3` and `minimax-h3-refactor`. So at least with the current ModularPipeline implementation, `references` does not seem to be wired into the pipeline input schema, even though it is used in the MiniMax-H3 documentation example.

For the multi-GPU issue, I believe the problem is specifically caused by the device boundary between the Qwen3-VL text encoder and the MiniMax-H3 transformer.

With:

```python
text_encoder -> cuda:1
transformer -> cuda:0
vae/audio_vae -> cuda:0
```

the text encoder itself runs successfully, but the denoising stage eventually receives tensors from different devices:

```text
position_ids -> cuda:1
transformer.inv_freq -> cuda:0
```

and fails at:

```python
freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(...)
```

I also confirmed that this is not specific to a particular A100 device. The same behavior occurs on my 8 × A100 80GB machine with:

* PyTorch 2.6.0+cu124
* CUDA 12.4
* Diffusers 0.40.0.dev0
* Python 3.11

One important point is that simply placing the text encoder on another GPU is not sufficient unless the intermediate tensors produced by the text encoder are explicitly moved to the transformer device (or the pipeline provides an automatic device-transfer mechanism).

So I think the documented example currently assumes some cross-device activation handling that is not actually implemented in this code path.

If useful, I can test a proposed fix. In particular, I can verify whether explicitly moving the relevant tensors/`position_ids` to the transformer device before the RoPE computation resolves the issue without introducing additional device mismatches later in the denoising loop.

It would also be helpful to know whether the intended design is:

```text
GPU1: Qwen3-VL
        |
        | activation transfer
        v
GPU0: MiniMax-H3 Transformer
        |
        v
GPU0: VAE / Audio VAE
```

or whether ModularPipeline is expected to handle this device placement automatically.

Thanks!



> **Comment by Saad-Mallebhari (2026-08-07T14:52:44Z):**

In `ModularPipeline`, individual component modules align incoming activations to their execution device (`self.device` / `self.inv_freq.device`) upon entry in their `forward()` pass. Rather than `ModularPipeline` managing top-level cross-device transfers, components move input tensors to their target device upon entry.

Here's what to update if you want to test it on your A100 setup:

### 1. Multi-GPU Device Alignment
In `src/diffusers/models/transformers/transformer_minimax_h3.py`, align `position_ids` to `self.inv_freq.device` prior to the RoPE frequency computation:
```python
position_ids = position_ids.to(self.inv_freq.device)
```

### 2. References Kwarg Schema Wiring
In `src/diffusers/modular_pipelines/minimax_h3/pipeline_minimax_h3.py`, add `references` to the accepted input arguments schema of `__call__` so `ModularPipeline` passes `references` through without discarding them.

Applying these two updates will enable both `references` and the two-GPU loading workflow.


> **Comment by jyx13121802323 (2026-08-09T14:11:20Z):**

> 在[此处应填写具体实现方式]中，各个组件模块在进入其通道时，`ModularPipeline`会将传入的激活值与其执行设备（`self.device`/ ）对齐。组件无需管理顶层跨设备传输，而是在进入时将输入张量移动到目标设备。`self.inv_freq.device``forward()``ModularPipeline`
> 
> 如果您想在 A100 设备上进行测试，请按以下步骤更新：
> 
> ### 1. 多GPU设备对齐
> 在RoPE 频率计算之前`src/diffusers/models/transformers/transformer_minimax_h3.py`，进行对齐`position_ids`：`self.inv_freq.device`
> 
> position_ids = position_ids.to(self.inv_freq.device)
> ### 2. 参考 Kwarg Schema Wiring
> 在 中`src/diffusers/modular_pipelines/minimax_h3/pipeline_minimax_h3.py`，添加`references`到 的接受输入参数模式中，`__call__`这样`ModularPipeline`就可以`references`在不丢弃它们的情况下通过。
> 
> 应用这两项更新将同时启用`references`双 GPU 加载工作流程。

Thanks for the suggestion. I tried the proposed fix by moving `position_ids` to the same device as `self.inv_freq`:

```python
def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = position_ids.to(self.inv_freq.device)
    freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
    freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
    freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()
```

This appears to resolve the original device mismatch in the RoPE computation. However, the multi-GPU example still fails later during the forward pass:

```text
Error in block: (MiniMaxH3Blocks)

RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:1 and cuda:0!
(when checking argument for argument weight in method wrapper_CUDA__cudnn_convolution)
```

This suggests that there is another cross-device tensor mismatch after the RoPE computation. In particular, the error now occurs inside `MiniMaxH3Blocks` during a convolution, which likely means that the convolution input activation is on `cuda:1` while the convolution weight is on `cuda:0`.

My current setup is:

* 8 × NVIDIA A100 80GB
* PyTorch 2.6.0+cu124
* CUDA driver 550.54.14
* Python 3.11
* Diffusers `minimax-h3-refactor`
* Qwen3-VL on `cuda:1`
* MiniMax-H3 transformer/VAE/audio VAE on `cuda:0`

So the RoPE fix seems to expose a second device mismatch rather than fully solving the multi-GPU issue.

Would it be more appropriate to handle the device transfer at the boundary between the text encoder and the MiniMax-H3 transformer/denoising step, rather than inside the RoPE implementation? I suspect that other activations produced by Qwen3-VL may also remain on `cuda:1` when entering the transformer on `cuda:0`.

I can provide the full traceback or add debugging output showing the device of the convolution inputs and parameters if that would be useful.



> **Comment by Saad-Mallebhari (2026-08-10T10:56:35Z):**

Awesome, glad to hear the RoPE fix got past that first crash!

You're spot on about the architecture. In Diffusers, ModularPipeline doesn't manage top-level activation transfers between modules. Instead, the design pattern is that individual models (like the transformer) align all incoming activations to their own device at the very top of their `forward()` method.

That second crash in the convolution block happens because the text embeddings (`encoder_hidden_states`) produced by Qwen3-VL on `cuda:1` enter the transformer on `cuda:0` while still sitting on `cuda:1`. When the transformer blocks execute, the convolution weights are on `cuda:0` while the input tensor is on `cuda:1`, which triggers the PyTorch device mismatch. (No need to worry about sending extra tracebacks or debug logs—the convolution error message already confirms this exact input tensor mismatch!).

To fix this, you just need to align `encoder_hidden_states` right at the entry of `forward()` in `src/diffusers/models/transformers/transformer_minimax_h3.py`:

```python
if encoder_hidden_states is not None and encoder_hidden_states.device != self.device:
    encoder_hidden_states = encoder_hidden_states.to(device=self.device)
```

And for the `references` argument being ignored, adding `"references"` to the accepted `__call__` schema whitelist in `src/diffusers/modular_pipelines/minimax_h3/pipeline_minimax_h3.py` will stop ModularPipeline from discarding it.

Give that a shot on your A100 setup it should handle the full multi-GPU denoising loop smoothly!


> **Comment by chakshu-dhannawat (2026-08-27T05:54:04Z):**

Hi, I'd like to pick up the multi-GPU device-mismatch part of this issue.

While building a repro against current main I noticed that `references` is already wired into `MiniMaxH3Blocks.inputs`, so the remaining blocker is the cross-device transfer between the Qwen3-VL text encoder and the MiniMax-H3 transformer.

Root cause. When the text encoder sits on `cuda:1` and the transformer on `cuda:0`, two inputs reach the transformer on the wrong device:

- `encoder_hidden_states` is used directly by `self.context_embedder`, whose weights are on `cuda:0`.
- `position_ids` is multiplied with `self.rope.inv_freq`, which is on `cuda:0`.

Both currently raise `RuntimeError: Expected all tensors to be on the same device`.

Proposed fix. Two small alignment points in `transformer_minimax_h3.py`:

1. In `MiniMaxH3RotaryPosEmbed.forward`, move `position_ids` to `self.inv_freq.device` before computing frequencies.
2. At the top of `MiniMaxH3Transformer3DModel.forward`, move `encoder_hidden_states` to `self.device` if it differs.

This keeps the fix scoped to the transformer entry point and mirrors the ModularPipeline pattern of each component aligning its own inputs.

Verification. I added a multi-GPU regression test that puts `encoder_hidden_states` and `position_ids` on `cuda:1` while the model is on `cuda:0`, and checks that the output matches the same-device baseline:

```bash
python -m pytest tests/models/transformers/test_models_transformer_minimax_h3.py::TestMiniMaxH3TransformerMultiGPU::test_cross_device_text_encoder_forward -v
# 1 passed
```

Full transformer suite is green except for pre-existing failures in `test_model_parallelism` and the context-parallel NCCL tests, which also fail on unmodified main in this environment.

AI-assisted contribution disclosure: I used an AI agent for code navigation and drafting, and verified the diff and test output locally.

If this direction sounds right, I will open a PR from my branch. cc @sayakpaul @DN6



> **Comment by yiyixuxu (2026-11-26T02:44:30Z):**

@jyx13121802323 can we close the issue now since it was reported against the PR while it was WIP? I think both issues are fixed in `main` no?



### #14779 [open] MLU backend utilities fail despite MLU device detection
URL: https://github.com/huggingface/diffusers/issues/14779  by SeptPonts 2026-09-15T10:04:43Z  comments: 5

### Describe the bug

MLU device detection was added in #12629, but the backend utility tables, `training_utils.free_memory()`, and test-device selection still omit MLU.

On an MLU machine, `get_device()` returns `"mlu"`, while `backend_empty_cache("mlu")` raises `TypeError`. The same problem affects synchronization, both peak-memory reset helpers, device count, and peak allocated-memory queries. An existing caller is the [SD3 ControlNet training example](https://github.com/huggingface/diffusers/blob/c5469b7ceb606edd7ba6570dcd17d38590a18db6/examples/controlnet/train_controlnet_sd3.py#L122-L124), which passes `accelerator.device.type` to `backend_empty_cache` after prompt encoding.

I would like to submit a small follow-up to #12629 covering three files:

- `src/diffusers/utils/torch_utils.py`: register MLU operations in the existing backend tables.
- `src/diffusers/training_utils.py`: release unused MLU allocator cache in `free_memory()`.
- `tests/testing_utils.py`: select an available MLU automatically and register its test backend operations.

The existing H3 training test runs successfully on MLU when the device is selected explicitly. The missing test-device selection currently makes it run on CPU by default on this machine.

### Reproduction

Run with Diffusers and a working `torch_mlu` installation on an MLU machine:

```python
import torch
import torch_mlu

from diffusers.utils.torch_utils import backend_empty_cache, get_device

print(torch.mlu.is_available())
print(get_device())
backend_empty_cache("mlu")
```

### Logs

```text
True
mlu
TypeError: 'NoneType' object is not callable
```

I compared the baseline with a local patch on the same device:

| Check | Baseline | Local patch |
| --- | --- | --- |
| Six production cache/sync/reset/count/memory helpers with `"mlu"` | All raise `TypeError` | All succeed; count and allocated-memory results match direct MLU calls |
| `free_memory()` after allocating and deleting a 64 MiB MLU tensor | Allocated: 0; reserved: 67,108,864 bytes | Allocated: 0; reserved: 0 bytes |
| Existing `TestMiniMaxH3TransformerTraining::test_training`, default device | Passes on CPU | Passes on `mlu:0` |
| Same baseline test with `DIFFUSERS_TEST_DEVICE=mlu` | Passes on `mlu:0` | — |

For the cache check, calling `torch.mlu.empty_cache()` directly also releases the reserved memory on the baseline. The SD3 ControlNet example above is a source-level caller of the reproduced failing helper; I have not run that complete training example.

### System Info

- Device: Cambricon MLU590-M9DK, one visible device.
- PyTorch: `2.12.1+cpu`; `torch_mlu`: `1.34.1+torch2.12.1`.
- Accelerate: `1.14.0`; Transformers: `5.14.1`; PEFT: `0.18.1`.
- Tested Diffusers baseline: `c5469b7ceb606edd7ba6570dcd17d38590a18db6`.
- Also checked current main, `759164b7ad116e091e9d3e222211c9aa27d835f6`: these three files are unchanged from the tested baseline.

### Who can help?

@sayakpaul, would this scope be welcome as a follow-up PR? This is AI-assisted work, and I am opening the issue first to coordinate as requested in CONTRIBUTING.md. I checked related open and closed PRs; #13824 addresses Z-Image RoPE compatibility and #14039 adds TPU support, neither covers these MLU utility gaps.


> **Comment by L4XB (2026-09-15T15:40:53Z):**

I have a fix ready for this and, per `CONTRIBUTING`, I am asking for a maintainer ack on the issue before opening the PR.

Confirmed the diagnosis and found three more places with the same gap that the report does not name.

**Why it raises rather than falling back.** `_device_agnostic_dispatch` in `utils/torch_utils.py` reaches for the `"default"` entry when a device is missing from a table and calls it *without* the callable check it applies to known devices. So `BACKEND_EMPTY_CACHE["default"]` is `None` → `TypeError: 'NoneType' object is not callable`, and `BACKEND_DEVICE_COUNT["default"]` is the literal `0` → `TypeError: 'int' object is not callable`. Every table there has keys `cuda / xpu / cpu / mps / neuron / default`, and `get_device()` can return `mlu`.

`BACKEND_MANUAL_SEED` is the quiet one: it does not raise, because `"default"` is `torch.manual_seed`. It just seeds the CPU generator only, so an MLU run is silently unseeded. `BACKEND_SUPPORTS_TRAINING` already resolves `True` via the default; registering it explicitly changes nothing.

**Three more, beyond the tables:**

1. `training_utils.free_memory()` walks cuda → mps → npu → xpu and stops, so the MLU allocator cache is never released — which matches your measurement of reserved staying at 67,108,864 bytes.
2. `empty_device_cache()` and `device_synchronize()` resolve the module with `getattr(torch, device_type, torch.cuda)`. `torch.mlu` only exists once `torch_mlu` has been imported, and diffusers never imports it — so on a process that had not imported it by hand, these two silently ran on **CUDA**. Importing `torch_mlu` under the availability guard (the same thing `training_utils.py` already does for `torch_npu`) closes that.
3. `tests/testing_utils.py` keeps its own copy of the seven tables and its own `_device_agnostic_dispatch`, which *does* apply the callable check to the default entry — so there the helpers do not raise, they quietly return `None`/`0`. Its device selection goes cuda → xpu → neuron → cpu, which is why your MLU box runs the suite on CPU.

The registration block would mirror the existing `if _neuron_device is not None:` block in `tests/testing_utils.py`, and matches how transformers registers the same backend (`if is_torch_mlu_available(): BACKEND_EMPTY_CACHE["mlu"] = torch.mlu.empty_cache`, …).

**Three things I deliberately would not touch, and would like your call on:**

- `src/diffusers/utils/testing_utils.py` carries a **third** copy of these tables. It emits a `FutureWarning` on import, is scheduled for removal in 1.0.0, and its deprecation message points at `diffusers.utils.torch_utils`. It is already missing the `neuron` rows that `tests/testing_utils.py` has, so adding MLU there deepens a copy on its way out. Say the word if you would rather keep the three in step.
- `training_utils.set_seed()` has an NPU branch and otherwise calls `torch.cuda.manual_seed_all`, a no-op without CUDA — so on MLU the accelerator RNG is left unseeded. One line to fix, but I have no MLU to confirm `torch.mlu.manual_seed_all` exists, and guessing wrong turns a silent no-op into an `AttributeError`.
- **NPU has the identical gap** in `utils/torch_utils.py`: `get_device()` can return `"npu"` and no table has an `npu` key. Different vendor API (`torch_npu.npu.*`, not `torch.npu.*`, as `free_memory` shows) and I cannot test it, so it belongs in a separate change by someone who can.
- `_FP64_UNSUPPORTED_DEVICES` / `_INT64_UNSUPPORTED_DEVICES` list `mps`, `npu`, `neuron`. Whether MLU belongs there is a hardware question I cannot answer.

**On evidence:** I do not have MLU hardware, so I cannot paste a real before/after from one. What I can show is the dispatch path and a `torch.mlu`-shaped stub proving each helper resolves to the right callable after registration, plus the unchanged behaviour on CPU/CUDA. If you would rather have this confirmed on real hardware first, the reporter is in a position to run it — happy to wait for that before the PR.

Let me know if the shape is right and I will open it.



> **Comment by L4XB (2026-09-15T16:35:36Z):**

Correcting myself: I said three of those were places "the report does not name", and two of them are
in fact in your own three-file plan — `training_utils.free_memory()` and `tests/testing_utils.py` are
both listed in the issue. Your scope was right; I misread it. Sorry.

What is actually beyond it is one thing, plus one detail inside the tables:

- `empty_device_cache()` and `device_synchronize()` in `utils/torch_utils.py` resolve the module with
  `getattr(torch, device_type, torch.cuda)`. Since diffusers never imports `torch_mlu`, `torch.mlu`
  is absent unless the caller imported it by hand — so those two silently ran on **CUDA** rather than
  raising. Importing `torch_mlu` under the availability guard (what `training_utils.py` already does
  for `torch_npu`) fixes that as a side effect of your fix.
- `BACKEND_MANUAL_SEED` is the quiet one in the tables: it does not raise, because `"default"` is
  `torch.manual_seed`, so an MLU run just leaves the accelerator generator unseeded.

Everything else in my earlier comment stands, including the two I would not touch without hardware
(`set_seed()`'s missing `torch.mlu.manual_seed_all`, and the identical NPU gap) and the third,
deprecated copy of the tables in `src/diffusers/utils/testing_utils.py`.

One thing I should have said plainly the first time: **I do not have an MLU**, so four of the eight
`torch.mlu.*` names in the branch — `synchronize`, `reset_peak_memory_stats`,
`reset_max_memory_allocated`, `max_memory_allocated` — rest entirely on your measurement. The other
four match how transformers registers the same backend. If a maintainer would rather have those four
written as `getattr(torch.mlu, ..., None)`, which is what the tables already do for the XPU
equivalents, that is a one-line change.



> **Comment by sayakpaul (2026-09-15T19:08:10Z):**

@SeptPonts the changes look reasonable to me. Feel free to open the PR. @L4XB thanks for offering to help, however, since @SeptPonts already laid down a solid approach, I think they deserve the chance better. I hope we will have future opportunities for contributions.


> **Comment by L4XB (2026-09-15T19:23:14Z):**

Completely fair, and the right call — @SeptPonts did the measurement on real hardware, which is the part I could not do. Looking forward to the PR.

Everything I found is in the thread above for them to take or leave, in particular the two beyond the three-file scope: `empty_device_cache()` / `device_synchronize()` resolving through `getattr(torch, device_type, torch.cuda)` and therefore silently running on **CUDA** unless something has already imported `torch_mlu`, and `BACKEND_MANUAL_SEED` falling through to `torch.manual_seed` so the accelerator generator is never seeded. Neither raises, which is why they are easy to miss next to the six that do.

@SeptPonts — happy to answer anything about those two if it helps, and no need to credit me.



> **Comment by SeptPonts (2026-09-16T06:23:34Z):**

Thanks @sayakpaul! I'll open a PR with the scope outlined above and include the before/after validation results from MLU hardware.
Thanks @L4XB for taking a look and sharing your findings.



### #15022 [open] Modular pipelines: warn when a `denoiser_input_fields` entry is not passed to the transformer
URL: https://github.com/huggingface/diffusers/issues/15022  by yiyixuxu 2026-10-13T19:36:59Z  comments: 4

The loop denoisers forward the fields tagged `denoiser_input_fields` whose name matches the transformer's forward signature and drop the rest silently, e.g. in `qwenimage/denoise.py`:

```python
transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
for field_name, field_value in block_state.denoiser_input_fields.items():
    if field_name in transformer_args and field_name not in guider_inputs:
        additional_cond_kwargs[field_name] = field_value
```

So a custom block tagging a field under a name the transformer does not take (a typo, or a field meant for another transformer) runs without any error and the conditioning never reaches the model. Same pattern in the `helios`, `wan`, `wan_animate_2`, `ltx2` and `minimax_h3` denoisers.

Proposal: work on a copy of the dict, pop the fields the block consumes by name and the ones matching the signature, and `logger.warning_once` whatever is left:

```
`denoiser_input_fields` not passed to the transformer: ['typo_field']
```

Two things found while trying this out:

- QwenImage's edit-plus and layered RoPE input steps still tag `txt_seq_lens` / `negative_txt_seq_lens` for the denoiser, but the transformer no longer takes them since #13674.
- diffusers' logger has no `warning_once` of its own. The existing `logger.warning_once` calls only work because transformers patches `logging.Logger` when it is imported.

---
*Posted by Claude Code on behalf of @yiyixuxu, who reviewed and approved this text.*


> **Comment by PrakshaaleJain (2026-10-15T13:17:59Z):**

Hi @yiyixuxu, if you like, I can work on this issue.


> **Comment by its-amann (2026-10-26T11:40:34Z):**

Hi! I'd like to work on this if it's still open. @PrakshaaleJain, I saw you offered earlier, so happy to step back if you've already started. @yiyixuxu, could you assign it to me? I'll follow up with a PR that includes the fix and tests.


> **Comment by its-amann (2026-10-29T11:38:43Z):**

Hi @yiyixuxu, just following up on this. I'm still keen to take it if it's open for contributors.


> **Comment by RudraBhaskar9439 (2026-11-20T16:18:48Z):**

Hi @yiyixuxu, I have a local fix and regression tests for the missing denoiser warning. I saw that two contributors offered to work on this earlier. If the issue is still available, would you be OK with me opening a PR?




### #14773 [open] LTX2 import fails with Transformers 5.5.0 due to annotation-only Gemma4Unified dependency
URL: https://github.com/huggingface/diffusers/issues/14773  by Imagineer99 2026-09-14T15:54:51Z  comments: 4

### Describe the bug

With Diffusers 0.40.0 and Transformers 5.5.0, importing `LTX2Pipeline` fails because `Gemma4UnifiedForConditionalGeneration` is unavailable. This blocks Gemma3-based LTX 2.3 use before any checkpoint is loaded. We encountered this in Unsloth Desktop, but the reproduction does not import Unsloth.

The unconditional import is also present on current [main](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2.py). In the five affected pipeline modules in 0.40.0, the missing class is used only in the `text_encoder` constructor annotation.

### Reproduction

In an environment with PyTorch, `diffusers==0.40.0`, and `transformers==5.5.0`:

```python
from diffusers import LTX2Pipeline
```

`LTX2ImageToVideoPipeline` fails similarly. No model download, GPU generation, or authentication is required to trigger this error.

### Logs

```text
RuntimeError: Failed to import diffusers.pipelines.ltx2.pipeline_ltx2 because of the following error (look up to see its traceback):
cannot import name 'Gemma4UnifiedForConditionalGeneration' from 'transformers'
```

### Proposed scope and validation

Would you accept a PR replacing the architecture-specific `text_encoder` annotation with `transformers.PreTrainedModel` and removing the unused Gemma imports across these five modules?

- `pipeline_ltx2.py`
- `pipeline_ltx2_image2video.py`
- `pipeline_ltx2_condition.py`
- `pipeline_ltx2_ic_lora.py`
- `pipeline_ltx2_hdr_lora.py`

An isolated candidate preserves Diffusers 0.40.0 and Transformers 5.5.0, with no changes to generation logic. Validation so far:

- Original fails with the error above on Windows and WSL; the candidate completes tiny-fixture text-to-video and image-to-video inference, including audio decoding and local save/reload, on both.
- On a separate Transformers 5.17.0 test stack, original and patched pipelines produce bit-for-bit identical video and audio tensors for those same tiny fixtures. The real Gemma4Unified class remains accepted by the annotation.
- All five affected pipeline imports/signature checks and MiniMax-H3/Music-3 modular pipeline imports pass.

The runtime checks use the upstream LTX tiny-component fixture and `hf-internal-testing/tiny-gemma3`. Full LTX 2.3 and MiniMax checkpoint generation have not been tested.

This investigation and candidate were AI-assisted. Following the contribution guidelines, I am opening this issue for maintainer acknowledgment before a PR, and am happy to adjust the approach or add repository-native regression coverage.

### System Info

Reproduced on both:

| | Windows | WSL2 Ubuntu |
|---|---|---|
| Diffusers | 0.40.0 | 0.40.0 |
| Transformers | 5.5.0 | 5.5.0 |
| Python | 3.13.5 | 3.13.12 |
| PyTorch | 2.10.0+cu130 | 2.11.0+cu130 |

Additional WSL environment: huggingface_hub 1.30.0, Accelerate 1.14.0, PEFT 0.18.1, safetensors 0.8.0. NVIDIA RTX 5070 Ti is available; the reproduction is import-only and does not use distributed execution.


> **Comment by L4XB (2026-09-15T15:40:52Z):**

I have a fix ready for this and, per `CONTRIBUTING`, I am asking for a maintainer ack on the issue before opening the PR.

**Two corrections to the scope first.**

1. It is **seven** modules, not five. `grep -rn "Gemma4Unified" --include="*.py" .` on `main` @ `759164b` matches only `src/diffusers/pipelines/ltx2/`, and in every one the symbol appears just in the `from transformers import (...)` block and the `text_encoder` annotation/docstring:

   `pipeline_ltx2.py`, `pipeline_ltx2_image2video.py`, `pipeline_ltx2_condition.py`, `pipeline_ltx2_ic_lora.py`, `pipeline_ltx2_hdr_lora.py`, `pipeline_ltx2_dfr.py`, `pipeline_ltx2_dfr_temporal_refine.py`

   The last two are newer than 0.40.0, which is why the report lists five.

2. It is **nine** public entry points, not seven. Resolving every name in `diffusers.__all__` on transformers 5.5.0:

   ```
   transformers 5.5.0, diffusers 0.41.0.dev0 @ 759164b
   before:  resolved OK: 693   failed: 9
   after:   resolved OK: 702   failed: 0
   ```

   The two extra are `LTX2AutoBlocks` and `LTX25AutoBlocks`, which never name the symbol themselves — `modular_blocks_ltx2.py` → `before_denoise.py` → `pipeline_ltx2_ic_lora.py` carries it in. Fixing the seven source modules clears all nine.

**The boundary is 5.9.0 → 5.10.0.** `src/transformers/models/gemma4_unified` is absent at tag `v5.9.0` and present at `v5.10.0`; `from transformers import Gemma4UnifiedForConditionalGeneration` raises on 5.5.0 and 5.9.0 and succeeds on 5.10.0/5.10.1/5.11.0/5.17.0. (5.10.0 is yanked on PyPI, so any table below uses 5.10.1, but the guard says `>= 5.10.0` because that is where the symbol appears.)

**The shape I would like to confirm before writing it up.** Bind the name to `transformers.PreTrainedModel` and re-import it behind `is_transformers_version(">=", "5.10.0")` in each affected module — the same pattern `pipelines/glm_image/pipeline_glm_image.py` already uses for `GlmImageForConditionalGeneration`. Repeated per file rather than factored into a shared helper, per the single-file policy in `PHILOSOPHY.md`.

I did consider the option floated in the issue — drop the import and annotate `text_encoder` as `PreTrainedModel` — but it is a behaviour change, not a cosmetic one. `DiffusionPipeline._get_signature_types()` reads that annotation, and `loaders/single_file.py::_map_component_types_to_config_dict` uses `component_value[0].__name__` to build the inferred `model_index.json` entry. Collapsing the union would turn `["transformers", "Gemma3ForConditionalGeneration"]` into `["transformers", "PreTrainedModel"]` for pipelines that carry `FromSingleFileMixin`. A `TYPE_CHECKING` block has the same problem unless every module also gets `from __future__ import annotations`, since the annotation is evaluated at class-definition time.

With the guard, introspection is byte-identical wherever the symbol exists (checked on 5.17.0, all seven pipelines, both `_get_signature_types()["text_encoder"]` and the single-file inferred config); on 5.5.0 the union becomes `(Gemma3ForConditionalGeneration, PreTrainedModel)` and the single-file entry is unchanged.

**Test.** `tests/others/test_dependencies.py::test_ltx2_pipelines_import_without_gemma4_unified` hides the symbol and reports an older transformers in a subprocess, then resolves all nine entry points. It sits next to `test_pipeline_module_imports`, which exists for the same class of bug but can only catch it on an environment that actually has the older dependency. Run on transformers 5.17.0 so it discriminates on the version CI installs:

```
merge base 759164b : 1 failed, 4 deselected in 3.29s
this branch        : 1 passed, 4 deselected in 3.65s
```

`pytest tests/modular_pipelines/ltx2` is unchanged: 16 failed / 140 passed / 18 skipped on both sides, the same CPU-offload tests.

Happy to open the PR if this shape is what you want, or to switch to the annotation change if you would rather take the `model_index.json` consequence.



> **Comment by L4XB (2026-09-15T16:36:15Z):**

Correcting my own comment above: the scope is bigger than I said, in a way that matters for the ack.

I had swept `diffusers.__all__` only at transformers 5.5.0, the version in the report. Sweeping the
rest of the 5.x range turns up a **second** class imported exactly the same way:
`Gemma4ForConditionalGeneration` (no "Unified"), added in transformers **5.5.0**, imported
unconditionally by five of the seven modules to annotate the optional `prompt_enhancer`. It raises
first in the same import tuple, so on transformers 5.0.0–5.4.x guarding only `Gemma4Unified` would
have left this issue's symptom exactly where it is.

```
                                    main @ 759164b        branch
transformers 5.0.0    691 ok / 11 failed        700 ok /  2 failed
transformers 5.4.0    693 ok /  9 failed        702 ok /  0 failed
transformers 5.5.0    693 ok /  9 failed        702 ok /  0 failed
transformers 5.9.0    693 ok /  9 failed        702 ok /  0 failed
```

Both boundaries are measured: `src/transformers/models/gemma4` is absent at tag `v5.4.0` and present
at `v5.5.0`; `gemma4_unified` is absent at `v5.9.0` and present at `v5.10.0`. The branch now guards
both, and introspection stays byte-identical on 5.17.0 — `_get_signature_types()` for both
`text_encoder` and `prompt_enhancer`, and the full `_infer_pipeline_config_dict()` mapping, for all
seven pipelines.

**The 2 that remain at 5.0.0 are the same pattern outside LTX2, and I would like your call on them.**
`pipelines/motif_video/pipeline_motif_video.py` and `pipeline_motif_video_image2video.py` import
`T5Gemma2Encoder` (transformers 5.1.0) the same way, and already carry a comment saying so:

```python
# NOTE: This pipeline requires transformers>=5.1.0 for T5Gemma2Encoder support.
```

Same one-line fix, but a different pipeline family whose authors clearly knew, and unlike LTX2 that
pipeline is genuinely unusable without the class rather than merely un-importable — so I did not
widen someone else's report into their code on my own. Happy to fold it into the same PR so the sweep
comes back clean at 5.0.0, or to leave it out.

Test is unchanged in shape, just widened to hide both symbols and report transformers 5.4.0:

```
tests/others/test_dependencies.py::…::test_ltx2_pipelines_import_on_older_transformers
  main @ 759164b (test file only) : 1 failed — cannot import name 'Gemma4ForConditionalGeneration'
  branch                          : 5 passed  (whole file)
```

Still asking before opening: is the two-symbol scope what you want, and does `motif_video` go in it?



> **Comment by noy-solvin (2026-11-19T09:37:19Z):**

Hi @sayakpaul, @yiyixuxu, @apolinario, @stevhliu
I've submitted PR #15175 that fixes this issue.
Would love a review whenever you get a chance. Happy to adjust anything!


> **Comment by yiyixuxu (2026-11-26T04:31:52Z):**

@Imagineer99 
thanks for reporting! and sorry for the slow response 
would you be able to open a PR? we could follow the pattern in glm-image: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/glm_image/pipeline_glm_image.py#L40 

if not, let us know, we will send a PR and add you ad co-author 



### #15014 [closed] gfx1201 (RDNA4) SageAttention attention backend: aiter package or Hub kernel?
URL: https://github.com/huggingface/diffusers/issues/15014  by Shan2L 2026-10-12T05:32:38Z  comments: 4

**Is your feature request related to a problem? Please describe.**

On AMD Radeon RDNA4 GPUs (gfx1201: Radeon AI PRO R9700 / R9600D, RX 9070 series), diffusers has no quantized attention backend:

- `SAGE` and `SAGE_HUB` are CUDA builds.
- The ROCm Hub FlashAttention kernel (`kernels-community/aiter-flash-attn`) ships tuning configs for gfx942 / gfx950 / gfx1250 only.

So these GPUs use native SDPA.

aiter has a SageAttention kernel for gfx1201: `aiter.gfx1201_sage_attention(q, k, v, softmax_scale)`, ROCm/aiter#6390 (under review).

- INT8 QK^T and FP8 PV with FP32 accumulation; Q/K are quantized per 32-row block and V per channel inside the kernel.
- Inputs: BF16 `[B, S, H, 128]`, Q/K/V of equal shape, no mask, non-causal.

MiniMax-H3 in diffusers matches that contract exactly. `MiniMaxH3AttnProcessor` passes no mask, the attention is self-attention with head dim 128, and the input is BF16. The same applies to the token refiner.

For reference, the same kernel as an opt-in vLLM-Omni backend (vllm-project/vllm-omni#8792, draft) on MiniMax-H3 T2VA: 2 x R9700 TP2, 1344x768, 4 s, 8 steps.

| Backend | Denoise per step | Request latency | Video PSNR vs SDPA |
| --- | ---: | ---: | ---: |
| SDPA | 20.9 s | 186 s | reference |
| gfx1201 SageAttention | 11.3 s | 109 s | 33.9 dB |

**Describe the solution you'd like.**

An opt-in diffusers attention backend for gfx1201 SageAttention. It would never be selected by default, and it would reject unsupported inputs (masks, causal, head dim other than 128, unequal Q/K/V shapes) with a clear error.

It would support Ulysses context parallelism (inference only, no LSE needed), so MiniMax-H3 can run with its existing `_cp_plan` on 2+ GPUs.

Before writing it, I'd like to check which delivery you prefer, given that #14436 moved ROCm FlashAttention from the `aiter` pip package to a Hub kernel:

1. **`AITER_SAGE`, using the `aiter` package.** The backend imports `aiter` and calls the op. It is a small diffusers change, but it adds back an optional `aiter` dependency and needs an aiter release that contains ROCm/aiter#6390.
2. **`AITER_SAGE_HUB`, a Hub kernel.** We publish the kernel as a `kernels` package. Like `kernels-community/aiter-flash-attn`, it would be a universal `torch-rocm` build: precompiled gfx1201 code objects plus a small Python loader registered as a torch custom op. diffusers would only add a `_HubKernelConfig` entry and the backend function. This does not depend on an aiter release. The open question is which Hub repo it should live in.
3. **A ROCm variant of `kernels-community/sage-attention`**, so the existing `SAGE_HUB` works unchanged. Its `sageattn` would route gfx1201 inputs it supports to this kernel and the rest to SDPA.

**Describe alternatives you've considered.**

- Keep SDPA on gfx1201. That leaves about 1.8x on MiniMax-H3 attention unused.
- A custom attention processor in user code. It works, but every model needs its own processor, and it does not compose with `set_attention_backend` or context parallelism.

**Additional context.**

I can implement whichever option you prefer, with tests and a MiniMax-H3 end-to-end comparison against SDPA on 2 x R9700 using Ulysses CP. A sparse variant (Sol-Attn, ROCm/aiter#6292) also exists. Following the note in `attention_dispatch.py` about per-model-tuned sparse methods, I am not proposing it here.

cc @sayakpaul

AI assistance: this proposal was drafted with the help of an AI coding assistant (GitHub Copilot). The numbers above were measured on the listed hardware.


> **Comment by sayakpaul (2026-10-12T07:08:42Z):**

Thanks for the proposal! Would it be possible to package the AITER Sage kernel with the Kernels project (similar to how SAGE and other attention backends like FA3 are listed [here](https://github.com/huggingface/diffusers/blob/883d51ef02cd7972eeaf158c652d229f3459f419/src/diffusers/models/attention_dispatch.py#L247))?

Once we have that I think we could add the attention backend as you're proposing here. @Abdennacer-Badaoui could help further with the build I think.


> **Comment by Shan2L (2026-10-12T08:56:14Z):**

Thanks @sayakpaul, that works for us.

Plan:
1. Package the gfx1201 Sage kernels as a Hub kernel, following `kernels-community/aiter-flash-attn-ck`: vendor the HIP sources from ROCm/aiter (INT8 Q/K + FP8 V quantization, and the attention kernel), build for `gfx1201`, and expose a `sageattn`-style entry point.
2. Add the diffusers attention backend on top of that Hub kernel, as proposed here.

The kernels are still under review in ROCm/aiter (https://github.com/ROCm/aiter/pull/6390). I'd like to vendor a reviewed aiter commit, so I'll open the Kernels PR once that lands and follow up here. @Abdennacer-Badaoui, I'll ping you on that PR for the build setup.



> **Comment by Abdennacer-Badaoui (2026-10-12T09:32:17Z):**

Thanks, @Shan2L ! Yes, sure.


> **Comment by Shan2L (2026-10-13T07:59:16Z):**

A quick update: the aiter review (ROCm/aiter#6390) is taking a bit longer than I expected, so I've gone ahead and started the Kernels package from that PR's current commit rather than waiting. I hope that's okay.

- The package (`aiter-sage-attention`, laid out like `aiter-flash-attn-ck`) builds for gfx1201 and passes its tests. Its outputs are bit-identical to aiter's build of the same commit.
- It exposes a `sageattn(q, k, v, tensor_layout, is_causal, sm_scale, return_lse)` entry point, so the diffusers backend can follow `_sage_attention_hub`.
- In diffusers, a local `aiter_sage_hub` backend runs MiniMax-H3 text-to-video+audio end to end on one R9700 (960x544, 124 frames, 9 steps). Each transformer call took 12.0 s instead of 16.5 s with the native backend, and the output looks the same (26.1 dB video PSNR to native). The VAE's attention uses head dim 64, so the backend sends calls the kernel can't take to SDPA.
- kernels-community asks new contributors to start with an issue, so I've opened one: https://github.com/huggingface/kernels-community/issues/1222. @Abdennacer-Badaoui, if you have time, that might be a good place to discuss the build.

I'll re-sync the vendored sources once the aiter PR is merged, and open the diffusers backend PR (as a draft) once the Hub kernel is available. Thanks again for the guidance!



### #14640 [closed] ModularPipeline: loading from a local directory still resolves components against the hub id stored in modular_model_index.json (MiniMax-H3 re-downloads 134 GB)
URL: https://github.com/huggingface/diffusers/issues/14640  by 4suino 2026-08-28T15:20:38Z  comments: 4

### Describe the bug

`ModularPipeline.from_pretrained(local_dir)` / `blocks.init_pipeline(local_dir)` followed by `load_components()` does not load the components from `local_dir` when the directory contains a `modular_model_index.json` whose specs carry a hub repo id. The specs are taken verbatim from the file, so every component is resolved against the hub instead of the directory that already holds it. For a converted checkpoint like `MiniMaxAI/MiniMax-H3` — whose index files ship nine `"pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3"` entries — a user who downloaded the snapshot to a local directory (e.g. `hf download MiniMaxAI/MiniMax-H3 --local-dir models/minimax-h3`) gets a second, full 134 GB download into the hub cache on first `load_components()`. Offline, the components simply fail to load and stay `None`.

The asymmetry is in `modular_pipeline.py` (v0.40.0):

- with `modular_model_index.json` present (line 1749), each spec is rebuilt from the file as-is (line 1755) — the hub id wins over the local directory;
- without it, the `model_index.json` fallback builds the spec as `"repo": pretrained_model_name_or_path` (line 1768) — i.e. the local directory, which is the expected behaviour.

So a local snapshot works only if the user hand-edits both index files to point at the local path (which is what we do).


### Reproduction

No download needed — an empty cache plus offline mode makes the hub lookup fail loudly instead of pulling 134 GB. `snapshot/` is a local copy of `MiniMaxAI/MiniMax-H3` with its original index files.

```python
import os
os.environ["HF_HOME"] = "/tmp/empty_hf_home"   # fresh, empty cache
os.environ["HF_HUB_OFFLINE"] = "1"

from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks

pipe = MiniMaxH3Blocks().init_pipeline("snapshot")   # contains scheduler/scheduler_config.json
pipe.load_components(names=["scheduler"])            # smallest component, 4 KB on disk
print(pipe.scheduler)                                # -> None
```

Rewriting the nine `pretrained_model_name_or_path` entries in `snapshot/modular_model_index.json` (and `model_index.json`) to the local path, and running the same script, loads `MiniMaxH3Scheduler` from disk.

### Logs

```shell
ValueError: Unable to load scheduler using load method: MiniMaxAI/MiniMax-H3 does not appear to have a file named scheduler_config.json.


(logged by `load_components`; the component is left as `None`. With the network on, the same call downloads the whole repo into the cache.)

### Expected behavior / proposed fix

When `pretrained_model_name_or_path` is a local directory, specs read from its `modular_model_index.json` should resolve against that directory whenever `<local_dir>/<subfolder>` exists — the same thing the `model_index.json` fallback already does — with the hub id kept only as a fallback for pruned snapshots that intentionally omit components. Alternatively, `save_pretrained` could write local/relative references so a downloaded snapshot is self-contained. Happy to open a PR for the first option if that shape is acceptable.
```

### System Info

- 🤗 Diffusers version: 0.40.0 (lines above from the v0.40.0 tag; same on main)
- Platform: macOS-26.6.2-arm64-arm-64bit-Mach-O
- Python version: 3.14.5
- PyTorch version (GPU?): 2.14.0.dev20260808 (MPS)
- Huggingface_hub version: 1.27.0
- Transformers version: 5.14.1
- Accelerate version: 1.14.0
- Accelerator: Apple M5 Max, 128 GB unified memory
- Using GPU in script?: no (loading only)
- Using distributed or parallel set-up in script?: no

### Who can help?

@apolinario (MiniMax-H3 integration, #14355) — and whoever owns Modular Diffusers loading

> **Comment by 22elix3r (2026-08-28T23:50:01Z):**

I'm investigating #14640 against current main, focusing on how `ModularPipeline.__init__` applies component specs from `modular_model_index.json` when `pretrained_model_name_or_path` is a local directory. I'll compare that path with the `model_index.json` fallback (which already binds `repo` to the local path), check the pointer-repo case where some components are intentionally Hub-only, and add regression coverage for a local snapshot whose index still points at a Hub id before proposing a fix.


> **Comment by yiyixuxu (2026-08-31T05:00:13Z):**

thanks @4suino  @22elix3r! 
  
However, the loading behavior is by design — the loader loads each component based on what `modular_model_index.json` says; we don't guess intent and reinterpret 
 
I understand the inconvenience, though. If you want a self-contained local folder, you can download it use `save_pretrained()` with `overwerite_modular_index = True`:

```py
  pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
  pipe.load_components()
  pipe.save_pretrained(local_dir, overwrite_modular_index=True)
```
 


> **Comment by 4suino (2026-08-31T07:56:02Z):**

Thanks — that makes sense, and `overwrite_modular_index=True` does produce exactly the self-contained folder we ended up building by hand, so we'll treat that as the canonical shape.

One gap worth noting for the record: the suggested flow starts from `from_pretrained("MiniMaxAI/MiniMax-H3")`, so a user who already has the snapshot on disk (e.g. via `hf download --local-dir`) still pays the full second download into the cache before they can save the self-contained copy — which is the original pain. A cheap middle ground could be documentation: a short note on the Modular loading page saying that `--local-dir` snapshots of published modular repos are not self-contained until re-saved with `overwrite_modular_index=True`, with the three-line recipe. Happy to send that doc PR if useful.



> **Comment by yiyixuxu (2026-09-04T00:17:29Z):**

hey so we now overwrite by default https://github.com/huggingface/diffusers/pull/14659, also documented that you may need to update the index manually if the snapshop on disk that you downloaded using `hf download` 

I think we can close this issue now, let me know if you still have more feedbacks! 



### #14410 [open] MiniMax H3 unnecessarily swapped w1 in VAE
URL: https://github.com/huggingface/diffusers/issues/14410  by bghira 2026-08-06T17:16:25Z  comments: 3

The MiniMax H3 model is very popular, congratulations to the MiniMax team.

However, there's currently an issue caused by the Diffusers' team once again modifying the layout of the model for seemingly no reason.

The video VAE's weights are reorganised from `[gate, up]` to `[up, gate]`.

why? why does this keep happening? what is the actual reason for the swap?

It's caused an enormous amount of time waste to try and make the two weight setups both loadable and function correctly during training and inference.

This isn't the first time it's happened, but it's the first time it's been done in a way that is **extremely difficult to track down the issue**. Usually, the unnecessary change arises from qkv projections being split when the upstream model release had them fused (and oftentimes their paper eg. Flux.1 and others have mentioned qkv fusion as ultimately required to achieve high levels of hardware utilisation)

Look at the result:

https://github.com/user-attachments/assets/766ea4b9-49dc-40f0-bdf4-af43b4cd9265

It's probably too late to fix it now, just like every other time the Diffusers pipeline has been released without enough time to absorb community feedback. Now we're stuck with two, incompatible sets of weights being distributed for H3 probably forever.

I understand that the tone of this post is going to upset people, but I'm not really worried about it. The time wasted through debugging this issue is worse. It's going to keep affecting people until either the weights are recalled and replaced or another model comes along to replace H3.

Can you simply stop making these changes, especially ones that don't cause obvious errors?

> **Comment by apolinario (2026-08-12T09:31:42Z):**

Hi, I don't think the unkind tone is productive but I understand your frustration. 

The reasons for the changes are not arbitrary tho: the reorders exist so H3 reuses diffusers' shared `SwiGLU` and attention machinery, which is what makes library-wide logic (PEFT targeting, attention backends, quantization exclusion lists, context parallelism in #14407, etc.) work uniformly instead of per-model. It is a trade-off to match all models to a lib-wide logic. But indeed maybe those changes could be documented beyond just the conversion code so that frustrations such as the one you hit don't occur. 

Also, a fused QKV is just one line of code away:
```python
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", workflow="t2va")
pipe.load_components(dtype=torch.bfloat16)
pipe.transformer.fuse_qkv_projections()
```


> **Comment by bghira (2026-08-12T16:22:25Z):**

i mentioned the tone would upset people in the original post - it's quite common for the tone of a post to become the point of contention instead of the technical issue underpinning it. i've seen it before, which is why i mentioned i already know that ahead of time. but nothing about is unkind. perhaps exasperated, but hardly even rude.

there's nothing about shared logic that requires the weights themselves have the difference baked in. it's a torch index selection that costs microseconds to do at runtime (even on Apple) and didn't require an actual weight layout change that is barely perceptible until you go digging to find the conversion code and check it out. all of the same internals work just fine when swapping the order at runtime, and exports of finetuned weights will have one fewer reason to produce corrupted outputs when using eg. ComfyUI to inference on them.

the "fuse qkv projections" code before (i'm not sure now how it behaves, i've ripped all of that out and replaced it with proper version on my end) wasn't suitable for training on. it left the original unsplit projections around, which the default PEFT target to_q/..k/..v end up training if not careful. but if it's so easy to fuse with a helper, then it is continued as the original question, "why do this?" - there's another issue open about that particular issue, we don't need to litigate qkv split here. we all know that change was historically done due to "project philosophy" which was created a while before the library supported so many different models and methods. now it's being reconsidered, but at a glacial pace; so, it's still happening. you could just as easily reverse the direction and have a helper to *unfuse* the projections. 


> **Comment by bghira (2026-08-12T16:31:49Z):**

the main action item here isn't to debate what or why it's done, but to 1) delete the converted weights from the hub, 2) convert the weights in the loader / some other way at runtime, 3) stick to "loudly failing" changes in baked weights in the future.

the pushed weights are likely mirrored in other places, but that's why it's so frustrating that this happened in the way that it has. the problem is already proliferating around various copies of the weights. there's no metadata or config flag to indicate it's swapped either.

things that have the same shape before and after conversion just shouldn't be baked into the weights. maybe it's an unpopular opinion, but i'd rather the code tell me i have an actual tensor mismatch rather than just silently corrupting outputs and training results. at this moment, it's only Diffusers who requires this layout. there's a lot of other bespoke repositories that are derived from the ComfyUI code or the original H3 repo, neither of which have that swap in place.

edit to add that this problem also exists in LoRAs that are trained against the swapped weights.



### #15208 [open] [modular] Per-workflow input signature in the docstring of blocks with `_workflow_map`
URL: https://github.com/huggingface/diffusers/issues/15208  by yiyixuxu 2026-11-08T08:36:27Z  comments: 3

Follow-up to #15136.

For the final assembled blocks that support workflows (the ones with `_workflow_map`, i.e. the auto blocks a checkpoint maps to), the signature and the docstring aren't informative enough. An input is listed, but you don't know which workflow actually uses it, whether it's required there, or what its default is, since the defaults can differ between workflows.

Today the way to see the exact signature is to prune the blocks with `get_workflow(name)` and look at `.inputs` on the result. That works, but it is better to put that information in the docstring for the users.

This PR proposes a better docstring for these blocks, limited to blocks that define `_workflow_map`: per workflow, the inputs it takes with their `required` flag and default, with the inputs that are the same in every workflow listed once.

## Examples

### `AnimaAutoBlocks`

Today (after #15136):

```
  Supported workflows:
    - `text2image`: requires `prompt`
    - `img2img`: requires `image`, `prompt`

  Inputs:
      prompt (`str`):
          The prompt or prompts to guide image generation.
      negative_prompt (`str`, *optional*):
          ...
      image (`Image | list`, *optional*, depends on the workflow, see `vae_encoder`):
          Reference image(s) for denoising. Can be a single image or list of images.
      ...
      strength (`float`, *optional*, defaults to 0.9):
          Strength for img2img/inpainting.
      ...
```

`image` is optional in the list but required for `img2img`. `strength` reads like a general input, but `text2image` does not use it.

Proposed:

```
  Supported workflows:
    - `text2image`: requires `prompt`
    - `img2img`: requires `image`, `prompt`

  Inputs (all workflows):
      prompt (`str`):
          The prompt or prompts to guide image generation.
      negative_prompt (`str`, *optional*):
          ...
      num_inference_steps (`int`, *optional*, defaults to 50):
          ...

  Inputs (`img2img`):
      image (`Image | list`):
          Reference image(s) for denoising. Can be a single image or list of images.
      strength (`float`, *optional*, defaults to 0.9):
          Strength for img2img/inpainting.
```

### `MiniMaxH3Blocks`

Today:

```
  Supported workflows:
    - `t2va`: requires `prompt`
    - `fl2va`: requires `prompt`, `image` or `prompt`, `last_image`
    - `ref2va`: requires `prompt`, `references`

  Inputs:
      ...
      references (`list`, *optional*, depends on the workflow, see `...`):
          ...
      num_frames (`int`, *optional*):
          Number of frames to generate, at the fixed 24 fps. ...
      image (`Image`, *optional*):
          ...
      last_image (`Image`, *optional*):
          ...
```

`num_frames` shows as optional with no default. In `t2va` and `fl2va` it defaults to 124, in `ref2va` it is required (the model reads the length from the references). The docstring can't say that today.

Proposed:

```
  Inputs (all workflows):
      prompt (`str`):
          ...
      height (`int`, *optional*):
          ...

  Inputs (`t2va`, `fl2va`):
      num_frames (`int`, *optional*, defaults to 124):
          Number of frames to generate, at the fixed 24 fps. ...

  Inputs (`fl2va`):
      image (`Image`, *optional*):
          ...
      last_image (`Image`, *optional*):
          ...

  Inputs (`ref2va`):
      references (`list`):
          ...
      num_frames (`int`):
          Number of frames to generate, at the fixed 24 fps. ...
```

Each per-workflow section is `get_workflow(name).inputs` minus what is already in the shared list. The same renderer change applies to the model-card doc in `modular_pipeline_utils.py`, which already loops over `_workflow_map` but only prints the trigger inputs.

---

_Posted by Claude Code on behalf of @yiyixuxu._


> **Comment by voropaevv (2026-11-09T10:31:21Z):**

I'd like to contribute an AI-assisted patch for this follow-up to #15136. The local patch uses each resolved workflow's input specs in the block docs and model card, with identical entries listed once. I checked Anima and MiniMax H3 without loading model weights. Would you be open to this PR, and should I wait for #15136 to merge before opening it?



> **Comment by yiyixuxu (2026-11-09T18:17:23Z):**

we are not looking for PR contribution for this issue
feel free to share your thoughts/feedbacks on design etc. we will add you as a co-author if your feedback is used in our PR 


> **Comment by voropaevv (2026-11-10T13:05:45Z):**

At `c2667549` (#15136), the model-card renderer raises on MiniMax H3's `fl2va` because it calls `.keys()` on a tuple of trigger sets (`prompt` + `image` or `prompt` + `last_image`). `format_workflow` already renders both alternatives. Keeping that summary separate lets the input rows come directly from `get_workflow(name).inputs`.

I'd share the grouping logic between `.doc` and the model card. Use the displayed spec to distinguish rows with different requiredness or defaults, and group identical rows by the workflows that use them. If a workflow cannot be resolved, show that in its section. Use "all workflows" only when the row is present in every declared workflow.

In an AI-assisted prototype based on `c2667549`, both renderers matched the resolved input names, requiredness, and displayed defaults for all five Anima and MiniMax H3 workflows. The check ran without loading model weights.




### #14025 [open] MagCache on Wan 2.2 Dual-Transformer Pipelines: Incorrect Step Accounting and Limited Effectiveness on a 4-Step Distilled Model (e.g. Wan2.2)
URL: https://github.com/huggingface/diffusers/issues/14025  by fatemehgh1313 2026-06-21T14:49:14Z  comments: 3


</h2><p>While evaluating MagCache on Wan 2.2, I found what appears to be a design issue in the current implementation for dual-transformer pipelines.</p><p>The current implementation assumes that a transformer executes exactly <code inline="">num_inference_steps</code> times during inference. This assumption is valid for single-transformer architectures such as Flux, SDXL, and Wan 2.1, but it does not hold for Wan 2.2, where denoising is split between two transformers that each execute only during part of the diffusion process.</p><p>After implementing a prototype fix that assigns independent MagCache configurations to each transformer based on the number of timesteps it actually executes, I still observe poor practical results on a distilled 4-step Wan 2.2 model:</p><ul><li><p>At thresholds below 0.3, MagCache provides little or no measurable speedup.</p></li><li><p>At thresholds greater than or equal to 0.3, runtime improves (~25%), but output quality degrades severely.</p></li><li><p>The same behavior occurs whether MagCache is applied only to the first transformer or to both transformers.</p></li></ul><p>I am therefore opening this issue both as a bug report regarding the current dual-transformer implementation and as a request for feedback regarding the expected behavior of MagCache on heavily step-distilled Wan 2.2 models.</p><hr><h2>Background</h2><p>According to the current MagCache documentation, calibration produces a set of <code inline="">mag_ratios</code> that describe the model's intrinsic magnitude decay curve.</p><p>Example workflow:</p><pre><code class="language-python">calib_config = MagCacheConfig(
    calibrate=True,
    num_inference_steps=4,
)
</code></pre><p>After calibration, the reported ratios are passed back into the inference configuration:</p><pre><code class="language-python">mag_config = MagCacheConfig(
    mag_ratios=[...],
    num_inference_steps=4,
)
</code></pre><p>The documentation also states that ratios calibrated at higher step counts can be reused at lower step counts through interpolation.</p><p>For example, ratios obtained from a 50-step calibration can be interpolated and reused for a 20-step or lower-step inference schedule.</p><hr><h2>Experimental Setup</h2><p>The tested model is not the original Wan 2.2 inference configuration.</p><p>The workflow is:</p><ol><li><p>Start from the original Wan 2.2 14B model (designed for approximately 50 inference steps).</p></li><li><p>Apply a step-distillation LoRA to convert the model into a 4-step model.</p></li><li><p>Evaluate MagCache on the resulting distilled model.</p></li></ol><p>Therefore all observations below correspond to:</p><pre><code class="language-text">Wan 2.2 14B (50-step)
    ↓
Step Distillation LoRA
    ↓
4-step distilled model
    ↓
MagCache evaluation
</code></pre><p>I also tested using interpolated MagCache ratios derived from the original 50-step calibration, following the workflow described in the documentation.</p><p>The resulting outputs were significantly degraded on the distilled 4-step model.</p><hr><h2>Potential Bug: Dual-Transformer Step Accounting</h2><p>Wan 2.2 divides denoising between two transformers:</p><ul><li><p><code inline="">transformer</code> (high-noise stage)</p></li><li><p><code inline="">transformer_2</code> (low-noise stage)</p></li></ul><p>The split is determined by <code inline="">boundary_ratio</code>.</p><p>For example, with a 4-step schedule:</p>
Stage | Executions
-- | --
High-noise transformer | 2
Low-noise transformer | 2

<p>Neither transformer executes 4 times.</p><p>However, MagCache currently receives:</p><pre><code class="language-python">num_inference_steps = 4
</code></pre><p>for both transformers.</p><p>Internally, calibration completion depends on:</p><pre><code class="language-python">if state.step_index &gt;= self.config.num_inference_steps:
</code></pre><p>This implicitly assumes:</p><pre><code class="language-text">transformer executions == num_inference_steps
</code></pre><p>which is not true for Wan 2.2.</p><p>As a result, calibration and state management appear to be tied to the total diffusion step count rather than the number of executions of the individual transformer.</p><hr><h2>Prototype Fix</h2><p>To address this, I implemented a prototype that treats each transformer as an independent MagCache scope.</p><p>The scheduler timesteps are first split according to the pipeline's <code inline="">boundary_ratio</code>.</p><p>Example:</p><pre><code class="language-text">4 diffusion steps

High-noise transformer: 2 steps
Low-noise transformer: 2 steps
</code></pre><p>Each transformer then receives:</p><ul><li><p>its own <code inline="">MagCacheConfig</code></p></li><li><p>its own <code inline="">num_inference_steps</code></p></li><li><p>its own <code inline="">mag_ratios</code></p></li><li><p>its own MagCache state</p></li></ul><p>instead of sharing a single global diffusion-step count.</p><p>Conceptually:</p><pre><code class="language-text">transformer
 ├── own state
 ├── own ratios
 └── own step count

transformer_2
 ├── own state
 ├── own ratios
 └── own step count
</code></pre><p>This guarantees that:</p><pre><code class="language-python">if state.step_index &gt;= self.config.num_inference_steps:
</code></pre><p>is evaluated against the number of executions of that specific transformer rather than the total diffusion schedule.</p><hr><h2>Results</h2><p>After implementing the per-transformer step accounting described above, I evaluated MagCache on the distilled 4-step Wan 2.2 model.</p><h3>Threshold &lt; 0.3</h3><p>No meaningful speedup was observed.</p><p>Runtime remained very close to the baseline.</p><h3>Threshold &gt;= 0.3</h3><p>Runtime improved from approximately:</p><pre><code class="language-text">40 s → 30 s
</code></pre><p>which corresponds to roughly a 25% speedup.</p><p>However, output quality degraded significantly and became visibly worse than the baseline.</p><h3>Transformer Coverage</h3>
<p>I tested both:</p>
<ul>
<li><p>applying MagCache only to <code inline="">transformer</code></p></li>
<li><p>applying MagCache to both <code inline="">transformer</code> and <code inline="">transformer_2</code></p></li>
</ul>

<p>The behavior was similar in terms of speed/quality tradeoff, but there was one noticeable difference:</p>

<ul>
<li><p>Applying MagCache only to the first transformer (<code inline="">transformer</code>) produced better output quality compared to applying MagCache to both transformers, but provided no meaningful speedup and had the same inference time as the base model with LoRA.</p></li>
<li><p>However, the quality was still significantly worse than the baseline without MagCache.</p></li>
</ul>

<p>This suggests that the second transformer may be more sensitive to MagCache reuse, or that the error introduced by caching accumulates differently across the two denoising stages.</p>

baseline wan22 14b 4step distilled:
https://github.com/user-attachments/assets/01424605-874a-4635-b07e-4817111227c7

baseline wan22 14b 4step distilled + apply magcache on both transformers:
https://github.com/user-attachments/assets/22e52d25-a524-496d-9756-18df0942adcd

baseline wan22 14b 4step distilled + apply magcache on first transformer:
https://github.com/user-attachments/assets/4e99b509-b17f-4c1e-8463-90b273fa3420

> **Comment by lcheng321 (2026-06-21T17:43:13Z):**

Submitted #14027 to fix the step counting bug.

Both transformers were receiving `num_inference_steps` equal to the full pipeline count, but each only runs for a fraction of the total steps. This misaligns `_advance_step`, the retention window, and `mag_ratios` indexing. The fix counts actual per-transformer steps from the scheduler timesteps and corrects each config before the denoising loop.

The poor effectiveness on heavily distilled models is a separate issue and not addressed here. MagCache relies on smooth magnitude decay across steps, which breaks down when 50 steps are distilled into 4.


> **Comment by lcheng321 (2026-06-25T15:06:26Z):**

@fatemehgh1313 Hi, PR #14027 has been submitted to fix the per-transformer step counting bug you identified. The fix computes actual execution counts from scheduler timesteps and corrects each MagCacheConfig before the denoising loop.

Would you be able to retest with that branch and report whether the step accounting issue is resolved? Specifically curious whether calibration now completes correctly and whether the retention window behaves as expected at your boundary_ratio split.

Thanks for the detailed write-up, it made the root cause easy to trace.


> **Comment by lcheng321 (2026-07-03T07:12:25Z):**

Hi @fatemehgh1313, I tested the fix locally with a small CPU repro and it resolves the step accounting bug. Before the fix, both transformers reported num_inference_steps = 4 despite each only running 2 times, and the retention step was wrong at 1 instead of 0. After the fix, both report the correct 2, the retention step is correctly 0, and 20 unit tests pass covering asymmetric splits and edge cases. I do not have your distilled 4 step Wan 2.2 setup to test the quality degradation you reported, so if you get a chance to retest on the fix branch, no pressure at all, but I would appreciate hearing whether calibration and the retention window now behave as expected on your side.



### #14433 [open] Title: Group offloading with use_stream=True fails on torchao int8 version=2 (pin_memory on CUDA qdata)
URL: https://github.com/huggingface/diffusers/issues/14433  by animede 2026-08-10T02:58:25Z  comments: 3

Title: Group offloading with use_stream=True fails on torchao int8 version=2 (pin_memory on CUDA qdata)

> Context: found while building an app that runs MiniMax-H3 through diffusers. The
> investigation was done with an AI coding agent. The reproduction below was executed on my
> own hardware and the output is pasted verbatim, and I am accountable for its accuracy.

### Describe the bug

`apply_group_offloading(...)` fails deterministically on the first forward when
`use_stream=True` and `low_cpu_mem_usage=True` are combined and the module's weights are
torchao `Int8Tensor` produced by `Int8WeightOnlyConfig(version=2)`. Both flags are
documented options, and the pairing is the natural one for a memory-constrained setup
(streamed prefetch + reduced host RAM):

```
RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned
```

This is a different failure from the one fixed in #14112. That PR was explicitly scoped to
the v1 `AffineQuantizedTensor` path and states:

> the int8 compile path that uses TorchAO `version=2` [...] support `is_pinned()` and
> `pin_memory()` on current main

`Int8Tensor` does implement `pin_memory()` — that part holds. The problem is what its
implementation pins: it forwards to the inner `qdata`, and at that point `qdata` is on
CUDA, so the pin raises. So v2 was left on the unconditional-pinning path on the
assumption that pinning works, and it does not.

### Reproduction

Self-contained, no pretrained weights (a 256-dim, 4-block Linear stack is enough):

```python
import torch
from diffusers.hooks.group_offloading import apply_group_offloading
from torchao.quantization import Int8WeightOnlyConfig, quantize_


def build_stack(num_blocks=4, dim=256):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(dim, dim * 2, bias=False, dtype=torch.bfloat16)
            self.lin2 = torch.nn.Linear(dim * 2, dim, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            return self.lin2(torch.nn.functional.gelu(self.lin1(x)))

    class Stack(torch.nn.Module):
        _supports_group_offloading = True

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(num_blocks)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    stack = Stack()
    for block in stack.blocks:
        quantize_(block, Int8WeightOnlyConfig(version=2))
    return stack


def run(use_stream, low_cpu_mem_usage):
    stack = build_stack()
    apply_group_offloading(
        stack,
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=use_stream,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    x = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        with torch.no_grad():
            stack(x)
    return "OK"


for label, stream, low_cpu in [
    ("use_stream=True,  low_cpu_mem_usage=True ", True, True),
    ("use_stream=False, low_cpu_mem_usage=True ", False, True),
    ("use_stream=True,  low_cpu_mem_usage=False", True, False),
]:
    try:
        print(f"{label} -> {run(stream, low_cpu)}", flush=True)
    except Exception as e:
        print(f"{label} -> {type(e).__name__}: {e}", flush=True)
```

Output:

```
use_stream=True,  low_cpu_mem_usage=True  -> RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned
use_stream=False, low_cpu_mem_usage=True  -> OK
use_stream=True,  low_cpu_mem_usage=False -> OK
```

### Traceback (tail; paths shortened)

```
File "diffusers/hooks/group_offloading.py", line 289, in _onload_from_memory
    with self._pinned_memory_tensors() as pinned_memory:
File "diffusers/hooks/group_offloading.py", line 205, in _pinned_memory_tensors
    param: tensor.pin_memory() if not tensor.is_pinned() else tensor
File "torchao/quantization/quantize_/workflows/int8/int8_tensor.py", line 427, in _
    pinned_qdata = args[0].qdata.pin_memory()
RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned
```

### Analysis

Two places disagree about who pins, and the disagreement is only observable for tensor
subclasses whose payload lives in a member:

- `_init_cpu_param_dict()` runs once at enable time and, via `_to_cpu()`, **skips** pinning
  when `low_cpu_mem_usage=True` (`return t if low_cpu_mem_usage else t.pin_memory()`).
- `_pinned_memory_tensors()` is entered from `_onload_from_memory()` on **every step** when
  `use_stream=True`, and pins unconditionally.

For a plain tensor the second path is merely redundant work. For `Int8Tensor` it reaches
`qdata.pin_memory()` on data that is not on the host, and raises.

The two working combinations in the output above are consistent with this: `use_stream=False`
never enters `_pinned_memory_tensors()` at all, and `low_cpu_mem_usage=False` pins once at
enable time, while the module is still entirely on CPU, so the per-step path finds
`is_pinned() == True` and skips.

### Suggested fix

Any of these would close the gap:

1. Make `_pinned_memory_tensors()` honour `low_cpu_mem_usage` (skip when the init path
   skipped), mirroring the two functions' intent.
2. Wrap the pin in a try/except and fall back to the unpinned CPU copy — this is what
   #14112's description says it does for tensors whose pinning ops are unavailable, but
   the current code has no such fallback for this path.
3. At minimum, reject the combination up front instead of failing mid-denoise, since the
   failure surfaces far from its cause.

Worth noting for (1): `use_stream=True, low_cpu_mem_usage=False` is not just "the
workaround" — in my measurements on a 33B transformer it is **4–5x faster to onload**
(0.04–0.07s vs 0.1–0.26s per block), because pinned memory can't be paged out. The cost is
~14–16GB of page-locked host RAM and ~22s spent pinning at enable time. Users who hit this
crash and retreat to `use_stream=False` lose that speedup.


### System Info

- diffusers: `main` (verified `hooks/group_offloading.py` is byte-identical to the commit
  I tested, f37ab93)
- torch 2.9.0+cu128, torchao 0.17.0, transformers 5.14.1
- GPU: NVIDIA RTX PRO 6000 Blackwell (sm_120), Ubuntu 24.04, Python 3.12
- I have only tested torchao 0.17.0; I have not checked whether a newer torchao changes
  `Int8Tensor.pin_memory()`'s behaviour here.

Originally hit while running MiniMax-H3 (33B) with the transformer int8-quantized and
group-offloaded; the reproduction above is the reduced case. Full context and the probe
scripts: https://github.com/animede/Diffusers_minimax-h3

### Who can help?

@sayakpaul @a-r-r-o-w — the issue template has no line for group offloading /
quantization, so I picked based on recent commits to `hooks/group_offloading.py`.
Apologies if that routes to the wrong person.

> **Comment by sayakpaul (2026-08-10T05:48:04Z):**

This is better submitted to TorchAO.


> **Comment by animede (2026-08-10T10:57:14Z):**

Thanks for taking a look! Before filing this with TorchAO I tried to isolate which side
the failure actually lives on, and I'd like to share what I found — it changed my
understanding of the bug.

In torchao alone (0.17.0), I couldn't reproduce any pinning problem: `pin_memory()` on a
CPU-quantized `Int8Tensor` works, `is_pinned()` works, and `.cpu()` on a CUDA-quantized
one moves `qdata` to the host correctly.

What does reproduce the exact error is the combination of two things in
`hooks/group_offloading.py`:

1. `_to_cpu()` calls `tensor.cpu()` for torchao tensors; on an already-CPU tensor that
   returns **the same object** (standard torch semantics), so `cpu_param_dict` ends up
   holding an alias of the live parameter rather than a copy. The plain-tensor path
   happens to be immune, because `tensor.data.cpu()`'s view survives a later
   `tensor.data = ...` reassignment.
2. `_swap_torchao_tensor()` then onloads by mutating that same object's attributes in
   place — after which the "CPU copy" in the dict is CUDA-backed, and pinning it raises.

Minimal reproduction using the two functions directly:

```python
lin = torch.nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)
quantize_(lin, Int8WeightOnlyConfig(version=2))
w = lin.weight

cpu_copy = w.cpu()                       # what _to_cpu() stores; `cpu_copy is w` -> True
_swap_torchao_tensor(w, w.to("cuda"))    # what onload does
print(cpu_copy.qdata.device)             # cuda:0 — the "CPU copy" followed the swap
cpu_copy.pin_memory()
# RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned
```

So as far as I can tell, torchao's `pin_memory()` behaves correctly with what it's given
— the tensor it receives just isn't the CPU copy it was meant to be. One could argue
torchao might defensively copy-and-pin here, and I'm happy to raise it there too if you
think that's the better home for it. But a `_to_cpu()` that makes a real copy for torchao tensors (or a swap that doesn't
mutate objects aliased by `cpu_param_dict`) would resolve it on this side — the fix
looks fairly contained.


> **Comment by michaelmanly (2026-10-15T08:34:04Z):**

Hey, reproduced this on a 4090 with torch 2.9.0 and torchao 0.17.0.

Using `use_stream=True` with `low_cpu_mem_usage=True` hits the same CUDA tensor pinning error.

Disabling either option worked in my tests, so those look like decent workarounds.

Results: https://aibadgr.com/evidence/diffusers-14433-group-offloading-use-stream-torchao-int8

I ran this through Badgr. If you want to try running AI workloads on cloud GPUs, happy to give you $5 in credits: https://aibadgr.com/



### #14730 [closed] [modular] move unpack out of the decoder blocks for flux, krea2 and ltx
URL: https://github.com/huggingface/diffusers/issues/14730  by yiyixuxu 2026-09-07T22:35:44Z  comments: 3

**The principle:** there are two possible transformations that sit between a VAE and a transformer — *normalize/denormalize* (the VAE's latent statistics) and *pack/unpack* (`[B, C, F, H, W]` <-> a token sequence `[B, S, D]`). Each must be applied and undone in mirrored pairs, and the core denoise group must hand `latents` back in the same form it received them — so the "`latents`" output is always in one consistent form that any block (a decoder, a latent upsampler, a second denoise group) can consume, and no block outside the group needs `height`/`width`/`num_frames` just to interpret tokens.

There are two acceptable patterns; both follow this principle.

**Pattern 1 (the most common one):** normalize/denormalize on unpacked dimension

```
encode → norm → pack → denoise → unpack → denorm → decode
```

The pack/unpack lives in core-denoise blocks: either at block level (the prepare-latents step packs, a dedicated after-denoise step unpacks) or inside the transformer's `forward` (the model patchifies/unpatchifies internally and blocks never pack at all).

**Pattern 2 (packed-space statistics):** when the VAE's statistics are defined over the *packed* channels, norm/denorm can only run on packed tensors, so pack/unpack has to happen inside the VAE blocks as well:

```
encode → pack → norm → denoise → denorm → unpack → decode
```

**Where the codebase stands** (all 19 modular families):

- Pattern 1, model level (the transformer patchifies internally, blocks never pack): `anima`, `cosmos`, `helios`, `hunyuan_video1_5`, `minimax_music3`, `stable_diffusion_3`, `stable_diffusion_xl`, `wan`, `wan_animate_2`, `z_image` — fine.
- Pattern 1, block level: `qwenimage`, `flux2`, `minimax_h3` — fine. `ltx2` is being moved to this pattern in #14612 (adds `LTX2UnpackLatentsStep` at the end of the core denoise group). `ideogram4` also follows it despite packed-space statistics, by tiling the stats onto the unpacked channels in its decoder (norm/denorm is elementwise and pack is a permutation, so they commute if you permute the stats along).
- Pattern 2: `ernie_image` (decode side: denorm on the packed latents, then unpack, then `vae.decode`; it is t2i-only so there is no encode side) — fine.
- **Outliers — where the fix is needed:** `flux`, `krea2` and `ltx` pack inside the denoise group but unpack inside the decoder block. That asymmetry strands packed `[B, seq_len, dim]` latents in the pipeline state after denoising, where no other block can consume them, and makes the decoders carry geometry inputs they don't otherwise need. (A leftover from porting the standard pipelines, where unpack → denorm → `vae.decode` is just the tail of `__call__`.)

**What to do:**

1. Start with a test in `ModularPipelineTesterMixin` (`tests/modular_pipelines/testing_utils/common.py`) that enforces the principle for every pipeline: each family's tester declares the one canonical form its pipelines keep `latents` in (the VAE form for pattern 1, the channel-packed form for pattern 2), and the test asserts the state actually carries that shape after denoising. A sketch of the idea (not actual code — untested, adapt as needed):

```python
# each tester declares the canonical latents form its family keeps in the state,
# for the geometry that get_dummy_inputs produces
expected_latents_shape = (1, 4, 32, 32)

# on ModularPipelineTesterMixin
def test_latents_output_in_vae_form(self):
    # run a partial pipeline without the decode step, and check the latents it leaves in the state
    # (existing tests already run sub-blocks this way, see e.g. test_modular_pipeline_stable_diffusion_xl.py)
    blocks = self.pipeline_blocks_class()
    blocks.sub_blocks.pop("decode")
    pipe = blocks.init_pipeline(self.pretrained_model_name_or_path)
    pipe.load_components()
    latents = pipe(**self.get_dummy_inputs(), output="latents")
    assert latents.shape == self.expected_latents_shape
```

Adding the test first surfaces everything that doesn't follow the principle — then look into each failure case to understand why, and whether there is any other pattern we have not identified here.

2. For each pipeline the test catches, add an unpack step at the end of the core denoise group (following `Flux2UnpackLatentsStep`, or `LTX2UnpackLatentsStep` from #14612, as the reference) and remove the unpack + related geometry inputs from the decoder block. Denormalization stays in the decoder, right before `vae.decode` (it just operates on the unpacked latents now).

3. Deprecate the old packed form instead of hard-switching. This is a behavior change: reading `latents` back from the state (e.g. `pipe(..., output="latents")`) currently gives *packed* latents for these three pipelines; after the move it gives unpacked latents. To keep existing workflows running (packed latents that were saved, or fed into a decoder-only pipeline), the decoder block should keep accepting the old packed form for a deprecation window: dispatch on `ndim`, and when latents come in packed, unpack them with a deprecation warning telling users to pass VAE-form latents instead.

**How to verify:** the new mixin test passes across `tests/modular_pipelines/`. Beyond that, this is a pure relocation of a reshape, so image/video outputs must be numerically identical and the rest of the existing tests under `tests/modular_pipelines/flux`, `krea2` and `ltx` should pass. Please also run the slow tests to show the image/video outputs are visually the same.

> **Comment by yiyixuxu (2026-09-07T22:36:13Z):**

cc @akshan-main if you're interested 


> **Comment by yiyixuxu (2026-09-07T22:36:40Z):**

Contributions are welcome, but we prefer existing/experienced contributors for this issue!
let me know if anyone is interested  


> **Comment by akshan-main (2026-09-07T22:39:35Z):**

@yiyixuxu happy to take this one!



### #15367 [open] MiniMaxH3ImageTransformer2DModel does not inherit PeftAdapterMixin
URL: https://github.com/huggingface/diffusers/issues/15367  by NancyFyong 2026-11-21T02:12:49Z  comments: 2

### Describe the bug

`MiniMaxH3ImageTransformer2DModel` does not inherit `PeftAdapterMixin`, so the standard PEFT/LoRA adapter API is unavailable on the MiniMax-H3 image transformer. The video transformer, `MiniMaxH3Transformer3DModel`, already inherits the mixin.

On current `main` (`c215605`) and `v0.43.0`:

- Image model: [`MiniMaxH3ImageTransformer2DModel`](https://github.com/huggingface/diffusers/blob/c215605a189169236544c1c9828789bfca0891d1/src/diffusers/models/transformers/transformer_minimax_h3_image.py#L365) is declared as `ModelMixin, ConfigMixin, AttentionMixin`.
- Video model: [`MiniMaxH3Transformer3DModel`](https://github.com/huggingface/diffusers/blob/c215605a189169236544c1c9828789bfca0891d1/src/diffusers/models/transformers/transformer_minimax_h3.py#L379) includes `PeftAdapterMixin`.

This blocks the native `add_adapter`, `load_lora_adapter`, `set_adapter(s)`, `enable_adapters`, `disable_adapters`, and `delete_adapters` workflow on H3-Image. I found this while integrating the model into `verl-omni`'s FlowGRPO and DiffusionNFT training actors:

- https://github.com/verl-project/verl-omni/blob/6481dab6/verl_omni/pipelines/minimax_h3_image_flow_grpo/diffusers_training_adapter.py#L35
- https://github.com/verl-project/verl-omni/blob/6481dab6/verl_omni/pipelines/minimax_h3_image_diffusion_nft/diffusers_training_adapter.py#L55

Both currently need a local compatibility subclass:

```python
from diffusers import MiniMaxH3ImageTransformer2DModel
from diffusers.loaders import PeftAdapterMixin


class H3ImageActor(MiniMaxH3ImageTransformer2DModel, PeftAdapterMixin):
    pass
```

A matching fix plus the standard model-level `LoraTesterMixin` coverage is ready in #15366. Comparing with the 3D integration also showed that PEFT support needs `@apply_lora_scale("attention_kwargs")`, the `attention_kwargs` forward argument, and pipeline plumbing; those are included in the PR as well.

### Reproduction

```python
from diffusers import MiniMaxH3ImageTransformer2DModel
from diffusers.loaders import PeftAdapterMixin
from peft import LoraConfig

print(issubclass(MiniMaxH3ImageTransformer2DModel, PeftAdapterMixin))
# False

model = MiniMaxH3ImageTransformer2DModel(
    num_attention_heads=2,
    attention_head_dim=16,
    hidden_size=24,
    num_layers=1,
    num_refiner_layers=1,
    ffn_dim=32,
    in_channels=4,
    patch_size=(1, 2, 2),
    text_dim=8,
    freq_dim=8,
    time_embed_hidden_dim=24,
    time_embed_dim=16,
    rope_freq_dim=2,
    num_pdd_heads=3,
)
model.add_adapter(LoraConfig(r=4, target_modules=["to_q"]))
```

Output:

```text
False
AttributeError: 'MiniMaxH3ImageTransformer2DModel' object has no attribute 'add_adapter'
```

### Context

This is needed for PEFT/LoRA training and adapter loading on MiniMax-H3-Image, including the `verl-omni` FlowGRPO and DiffusionNFT pipelines linked above. The proposed fix is to inherit `PeftAdapterMixin` the same way `MiniMaxH3Transformer3DModel` does; packed attention and PDD head selection are unaffected.

### Logs

```text
False
AttributeError: 'MiniMaxH3ImageTransformer2DModel' object has no attribute 'add_adapter'
```

### System Info

```text
- 🤗 Diffusers version: 0.44.0.dev0 (current main c215605); also affected: 0.43.0
- Platform: macOS-15.6-arm64-arm-64bit
- Running on Google Colab?: No
- Python version: 3.11.18
- PyTorch version (GPU?): 2.15.0 (False)
- Huggingface_hub version: 2.4.0
- Transformers version: 5.23.0
- Accelerate version: 1.16.0
- PEFT version: 0.22.0
- Safetensors version: 0.8.0
- xFormers version: not installed
- Accelerator: Apple M3 Pro
- Using GPU in script?: No
- Using distributed or parallel set-up in script?: No
```

### Who can help?

LoRA/PEFT-related model support: @sayakpaul

> **Comment by NancyFyong (2026-11-21T02:13:02Z):**

Opened #15366 with the fix. It adds `PeftAdapterMixin` to `MiniMaxH3ImageTransformer2DModel` and the standard model-level `LoraTesterMixin` coverage; the H3-Image model test file now passes with 45 passed, 39 skipped, 1 xpassed on CPU.


> **Comment by NancyFyong (2026-11-21T02:27:35Z):**

Updated #15366 after checking the `MiniMaxH3Transformer3DModel` integration more closely. The 3D support is not only the mixin: it also wires `attention_kwargs` and `@apply_lora_scale` through `forward`, and the modular pipeline forwards `attention_kwargs` into the denoiser. The PR now includes those pieces, plus explicit tests for LoRA scale semantics, a backward pass, and end-to-end pipeline scale routing. Pipeline-level `load_lora_weights` and checkpoint conversion remain separate work.



### #13375 [open] Investigate the current DtoH sync solution in pipelines
URL: https://github.com/huggingface/diffusers/issues/13375  by sayakpaul 2026-04-01T03:03:24Z  comments: 2

looks good, still feel like we need to broaden the scope of that fix within diffusers at some point :) I'll be out on sabbatical for the next month but I can help when I get back

_Originally posted by @jbschlosser in https://github.com/huggingface/diffusers/pull/13356#discussion_r3019482381_
            

> **Comment by wunianze666-netizen (2026-09-20T17:15:20Z):**

I investigated a currently uncovered DtoH-sync path in `Cosmos3OmniPipeline` on current main (`80c7ed262`). The denoising path still has three independent scalar round-trips:

- `_mask_velocity_predictions()` branches on `mask.sum() > 0`, which emits `aten::_local_scalar_dense` once per active modality and cond/uncond pass.
- each scheduler timestep goes through `t.item()` and a new `torch.full(...)` before being sent back to the model device.
- the first scheduler step searches for its timestep index even though this pipeline always starts at index zero.

I prepared a focused patch at https://github.com/wunianze666-netizen/diffusers/commit/169ca7974bd9e62cbbc8c6af2fca60dbd09c0c03. It keeps masking tensor-native with `torch.where` (including the prior all-conditioned/NaN-to-zero behavior), sets the scheduler begin index after schedule construction, and expands the device timestep without copying it through Python.

Executable evidence:
- before the production change, the mask regression observed 3 `aten::_local_scalar_dense` calls; the scheduler-search guard failed; and expanded transformer timesteps did not share scheduler storage
- after the change, both focused regressions pass
- full `tests/pipelines/cosmos/test_cosmos3.py`: 22 passed, 20 skipped
- Ruff check/format and `git diff --check` pass
- self-review found no blocking, non-blocking, or dead-code findings

I did not open another PR yet because this is a focused slice of the broader #13375 work. Would this Cosmos3-specific scope be useful as a PR?


> **Comment by wunianze666-netizen (2026-09-21T17:57:10Z):**

I found a second, independent #13375 slice in MiniMax-H3 and prepared a verified patch at https://github.com/wunianze666-netizen/diffusers/commit/be8f94790.

`MiniMaxH3SetTimestepsStep` converted three device scalars to Python on every model-evaluation step, rebuilt the packed-row timestep map on CPU, then copied it back to the model device. Both schedulers also searched for their known first index and extracted it with `.item()`.

The patch keeps the computation device-native with a fixed four-role timestep table (generated video, generated audio, conditioned video, conditioned audio) and sets both scheduler begin indices to zero. Duplicate table entries are intentionally retained when roles share a timestep; the regression verifies that every packed row still selects exactly its previous value.

Verification:
- before: timestep setup emitted 9 `aten::_local_scalar_dense` calls for the 4-point test schedule
- after: setup plus both schedulers' first steps emit 0 scalar extractions
- T2VA and FL2VA generator-reproducibility tests pass
- final focused run: 3 passed, 132 deselected
- Ruff, formatting, `git diff --check`, secret/path scan, and project self-review pass (0 blocking, 0 non-blocking findings; READY)

Per the AI-assisted contribution guide, I have not opened a PR without explicit maintainer acknowledgment. Would this MiniMax-H3 scope be welcome as a focused PR? I can keep the separate Cosmos3 slice from my earlier comment parked until its scope is also confirmed.



### #14882 [closed] MiniMax-H3 video decode only enables fp16 autocast on CUDA, so the block silently runs in fp32 on other accelerators
URL: https://github.com/huggingface/diffusers/issues/14882  by li-lizhe 2026-09-27T14:48:50Z  comments: 2

### Describe the bug

`MiniMaxH3VideoDecodeStep.__call__` (`src/diffusers/modular_pipelines/minimax_h3/decoders.py`, line ~187) wraps the VAE decode in an fp16 autocast whose gate is CUDA-only:

```python
with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
    video = components.vae.decode(latents, return_dict=False)[0]
```

The block's own `description` states the intent - "the decode itself runs under float16 autocast even though the VAE weights are float32" - but with `enabled=device.type == "cuda"` that is only true on CUDA. On every other accelerator (Ascend NPU, Intel XPU, Apple MPS) autocast is silently disabled, so the same workflow decodes with the VAE's fp32 weights: slower, and numerically on a different path than the documented CUDA behaviour.

Suggested change - gate on "not CPU" instead of "CUDA only":

```python
enabled=device.type != "cpu",
```

CPU keeps the current behaviour (autocast disabled).

### Reproduction

On Ascend 910B2 NPU (torch 2.15.0.dev20260917+cpu + `torch_npu`), the `enabled` flag is the only thing keeping the NPU on the fp32 path:

```python
import torch

x = torch.randn(8, 8, device="npu", dtype=torch.float32)
with torch.autocast(device_type="npu", dtype=torch.float16, enabled=True):
    print((x @ x).dtype)   # torch.float16
with torch.autocast(device_type="npu", dtype=torch.float16, enabled=False):
    print((x @ x).dtype)   # torch.float32
```

### Not measured

End-to-end NPU decode speed-up of the change is **not** measured here - no A/B benchmark of the decode block was run, only the dtype-path check above. CUDA behaviour is unchanged by the linked PR.

### Related but distinct

#14746 reports a trade-off in the opposite direction on CUDA (fp32 VAE weights + fp16 decode costs VRAM and forces a downcast). This issue is only about the device gate being CUDA-only; it does not argue about whether the fp16 autocast should exist at all.

### System Info

- diffusers: main
- hardware: Ascend 910B2 NPU (torch 2.15.0.dev20260917+cpu + `torch_npu`); applies to any non-CUDA accelerator.


> **Comment by li-lizhe (2026-09-28T10:28:25Z):**

Resolved upstream by #14754 (commit 51a454be): the fix was to **drop** the fp16 autocast around the video decode block and run it at the dtype you pass, rather than enabling autocast on more accelerators — the PR's error table shows the previous fp32-weights + fp16-autocast path was the inaccurate one. The premise of this issue no longer holds, so I'm closing it.



> **Comment by li-lizhe (2026-09-28T10:30:25Z):**

Resolved upstream by #14754 (commit 51a454be): the fix was to **drop** the fp16 autocast around the video decode block and run it at the dtype you pass, rather than enabling autocast on more accelerators — the PR's error table shows the previous fp32-weights + fp16-autocast path was the inaccurate one. The premise of this issue no longer holds, so I'm closing it.




### #15341 [open] Standardize how audio pipelines expose their output sample rate
URL: https://github.com/huggingface/diffusers/issues/15341  by yiyixuxu 2026-11-19T02:15:25Z  comments: 1

An audio pipeline returns a bare waveform, so anything that saves it has to find the sample rate somewhere else. Today each pipeline keeps it in a different place:

| Where | Pipelines |
|---|---|
| `pipe.vocoder.config.output_sampling_rate` | LTX-2 |
| `pipe.vae.config.sampling_rate` | Stable Audio |
| `pipe.audio_sampling_rate` | LTX-2, MiniMax-H3, Pika H3 Flex, Echo (modular) |
| `pipe.sound_sampling_rate` | Cosmos (modular) |
| `pipe.sampling_rate` | MiniMax Music 3 (modular) |

The same name also doesn't always mean the same thing: in LTX-2, `audio_sampling_rate` is the audio VAE's rate (16000), while the returned waveform is at `vocoder.config.output_sampling_rate` (24000).

This makes generic tooling guess:

- `diffusers-cli run` falls back to 16000 unless the user passes `--sampling-rate`.
- `diffusers serve` (#15017) tries a list of six attribute paths in order.

### Proposal

Pick one way for a pipeline to say what rate its output audio is at. Two options:

1. A single property on every audio pipeline (e.g. `audio_sampling_rate`) that always means the rate of the returned waveform.
2. Return the rate with the output: a `sampling_rate` field on the output class, or an output param next to `audio` for modular pipelines.

(1) is a small change and modular pipelines are most of the way there. (2) is more robust since the rate travels with the data.

*Posted by Claude on behalf of @yiyixuxu. The text was drafted together and approved by yiyixuxu.*


> **Comment by yiyixuxu (2026-11-19T02:15:53Z):**

cc @DN6 



### #14403 [open] convrot for int8 and fp4
URL: https://github.com/huggingface/diffusers/issues/14403  by ghost 2026-08-06T14:27:20Z  comments: 1

**Is your feature request related to a problem? Please describe.**

to use https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors

**Describe the solution you'd like.**

https://arxiv.org/pdf/2512.03673

**Describe alternatives you've considered.**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context.**
https://github.com/feice-huang/ConvRot
https://github.com/BobJohnson24/ComfyUI-INT8-Fast

> **Comment by PrakshaaleJain (2026-08-08T16:02:56Z):**

Hey, if anyone is not working on this, I can pick this up.




### #15351 [open] Derive the GMS manifest from modular block declarations
URL: https://github.com/huggingface/diffusers/issues/15351  by yiyixuxu 2026-11-19T06:33:16Z  comments: 1

#15017 adds `diffusers serve`, which describes a model with a manifest: its tasks, inputs, parameters and outputs.

Right now modular pipelines go through the same path as standard pipelines: the blocks are converted into a `__call__`-like signature, and the manifest is guessed from argument names and type hints. That makes sense for standard pipelines, where the signature is all we have. But modular blocks are declarative, and we shouldn't need to guess for our own pipelines. I think the blocks can state most of this directly. A few ideas to start with:

**1. `InputParam.metadata`**
`InputParam` already has a `metadata` dict (Mellon uses it). It could carry what the manifest needs:
- the public name (`num_inference_steps` → `steps`)
- the kind: text, image, video, audio, mask, or a plain setting
- whether it's user-facing at all (`latents`, `timesteps` are not)

**2. Tasks from workflows**
Tasks are guessed from whether there's an `image` / `video` input, so a model gets at most two. Blocks already declare their workflows and trigger inputs in `_workflow_map`. One task per workflow would also cover workflows triggered by other inputs, e.g. `references` in MiniMax-H3.

**3. Require `type_hint` on user-facing inputs**
A parameter's schema comes from its type hint (`int` → integer, `list[float]` → array of numbers). An input without one can't be described and is dropped today. We could require a `type_hint` on every `InputParam` that's exposed, and check it in the tests.

**4. Reuse it in `save_pretrained()`**
If the manifest comes from the blocks alone, the same function can run in `save_pretrained()` and write a `gms.json` next to `modular_model_index.json`. `serve` already picks up a `gms.json` from the model repo, so new models would ship with their manifest.

**What the code could look like**
A made-up sketch to show the idea, not tested. The `"gms"` key and its fields are placeholders.

```python
# in a block
InputParam("num_inference_steps", type_hint=int, default=28, metadata={"gms": {"name": "steps"}})
InputParam("image", type_hint=PIL.Image.Image, metadata={"gms": {"type": "image"}})
InputParam("latents", type_hint=torch.Tensor)  # no metadata, not exposed
```

```python
def derive_manifest(model_id, blocks):
    tasks = {}
    for workflow, triggers in blocks._workflow_map.items():
        # the blocks that run for this workflow, so only its own inputs are listed
        workflow_blocks = blocks.get_workflow(workflow)

        inputs, parameters = {}, {}
        for param in workflow_blocks.inputs:
            gms = (param.metadata or {}).get("gms")
            if gms is None:  # not user-facing
                continue
            name = gms.get("name", param.name)

            if gms.get("type") in ("text", "image", "video", "audio", "mask"):
                inputs[name] = {"type": gms["type"], "required": param.required or param.name in triggers}
            else:
                schema = type_hint_to_schema(param.type_hint)  # int -> {"type": "integer"}
                if param.default is not None:
                    schema["default"] = param.default
                parameters[name] = schema

        tasks[workflow] = {"inputs": inputs, "parameters": parameters}

    return {"id": model_id, "tasks": tasks}
```

No rush since the PR is still WIP, opening this to collect ideas.

*Posted by Claude on behalf of @yiyixuxu. The text was drafted together and approved by yiyixuxu.*


> **Comment by yiyixuxu (2026-11-19T07:06:46Z):**

cc @DN6 



### #14376 [open] Secret Leaks workflow fails on every new-branch push (6 of the last 30 runs)
URL: https://github.com/huggingface/diffusers/issues/14376  by 4ktLuffy 2026-08-03T18:23:25Z  comments: 1

### Describe the bug

The `Secret Leaks` workflow fails on every push that creates a new branch, in this repo and in every fork. It is **6 of the last 30 runs on `huggingface/diffusers` itself** — `cli-autooffload`, `minimax-h3-refactor`, `group-offload-auto-offload`, `minimax-h3`, `modular-docstring-configs-oneperline`, and a dependabot branch. Pushes to `main` pass.

Every failure is the same two annotations:

```
scripts/convert_vq_diffusion_to_diffusers.py#L11
scripts/convert_vq_diffusion_to_diffusers.py#L20
Found unverified AzureSasToken result with HTML encoding
```

### Why it only happens on new branches

The action picks its scan range from the push event. On a new branch `github.event.before` is all zeros, and the action's own script maps that to an empty base, which means "scan the whole history":

```sh
if [ "push" == "push" ]; then
    HEAD=<sha>
    if [ 0000000000000000000000000000000000000000 == "0000000000000000000000000000000000000000" ]; then
      BASE=""
    else
      BASE=0000000000000000000000000000000000000000
    fi
fi
...
docker run ... trufflehog git file:///tmp/ --since-commit ${BASE:-''} --branch ${HEAD:-''} --fail ...
```

From the logs, the difference is visible directly:

- failing run [30836004171](https://github.com/huggingface/diffusers/actions/runs/30836004171) (new branch): `BASE=""`
- passing run [30833246954](https://github.com/huggingface/diffusers/actions/runs/30833246954) (`main`): `BASE=f83ba3b7cd758112a36d1c0d63fe6864295ea05b`

So a push to `main` scans only the new commits, while a new branch rescans back to the first commit and reaches a file added in 2022.

### The finding itself is real

Worth stating plainly, because it changes what the right fix is: this is not a scanner false positive. `scripts/convert_vq_diffusion_to_diffusers.py` documents a `wget` command whose URL carries a genuine Azure SAS token, with `se=2030-05-31` and `sp=r`. It is third-party (Microsoft's VQ-Diffusion model hosting), read-only, and has been public in this repo since 2022, so the exposure is not this project's — but the detector is correct to flag it.

Two consequences:

1. The `# trufflehog:ignore` marker already on that line does not suppress it, because the match comes from the URL-encoded form (the annotation says "with HTML encoding") rather than the plain one.
2. Editing or removing the line would not fix the failures either, since the scan reaches it through history rather than through the working tree.

### Suggested fix

Give the action an explicit base so branch pushes scan only their own commits, which is what the `main` path already does:

```yaml
      with:
        base: ${{ github.event.repository.default_branch }}
        head: HEAD
        extra_args: --results=verified,unknown
```

Only the `with:` block changes — I have left the pinned `uses:` SHA out deliberately, since #14362 is bumping it.

That keeps `--results=verified,unknown` and the current detector coverage; it only stops re-scanning history that has already been scanned on `main`. If you would rather keep full-history scanning, the alternative is an exclude entry for that one path, though that is a wider hole.

I am happy to open the PR for whichever you prefer — I did not want to change CI configuration without asking first.

### If this is already known

The workflow has been configured this way since #8430 in June 2024, so it is possible this is understood and tolerated. The reason I think it is still worth fixing: at 6 failures in 30 runs, a red `Secret Leaks` badge is the normal state for anyone pushing a branch, which is the condition under which a genuine future finding gets waved through. That risk is what makes it more than cosmetic.

### Reproduction

Push any new branch to a fork or to this repo and watch the `Secret Leaks` workflow. It fails at the `trufflehog` step with exit code 183.

### System Info

Not environment-specific; it is a CI configuration issue. Observed on `main` at `6f2010e8b`.

### Who can help?

Anyone maintaining CI for this repo.


> **Comment by Saad-Mallebhari (2026-08-06T17:59:14Z):**

Hi @4ktLuffy

Thanks for the detailed issue report and the exact CI log analysis.

Root Cause

In `.github/workflows/secret_leaks.yml`, the TruffleHog scanner action omits explicit `base` and `head` parameters. On new branch pushes, `github.event.before` is set to a zero-hash (`0000000000000000000000000000000000000000`), causing TruffleHog to fall back to an empty base range (`BASE=""`) and scan full repository history back to initial commit. During full historical scanning, TruffleHog detects a public 2022 Azure SAS token in `scripts/convert_vq_diffusion_to_diffusers.py`, causing Secret Leaks workflow failures on every new branch push.

Proposed Solution

1. In `.github/workflows/secret_leaks.yml`, explicitly set `base` and `head` parameters to bound branch scans against the repository's default branch (`main`):
```yaml
with:
  base: ${{ github.event.repository.default_branch }}
  head: HEAD
  extra_args: --results=verified,unknown
```
2. Add an explicit ignore entry in `.trufflehogignore` for `scripts/convert_vq_diffusion_to_diffusers.py` to prevent legacy URL-encoded SAS token detections if full-history scans are triggered.
3. Validate `.github/workflows/secret_leaks.yml` workflow syntax.

Would you like me to open a Pull Request with this Secret Leaks CI workflow fix and accompanying configuration updates?



### #15009 [open] [modular] `SequentialPipelineBlocks` gets `required` wrong in both `inputs` and `required_inputs`
URL: https://github.com/huggingface/diffusers/issues/15009  by yiyixuxu 2026-10-11T22:25:52Z  comments: 1


## What happens

A `SequentialPipelineBlocks` has two views of its inputs, and each gets the `required` flag wrong in a different way.

**1. `required_inputs` keeps names that an earlier sub-block produces.** In `Flux2CoreDenoiseStep`, `prepare_latents` declares `latents` as optional and produces it, and `set_timesteps` does the same for `timesteps` and `num_inference_steps`. The denoise loop then requires all three. `inputs` says optional, `required_inputs` says required:

```python
from diffusers.modular_pipelines.flux2.modular_blocks_flux2 import Flux2CoreDenoiseStep

core = Flux2CoreDenoiseStep()
print([(i.name, i.required) for i in core.inputs if i.name in ("latents", "timesteps", "num_inference_steps")])
print(sorted(core.required_inputs))
```

```
[('latents', False), ('num_inference_steps', False), ('timesteps', False)]
['batch_size', 'guidance', 'latent_ids', 'latents', 'num_inference_steps', 'prompt_embeds', 'timesteps', 'txt_ids']
```

Nothing produced those values from the user's point of view either way, but the first view is the right one here: the block fills them in itself.

**2. `inputs` keeps the first declaration's flag even when a later sub-block requires the value and nobody produces it.** In `StableDiffusion3I2ICoreDenoiseStep`, `additional_inputs` declares `image_latents` as optional, `prepare_img2img_latents` requires it, and no sub-block produces it. The user has to pass it, so it is required. `inputs` says optional, `required_inputs` says required:

```python
from diffusers.modular_pipelines.stable_diffusion_3.modular_blocks_stable_diffusion_3 import StableDiffusion3I2ICoreDenoiseStep

core = StableDiffusion3I2ICoreDenoiseStep()
print([(i.name, i.required) for i in core.inputs if i.name == "image_latents"])
print("image_latents" in core.required_inputs)
```

```
[('image_latents', False)]
True
```

Here the second view is the right one.

`LoopSequentialPipelineBlocks` has the same two problems.

## Why

`inputs` walks the sub-blocks in order, skips a name once an earlier sub-block produced it, and otherwise keeps the first declaration as is. It never looks at what later sub-blocks require.

`required_inputs` is computed separately, as a plain union of every sub-block's `required_inputs`. It never looks at what earlier sub-blocks produce.

## Effect

The docstrings are rendered from `inputs`, so they are wrong in case 2 directly. In case 1 the wrong flag reaches the docstring through blocks that rebuild `required` from their sub-blocks' `required_inputs`, which is why the `*optional*` marker is missing from `latents`, `timesteps` and `num_inference_steps` in the Flux2 auto blocks and in Anima, Cosmos, Helios, Hunyuan Video 1.5, LTX, LTX-2, MiniMax H3, QwenImage, SD3, SDXL, Wan and Z-Image. Anything that reads `required_inputs` directly, such as the Mellon node config, gets case 1 wrong too.

## Fix

Decide `required` in the single ordered walk: a name is required if some sub-block requires it, no earlier sub-block produced it, and no earlier declaration gives it a default. Derive `required_inputs` from that, so there is only one answer. PR to follow.

---

_Posted by Claude Code on behalf of @yiyixuxu, who reviewed and approved this text._


> **Comment by sayakpaul (2026-10-12T03:06:05Z):**

Just for my own understanding:

* First view: `core.inputs` — input specifications, each with a name and a .required flag.
* Second view: `core.required_inputs` — the set of names the block considers mandatory.



### #14985 [closed] MiniMax-H3: _decode_video_file turns ±90° display-rotated videos upside down
URL: https://github.com/huggingface/diffusers/issues/14985  by hsbcelebi 2026-10-07T23:47:22Z  comments: 1

### Describe the bug

`_decode_video_file` in `src/diffusers/modular_pipelines/minimax_h3/references.py` turns clips that carry a ±90° display matrix **upside down**, compared with what ffmpeg's autorotate and every player show. Phone portrait videos are exactly that case (iPhone stores landscape pixels plus a −90° display matrix), so a portrait reference video reaches MiniMax-H3 / Viggle-Animate rotated 180°.

The line in question (v0.41.0; unchanged on `main` since #14355):

```python
turns = round(rotation / 90.0) % 4
if turns:
    frames = np.ascontiguousarray(np.rot90(frames, k=-turns, axes=(1, 2)))
```

With PyAV's `frame.rotation`, the frame that matches ffmpeg's autorotated output is `np.rot90(frames, k=+turns, axes=(1, 2))`. 180° clips are unaffected, because both signs give the same result there. Only ±90° is wrong.

We hit this while integrating Viggle-Animate through the MiniMax-H3 modular pipeline. Our decoder had copied this formula, and a test against ffmpeg's own autorotate caught it. Happy to open a PR flipping the sign if that is welcome.

### Reproduction

Standalone. It needs `av`, `numpy` and an `ffmpeg` binary, and does not import diffusers. It copies the line above verbatim and compares frame 0 against ffmpeg's autorotated decode:

```python
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import numpy as np


def run(*args: str) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True).stdout


def ffmpeg_upright(path: Path, width: int, height: int) -> np.ndarray:
    raw = run("-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-")
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    base = tmp / "base.mp4"
    # 320x192 landscape, grey, a red 96x64 block in the top-left corner.
    run("-f", "lavfi", "-i", "color=c=gray:s=320x192:d=1:r=24", "-vf",
        "drawbox=x=0:y=0:w=96:h=64:color=red:t=fill", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(base))
    failures = 0
    for display_rotation in (90, -90, 180):
        tagged = tmp / f"rot{display_rotation}.mp4"
        run("-display_rotation", str(display_rotation), "-i", str(base), "-c", "copy", str(tagged))
        with av.open(str(tagged)) as container:
            frame = next(container.decode(container.streams.video[0]))
            rotation = frame.rotation
            frames = np.stack([frame.to_ndarray(format="rgb24")])
        turns = round(rotation / 90.0) % 4
        diffusers = np.rot90(frames, k=-turns, axes=(1, 2))[0]
        opposite = np.rot90(frames, k=+turns, axes=(1, 2))[0]
        height, width = diffusers.shape[:2]
        truth = ffmpeg_upright(tagged, width, height)
        err_diffusers = float(np.abs(diffusers.astype(int) - truth.astype(int)).mean())
        err_opposite = float(np.abs(opposite.astype(int) - truth.astype(int)).mean())
        # Codec noise is ~2; a frame turned the wrong way is an order of magnitude more.
        ok = err_diffusers <= err_opposite + 1.0
        failures += not ok
        print(f"-display_rotation {display_rotation:>4}: frame.rotation={rotation:>6} turns={turns} "
              f"mean|diff| vs ffmpeg: k=-turns {err_diffusers:6.2f} · k=+turns {err_opposite:6.2f} "
              f"→ {'k=-turns matches ffmpeg' if ok else 'k=-turns is upside down vs ffmpeg'}")
    print(f"av {av.__version__}, numpy {np.__version__}, "
          + subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0])
    return failures


if __name__ == "__main__":
    sys.exit(main())
```

### Logs

```shell
-display_rotation   90: frame.rotation=    90 turns=1 mean|diff| vs ffmpeg: k=-turns  26.93 · k=+turns   1.87 → k=-turns is upside down vs ffmpeg
-display_rotation  -90: frame.rotation=   -90 turns=3 mean|diff| vs ffmpeg: k=-turns  26.93 · k=+turns   1.87 → k=-turns is upside down vs ffmpeg
-display_rotation  180: frame.rotation=  -180 turns=2 mean|diff| vs ffmpeg: k=-turns   1.87 · k=+turns   1.87 → k=-turns matches ffmpeg
av 15.1.0, numpy 2.5.3, ffmpeg version 8.1.2 Copyright (c) 2000-2026 the FFmpeg developers
```

A mean absolute difference of ~1.9 is codec noise. 26.9 is the frame turned the wrong way.

### System info

- diffusers v0.41.0. The line was read from the tagged source, and `main` has the same line.
- PyAV 15.1.0, NumPy 2.5.3, FFmpeg 8.1.2, macOS (arm64).
- Also seen on Linux (PyAV 15.1, diffusers 0.41.0): our decoder with `k=-turns` failed the same ffmpeg-autorotate comparison, and with `k=+turns` real iPhone portrait clips come out upright on the device.


> **Comment by yiyixuxu (2026-10-08T05:11:39Z):**

@hsbcelebi thanks for reporting the issue! 
I saw @guspan-tanadi already opened a PR, let's work together from there?
would you be able to share a before/after example from a real phone portrait clip? (since I'm not an expert on this subject, it would be really helpful to see) 

