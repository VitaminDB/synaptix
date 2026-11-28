# t8star/minimax-h3-4step-turbo-loras-comfyui-exp

Источник: https://huggingface.co/t8star/minimax-h3-4step-turbo-loras-comfyui-exp · скачано 2026-11-28
Загрузок: 0 · лайков: 66 · базовая:  · теги:  · обновлено: 2026-08-09T07:20:42.000Z

---

基于原作者：https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora  
适配comfyui  
目前由于音频和视频采样差异，所以4步会导致爆音，解决办法是提高到8-10步或者用我开源的双时钟采样器  
模型用minimax_h3_fl2va_int8_convrot.safetensors，不能用剪裁的pruned版本  
采样器euler，调度器beta，请严格按照这个来  

使用教程：https://www.bilibili.com/video/BV1hFuE6tEvt/  
youtube：https://www.youtube.com/watch?v=Xj0IZ4Iq2uk  
