# Загрузка весов: .pth (torch, только тензоры) или .safetensors; снять обёртки params/params_ema/state_dict и префикс module.
import torch
def load_sd(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location='cpu', weights_only=True)
    for k in ('params_ema', 'params', 'state_dict'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return {(k[7:] if k.startswith('module.') else k): v.float() for k, v in sd.items()}
