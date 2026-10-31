# RIFE (Practical-RIFE / vs-rife flownet_v*.pkl) → safetensors + эталон t=0.5. python -I rife_convert.py IN.pkl OUT_DIR ВЕРСИЯ(4.25|4.25.lite|4.26)
import sys, os, json, math, importlib, torch, torch.nn.functional as F
from safetensors.torch import save_file
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, 'arch', 'rife'))
from loadw import load_sd
src, out, ver = sys.argv[1], sys.argv[2], sys.argv[3]
mod = importlib.import_module({'4.25': 'IFNet_HDv3_v4_25', '4.25.lite': 'IFNet_HDv3_v4_25_lite', '4.26': 'IFNet_HDv3_v4_26'}[ver])
modulo = 128 if ver.endswith('lite') else 64
sd = {k: v for k, v in load_sd(src).items() if not k.startswith(('teacher.', 'caltime.'))}
net = mod.IFNet(); head = mod.Head()
net.load_state_dict({k: v for k, v in sd.items() if not k.startswith('encode.')}, strict=False)
head.load_state_dict({k[7:]: v for k, v in sd.items() if k.startswith('encode.')})
net.eval(); head.eval()
t = {k: v.float().contiguous() for k, v in sd.items()}
lite = ver.endswith('lite')
meta = {'arch': 'rife', 'version': ver, 'modulo': str(modulo), 'scales': '32,16,8,4,1' if lite else '16,8,4,2,1', 'name': f'RIFE {ver}'}
os.makedirs(out, exist_ok=True)
save_file(t, os.path.join(out, 'model.safetensors'), metadata=meta)
json.dump(meta, open(os.path.join(out, 'config.json'), 'w'))
# эталон: два «кадра» со сдвигом (гладкий шум), размер — кратный modulo
torch.manual_seed(0)
H, W = 128, 256
base = F.interpolate(torch.rand(1, 3, H // 8, W // 8 + 4), size=(H, W + 32), mode='bilinear', align_corners=False)
img0 = (base[:, :, :, 0:W] * 255).round() / 255
img1 = (base[:, :, :, 6:W + 6] * 255).round() / 255
div = torch.tensor([(W - 1.0) / 2.0, (H - 1.0) / 2.0])
gx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(1, 1, H, W)
gy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(1, 1, H, W)
grid = torch.cat([gx, gy], 1)
with torch.no_grad():
    f0, f1 = head(img0), head(img1)
    ts = torch.full([1, 1, H, W], 0.5)
    y = net(img0, img1, ts, div, grid, f0, f1)
save_file({'img0': img0.contiguous(), 'img1': img1.contiguous(), 'y': y.contiguous(), 'f0': f0.contiguous()}, os.path.join(out, '..', f'ref-{os.path.basename(out)}.safetensors'))
print(f'RIFE {ver}: тензоров {len(t)}, y {tuple(y.shape)} [{y.min():.3f}, {y.max():.3f}]')
