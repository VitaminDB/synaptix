# SPAN (spandrel) → слитые свёртки в safetensors + эталон. python -I span_convert.py IN.pth|safetensors OUT_DIR ИМЯ
import sys, os, json, torch, torch.nn.functional as F
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arch'))
from loadw import load_sd
from span_arch import SPAN
src, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
sd = load_sd(src)
C = sd['conv_1.sk.weight'].shape[0]
scale = int((sd['upsampler.0.weight'].shape[0] // 3) ** 0.5)
norm = 'no_norm' not in sd
m = SPAN(num_in_ch=3, num_out_ch=3, feature_channels=C, upscale=scale, norm=norm)
if not norm: sd['no_norm'] = torch.zeros(1)
missing, unexpected = m.load_state_dict(sd, strict=False)
assert not [k for k in missing if 'eval_conv' not in k], missing
m = m.eval().float()
class D: pass
d = D(); d.scale = scale
def fused(c):
    c.update_params()
    return c.eval_conv.weight.detach().clone(), c.eval_conv.bias.detach().clone()
t = {}
t['conv_1.w'], t['conv_1.b'] = fused(m.conv_1)
for i in range(1, 7):
    b = getattr(m, f'block_{i}')
    for j, c in enumerate([b.c1_r, b.c2_r, b.c3_r], 1):
        t[f'block{i}.c{j}.w'], t[f'block{i}.c{j}.b'] = fused(c)
t['conv_cat.w'], t['conv_cat.b'] = m.conv_cat.weight.detach(), m.conv_cat.bias.detach()
t['conv_2.w'], t['conv_2.b'] = fused(m.conv_2)
up = m.upsampler[0]
t['up.w'], t['up.b'] = up.weight.detach(), up.bias.detach()
t = {k: v.float().contiguous() for k, v in t.items()}
C = t['conv_1.w'].shape[0]
meta = {'arch': 'span', 'feat': str(C), 'scale': str(d.scale), 'norm': '1' if m.is_norm else '0', 'name': name}
os.makedirs(out, exist_ok=True)
save_file(t, os.path.join(out, 'model.safetensors'), metadata=meta)
json.dump({'arch': 'span', 'feat': C, 'scale': d.scale}, open(os.path.join(out, 'config.json'), 'w'))
torch.manual_seed(0)
T, H, W = 2, 40, 56
x = F.interpolate(torch.rand(T, 3, H // 4, W // 4), size=(H, W), mode='bilinear', align_corners=False)
x = (x * 255).round() / 255
with torch.no_grad():
    y = m(x)
save_file({'x': x.contiguous(), 'y': y.contiguous()}, os.path.join(out, '..', f'ref-{os.path.basename(out)}.safetensors'))
print(f'{name}: C={C} scale={d.scale} norm={m.is_norm} params={sum(v.numel() for v in t.values())} y {tuple(y.shape)} [{y.min():.3f}, {y.max():.3f}]')
