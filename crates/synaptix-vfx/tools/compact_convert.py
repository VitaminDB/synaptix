# SRVGGNetCompact (Real-ESRGAN animevideov3, AnimeJaNai) → safetensors + эталон. python -I compact_convert.py IN OUT_DIR ИМЯ
import sys, os, json, torch, torch.nn.functional as F
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loadw import load_sd
src, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
sd = load_sd(src)
idx = sorted({int(k.split('.')[1]) for k in sd if k.startswith('body.')})
convs = [i for i in idx if sd[f'body.{i}.weight'].dim() == 4]
acts = [i for i in idx if sd[f'body.{i}.weight'].dim() == 1]
feat = sd['body.0.weight'].shape[0]
nconv = len(convs) - 2
scale = int((sd[f'body.{convs[-1]}.weight'].shape[0] // 3) ** 0.5)
t = {}
for j, i in enumerate(convs):
    t[f'conv{j}.w'], t[f'conv{j}.b'] = sd[f'body.{i}.weight'], sd[f'body.{i}.bias']
for j, i in enumerate(acts):
    t[f'act{j}'] = sd[f'body.{i}.weight']
t = {k: v.float().contiguous() for k, v in t.items()}
meta = {'arch': 'compact', 'feat': str(feat), 'convs': str(nconv), 'scale': str(scale), 'name': name}
os.makedirs(out, exist_ok=True)
save_file(t, os.path.join(out, 'model.safetensors'), metadata=meta)
json.dump(meta, open(os.path.join(out, 'config.json'), 'w'))
torch.manual_seed(0)
T, H, W = 2, 40, 56
x = F.interpolate(torch.rand(T, 3, H // 4, W // 4), size=(H, W), mode='bilinear', align_corners=False)
x = (x * 255).round() / 255
with torch.no_grad():
    o = x
    for j in range(nconv + 2):
        o = F.conv2d(o, t[f'conv{j}.w'], t[f'conv{j}.b'], padding=1)
        if j < nconv + 1: o = F.prelu(o, t[f'act{j}'])
    y = F.pixel_shuffle(o, scale) + F.interpolate(x, scale_factor=scale, mode='nearest')
save_file({'x': x.contiguous(), 'y': y.contiguous()}, os.path.join(out, '..', f'ref-{os.path.basename(out)}.safetensors'))
print(f'{name}: feat={feat} convs={nconv} scale={scale} acts={len(acts)} y {tuple(y.shape)}')
