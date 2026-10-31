# RealPLKSR (spandrel/neosr) → safetensors с понятными ключами + эталон. python -I plksr_convert.py IN OUT_DIR ИМЯ
import sys, os, json, torch, torch.nn.functional as F
from safetensors.torch import save_file
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, 'arch'))
from loadw import load_sd
from realplksr_arch import RealPLKSR
src, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
sd = load_sd(src)
dim = sd['feats.0.weight'].shape[0]
nb = max(int(k.split('.')[1]) for k in sd if k.startswith('feats.')) - 2
k = sd['feats.1.lk.conv.weight'].shape[2]
pdim = sd['feats.1.lk.conv.weight'].shape[0]
ln = 'feats.1.layer_norm.bias' in sd
dys = 'to_img.init_pos' in sd
last = f'feats.{nb + 2}'
scale = int((sd[last + '.weight'].shape[0] // 3) ** 0.5)
ea = 'feats.1.attn.f.0.weight' in sd
m = RealPLKSR(dim=dim, n_blocks=nb, upscaling_factor=scale, kernel_size=k, split_ratio=pdim / dim, use_ea=ea, norm_groups=4, dysample=dys, layer_norm=ln)
m.load_state_dict(sd, strict=True)
m.eval()
t = {'first.w': sd['feats.0.weight'], 'first.b': sd['feats.0.bias'], 'last.w': sd[last + '.weight'], 'last.b': sd[last + '.bias']}
for i in range(nb):
    p = f'feats.{i + 1}'
    t[f'b{i}.cm1.w'], t[f'b{i}.cm1.b'] = sd[p + '.channel_mixer.0.weight'], sd[p + '.channel_mixer.0.bias']
    t[f'b{i}.cm2.w'], t[f'b{i}.cm2.b'] = sd[p + '.channel_mixer.2.weight'], sd[p + '.channel_mixer.2.bias']
    t[f'b{i}.lk.w'], t[f'b{i}.lk.b'] = sd[p + '.lk.conv.weight'], sd[p + '.lk.conv.bias']
    if ea: t[f'b{i}.ea.w'], t[f'b{i}.ea.b'] = sd[p + '.attn.f.0.weight'], sd[p + '.attn.f.0.bias']
    t[f'b{i}.refine.w'], t[f'b{i}.refine.b'] = sd[p + '.refine.weight'], sd[p + '.refine.bias']
    if ln: t[f'b{i}.ln.g'], t[f'b{i}.ln.b'] = sd[p + '.layer_norm.weight'], sd[p + '.layer_norm.bias']
    else: t[f'b{i}.norm.g'], t[f'b{i}.norm.b'] = sd[p + '.norm.weight'], sd[p + '.norm.bias']
if dys:
    t['dys.offset.w'], t['dys.offset.b'] = sd['to_img.offset.weight'], sd['to_img.offset.bias']
    t['dys.scope.w'] = sd['to_img.scope.weight']
    t['dys.init_pos'] = sd['to_img.init_pos'].reshape(-1)
    t['dys.end.w'], t['dys.end.b'] = sd['to_img.end_conv.weight'], sd['to_img.end_conv.bias']
t = {kk: v.float().contiguous() for kk, v in t.items()}
meta = {'arch': 'realplksr', 'dim': str(dim), 'blocks': str(nb), 'k': str(k), 'pdim': str(pdim), 'ea': '1' if ea else '0', 'groups': '4',
        'layer_norm': '1' if ln else '0', 'scale': str(scale), 'dysample': '1' if dys else '0', 'name': name}
os.makedirs(out, exist_ok=True)
save_file(t, os.path.join(out, 'model.safetensors'), metadata=meta)
json.dump({kk: v for kk, v in meta.items()}, open(os.path.join(out, 'config.json'), 'w'))
torch.manual_seed(0)
T, H, W = 2, 40, 56
x = F.interpolate(torch.rand(T, 3, H // 4, W // 4), size=(H, W), mode='bilinear', align_corners=False)
x = (x * 255).round() / 255
with torch.no_grad():
    y = m(x)
save_file({'x': x.contiguous(), 'y': y.contiguous()}, os.path.join(out, '..', f'ref-{os.path.basename(out)}.safetensors'))
print(f'{name}: dim={dim} blocks={nb} k={k} pdim={pdim} ln={ln} dysample={dys} ea={ea} scale={scale} y {tuple(y.shape)} [{y.min():.3f}, {y.max():.3f}]')
