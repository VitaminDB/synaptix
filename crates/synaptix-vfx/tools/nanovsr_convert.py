# NanoVSR .pth (ветвистая RepVGG) → слитые свёртки в safetensors + эталон на маленьком клипе.
# Использование: python -I nanovsr_convert.py IN.pth OUT_DIR
import sys, json, torch, torch.nn.functional as F
from safetensors.torch import save_file

src, out = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location='cpu', weights_only=True)
sd = sd.get('params', sd.get('state_dict', sd))

def fuse_bn(w, p):
    g, b, m, v = sd[p+'.weight'], sd[p+'.bias'], sd[p+'.running_mean'], sd[p+'.running_var']
    std = (v + 1e-5).sqrt()
    return w * (g/std).reshape(-1,1,1,1), b - m*g/std

def rep(p):
    w3, b3 = fuse_bn(sd[p+'.rbr_dense.0.weight'], p+'.rbr_dense.1')
    w1, b1 = fuse_bn(sd[p+'.rbr_1x1.0.weight'], p+'.rbr_1x1.1')
    w = w3 + F.pad(w1, [1,1,1,1]); b = b3 + b1
    if p+'.rbr_identity.weight' in sd:
        cin = w.shape[1]
        idk = torch.zeros(w.shape[0], cin, 3, 3)
        for i in range(w.shape[0]): idk[i, i % cin, 1, 1] = 1
        wi, bi = fuse_bn(idk, p+'.rbr_identity')
        w = w + wi; b = b + bi
    return w.contiguous(), b.contiguous()

C = sd['feat_extract.rbr_dense.0.weight'].shape[0]
nb = sum(1 for k in sd if k.startswith('forward_net.') and k.endswith('.rbr_dense.0.weight'))
t = {}
t['feat.w'], t['feat.b'] = rep('feat_extract')
for i in range(nb):
    t[f'fwd.{i}.w'], t[f'fwd.{i}.b'] = rep(f'forward_net.{i}')
    t[f'bwd.{i}.w'], t[f'bwd.{i}.b'] = rep(f'backward_net.{i}')
t['fusion.w'], t['fusion.b'] = sd['fusion.weight'], sd['fusion.bias']
for n, k in [('up1','upsample1'), ('up2','upsample2')]:
    t[n+'.w'], t[n+'.b'], t[n+'.prelu'] = sd[k+'.conv.weight'], sd[k+'.conv.bias'], sd[k+'.prelu.weight']
t['last.w'], t['last.b'] = sd['conv_last.weight'], sd['conv_last.bias']
t = {k: v.float().contiguous() for k, v in t.items()}
meta = {'arch': 'nanovsr', 'num_feat': str(C), 'num_blocks': str(nb), 'scale': '4'}
import os; os.makedirs(out, exist_ok=True)
save_file(t, os.path.join(out, 'model.safetensors'), metadata=meta)
json.dump({'arch':'nanovsr','num_feat':C,'num_blocks':nb,'scale':4}, open(os.path.join(out,'config.json'),'w'))

# Эталон: та же сеть в слитом виде (deploy) на клипе T×3×H×W, кадры — гладкий шум.
def lrelu(x): return F.leaky_relu(x, 0.1)
def conv(x, n, pad=1): return F.conv2d(x, t[n+'.w'], t[n+'.b'], padding=pad)
def blocks(x, d):
    for i in range(nb): x = lrelu(conv(x, f'{d}.{i}'))
    return x
def forward(x):  # x [T,3,H,W]
    T = x.shape[0]
    feats = lrelu(conv(x, 'feat'))
    fw, p = [], torch.zeros_like(feats[0:1])
    for i in range(T): p = blocks(feats[i:i+1] + p, 'fwd'); fw.append(p)
    bw, p = [None]*T, torch.zeros_like(feats[0:1])
    for i in range(T-1, -1, -1): p = blocks(feats[i:i+1] + p, 'bwd'); bw[i] = p
    outs = []
    for i in range(T):
        f = conv(torch.cat([fw[i], bw[i]], 1), 'fusion', 0)
        o = F.prelu(F.pixel_shuffle(conv(f, 'up1'), 2), t['up1.prelu'])
        o = F.prelu(F.pixel_shuffle(conv(o, 'up2'), 2), t['up2.prelu'])
        o = conv(o, 'last') + F.interpolate(x[i:i+1], scale_factor=4, mode='bilinear', align_corners=False)
        outs.append(o)
    return torch.cat(outs, 0)

torch.manual_seed(0)
T, H, W = 5, 36, 52
x = F.interpolate(torch.rand(T, 3, H//4, W//4), size=(H, W), mode='bilinear', align_corners=False)
x = (x * 255).round() / 255  # как 8-битные кадры декодера
with torch.no_grad(): y = forward(x)
save_file({'x': x.contiguous(), 'y': y.contiguous()}, os.path.join(out, 'ref.safetensors'))
print(f'C={C} blocks={nb} params={sum(v.numel() for v in t.values())} ref y {tuple(y.shape)} mean {y.mean():.4f}')
