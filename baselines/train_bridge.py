"""Training entry point for the dose-conditioned bridge network and the single-dose controls."""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_eval import UNetWithTime, SinusoidalEmbedding, compute_psnr, compute_ssim
from baselines.train_baselines import REDCNN, split_slice_ids, OUT_ROOT
from baselines.train_boosters import BridgeAllStepDataset, BlindWrap, InterpDataset

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRECOMPUTED = {
    ('2detect', 'uniform'): f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed",
    ('2detect', 'equal_improvement'): f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed_equal_improvement",
    ('ldct', 'uniform'): f"{DATA_ROOT}/LDCT_dose_bridge/precomputed",
    ('ldct', 'equal_improvement'): f"{DATA_ROOT}/LDCT_dose_bridge/precomputed_equal_improvement",
    ('2detect_realI0', 'geometric'): f"{DATA_ROOT}/2DeteCT_dose_bridge_realI0/precomputed_geometric",
    ('2detect_effI0', 'geometric'): f"{DATA_ROOT}/2DeteCT_dose_bridge_effI0/precomputed_geometric",
    ('2detect_rep5', 'uniform'): f"{DATA_ROOT}/2DeteCT_dose_bridge_rep5/precomputed",
    ('2detect_rep4', 'uniform'): f"{DATA_ROOT}/2DeteCT_dose_bridge_rep4/precomputed",
    ('ldct_rep4', 'uniform'): f"{DATA_ROOT}/LDCT_dose_bridge_rep4/precomputed",
    ('2detect_rep8', 'uniform'): f"{DATA_ROOT}/2DeteCT_dose_bridge_rep8/precomputed",
}
ALPHAS = {'2detect': 0.01, 'ldct': 0.1, '2detect_realI0': 1500.0 / 53000.0,
          '2detect_effI0': 425.0 / 15000.0, '2detect_rep5': 0.01, '2detect_rep4': 0.01, 'ldct_rep4': 0.1, '2detect_rep8': 0.01}


class UNetRes(nn.Module):
    """Paper U-Net + global residual; zero-init output conv => identity at init."""
    def __init__(self, base_ch=64, t_dim=128, no_norm=False):
        super().__init__()
        self.net = UNetWithTime(base_ch=base_ch, t_dim=t_dim)
        if no_norm:
            for m in self.net.modules():
                for name, child in list(m.named_children()):
                    if isinstance(child, nn.GroupNorm):
                        setattr(m, name, nn.Identity())
        nn.init.zeros_(self.net.out.weight)
        nn.init.zeros_(self.net.out.bias)

    def forward(self, x, t):
        return x + self.net(x, t)


class UNetResRef(nn.Module):
    """UNetRes followed by a full-resolution refinement stage: the input and the U-Net output are
    concatenated and passed through a stack of stride-1 residual 3x3 convolutions (no pooling, so fine
    detail is never downsampled), each conditioned on t through the U-Net's own time embedding.
    Zero-init last conv => the refinement stage is the identity at init."""
    def __init__(self, base_ch=64, t_dim=128, ref_ch=96, ref_depth=5):
        super().__init__()
        self.unet = UNetRes(base_ch=base_ch, t_dim=t_dim)
        self.inp = nn.Conv2d(2, ref_ch, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(ref_ch, ref_ch, 3, padding=1) for _ in range(ref_depth)])
        self.tproj = nn.ModuleList([nn.Linear(t_dim, ref_ch) for _ in range(ref_depth)])
        self.out = nn.Conv2d(ref_ch, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        u = self.unet(x, t)
        temb = self.unet.net.t_embed(t)
        h = F.relu(self.inp(torch.cat([x, u], 1)))
        for c, p in zip(self.convs, self.tproj):
            h = h + F.relu(c(h) + p(temb)[:, :, None, None])
        return u + self.out(h)


class REDCNNNoReLU(REDCNN):
    def forward(self, x):
        r1 = x
        out = F.relu(self.conv1(x)); out = F.relu(self.conv2(out)); r2 = out
        out = F.relu(self.conv3(out)); out = F.relu(self.conv4(out)); r3 = out
        out = F.relu(self.conv5(out))
        out = self.tconv1(out) + r3
        out = self.tconv2(F.relu(out)); out = self.tconv3(F.relu(out)) + r2
        out = self.tconv4(F.relu(out)); out = self.tconv5(F.relu(out)) + r1
        return out


class REDCNNT(REDCNN):
    """RED-CNN with a 2-channel input (image, t-plane); global residual on the image channel only."""
    def __init__(self, ch=96):
        super().__init__(ch)
        self.conv1 = nn.Conv2d(2, ch, 5, padding=2)

    def forward(self, x):
        r1 = x[:, :1]
        out = F.relu(self.conv1(x)); out = F.relu(self.conv2(out)); r2 = out
        out = F.relu(self.conv3(out)); out = F.relu(self.conv4(out)); r3 = out
        out = F.relu(self.conv5(out))
        out = self.tconv1(out) + r3
        out = self.tconv2(F.relu(out)); out = self.tconv3(F.relu(out)) + r2
        out = self.tconv4(F.relu(out)); out = self.tconv5(F.relu(out)) + r1
        return F.relu(out)


class TChan(nn.Module):
    """Wrap a blind 2-channel-input net: x -> net(cat[x, t*ones])."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t):
        tt = t.view(-1, 1, 1, 1).float().expand(-1, 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, tt], 1))


class ResBlockNoNorm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x)))


class DRUNet(nn.Module):
    """Zhang et al. (TPAMI 2021) 'DRUNet': U-Net with residual blocks, strided-conv
    down / transposed-conv up, no normalisation, global residual, noise-level map
    as an extra input channel. nc=(64,128,256,512), nb res-blocks per scale."""
    def __init__(self, in_ch=2, nc=(64, 128, 256, 512), nb=4):
        super().__init__()
        self.head = nn.Conv2d(in_ch, nc[0], 3, padding=1)
        self.down = nn.ModuleList()
        for i in range(3):
            self.down.append(nn.Sequential(*[ResBlockNoNorm(nc[i]) for _ in range(nb)],
                                           nn.Conv2d(nc[i], nc[i + 1], 2, stride=2)))
        self.body = nn.Sequential(*[ResBlockNoNorm(nc[3]) for _ in range(nb)])
        self.up = nn.ModuleList()
        for i in range(3, 0, -1):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(nc[i], nc[i - 1], 2, stride=2),
                                         *[ResBlockNoNorm(nc[i - 1]) for _ in range(nb)]))
        self.tail = nn.Conv2d(nc[0], 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)

    def forward(self, xin):
        x = xin[:, :1]
        h = self.head(xin)
        skips = []
        for d in self.down:
            skips.append(h); h = d(h)
        h = self.body(h)
        for u in self.up:
            h = u(h) + skips.pop()
        return x + self.tail(h)


class LayerNorm2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class NAFBlock(nn.Module):
    """Chen et al. (ECCV 2022) NAFNet block: LN -> 1x1 -> 3x3 dw -> SimpleGate -> SCA -> 1x1, + FFN."""
    def __init__(self, c, dw=2, ffn=2):
        super().__init__()
        dwc = c * dw
        self.n1 = LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, dwc, 1)
        self.c2 = nn.Conv2d(dwc, dwc, 3, padding=1, groups=dwc)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dwc // 2, dwc // 2, 1))
        self.c3 = nn.Conv2d(dwc // 2, c, 1)
        self.n2 = LayerNorm2d(c)
        self.f1 = nn.Conv2d(c, c * ffn, 1)
        self.f2 = nn.Conv2d(c * ffn // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    @staticmethod
    def sg(x):
        a, b = x.chunk(2, dim=1)
        return a * b

    def forward(self, x):
        h = self.c2(self.c1(self.n1(x)))
        h = self.sg(h)
        h = h * self.sca(h)
        x = x + self.beta * self.c3(h)
        h = self.f2(self.sg(self.f1(self.n2(x))))
        return x + self.gamma * h


class NAFNet(nn.Module):
    """NAFNet (width 32, enc [2,2,4,8], middle 12, dec [2,2,2,2]) with global residual."""
    def __init__(self, in_ch=2, width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2)):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2)); c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2))); c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.ending = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.ending.weight); nn.init.zeros_(self.ending.bias)

    def forward(self, xin):
        x = xin[:, :1]
        h = self.intro(xin)
        skips = []
        for e, d in zip(self.encs, self.downs):
            h = e(h); skips.append(h); h = d(h)
        h = self.mid(h)
        for u, dcd in zip(self.ups, self.decs):
            h = u(h) + skips.pop(); h = dcd(h)
        return x + self.ending(h)


class DnCNN(nn.Module):
    """Zhang et al. (TIP 2017): 17 conv layers, 64 ch, BN, residual (noise) learning; blind."""
    def __init__(self, depth=17, ch=64):
        super().__init__()
        layers = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.body(x)


class SobelConv(nn.Module):
    """32 fixed 3x3 Sobel filters (4 orientations x 8 copies), each with a learnable scale."""
    def __init__(self, ch=32):
        super().__init__()
        k = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                          [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]], dtype=torch.float32)
        w = k.repeat(ch // 4, 1, 1)[:, None]                          # (ch, 1, 3, 3)
        self.register_buffer('weight', w)
        self.scale = nn.Parameter(torch.ones(ch, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, padding=1)


class EDCNN(nn.Module):
    """Edge-enhancement-based densely connected CNN (Liang et al., ICSP 2020), ~0.08M params."""
    def __init__(self, in_ch=1, out_ch=32, sobel_ch=32, n_blocks=8):
        super().__init__()
        self.sobel = SobelConv(sobel_ch)
        self.blocks = nn.ModuleList()
        c_in = in_ch + sobel_ch
        for i in range(n_blocks):
            last = i == n_blocks - 1
            self.blocks.append(nn.Sequential(nn.Conv2d(c_in, out_ch, 1), nn.LeakyReLU(0.2, inplace=True),
                                             nn.Conv2d(out_ch, 1 if last else out_ch, 3, padding=1)))
            c_in += out_ch
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feats = [x, self.sobel(x)]
        for i, b in enumerate(self.blocks):
            out = b(torch.cat(feats, 1))
            if i < len(self.blocks) - 1:
                out = self.act(out); feats.append(out)
        return F.relu(x + out)


class REDCNNWide(nn.Module):
    """RED-CNN scaled: 128 channels, D=7 conv + 7 deconv (blind). Same skip rule as
    RED-CNN: features after every even conv are added back before the mirrored
    deconv's ReLU; the input is added at the end."""
    def __init__(self, ch=128, depth=7, in_ch=1):
        super().__init__()
        k, p = 5, 2
        self.depth = depth
        self.convs = nn.ModuleList([nn.Conv2d(in_ch if i == 0 else ch, ch, k, padding=p) for i in range(depth)])
        self.deconvs = nn.ModuleList([nn.ConvTranspose2d(ch, 1 if i == depth - 1 else ch, k, padding=p) for i in range(depth)])

    def forward(self, x):
        skips = {0: x[:, :1]}
        out = x
        for i, c in enumerate(self.convs, start=1):
            out = F.relu(c(out))
            if i % 2 == 0 and i < self.depth:
                skips[i] = out
        for j, d in enumerate(self.deconvs, start=1):
            out = d(out)
            src = self.depth - j          # tconv_j pairs with conv_{D-j}
            if src in skips:
                out = out + skips[src]
            if j < self.depth:
                out = F.relu(out)
        return F.relu(out)


class FiLM(nn.Module):
    """Per-channel scale/shift from the time embedding; zero-initialised (identity at init)."""
    def __init__(self, t_dim, ch):
        super().__init__()
        self.lin = nn.Linear(t_dim, 2 * ch)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, h, temb):
        s, b = self.lin(temb).chunk(2, dim=1)
        return h * (1.0 + s[:, :, None, None]) + b[:, :, None, None]


class WideResT(nn.Module):
    """Full-resolution t-conditioned branch with the RED-CNN-wide topology (D 5x5 convs + D 5x5
    deconvs, `ch` channels; features after every even conv are added back before the mirrored
    deconv) and FiLM(t) after every conv / deconv. Returns `out_ch` full-resolution maps
    (no global residual, no final ReLU)."""
    def __init__(self, ch=128, depth=7, t_dim=128, in_ch=1, out_ch=128):
        super().__init__()
        k, p = 5, 2
        self.depth = depth
        self.convs = nn.ModuleList([nn.Conv2d(in_ch if i == 0 else ch, ch, k, padding=p) for i in range(depth)])
        self.deconvs = nn.ModuleList([nn.ConvTranspose2d(ch, out_ch if i == depth - 1 else ch, k, padding=p)
                                      for i in range(depth)])
        self.film_c = nn.ModuleList([FiLM(t_dim, ch) for _ in range(depth)])
        self.film_d = nn.ModuleList([FiLM(t_dim, ch) for _ in range(depth - 1)])

    def forward(self, x, temb):
        skips, out = {}, x
        for i, c in enumerate(self.convs, start=1):
            out = F.relu(self.film_c[i - 1](c(out), temb))
            if i % 2 == 0 and i < self.depth:
                skips[i] = out
        for j, d in enumerate(self.deconvs, start=1):
            out = d(out)
            src = self.depth - j
            if src in skips:
                out = out + skips[src]
            if j < self.depth:
                out = F.relu(self.film_d[j - 1](out, temb))
        return out


def time_mlp(t_dim=128):
    return nn.Sequential(SinusoidalEmbedding(t_dim), nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))


class WideFiLM(nn.Module):
    """Stand-alone full-resolution t-conditioned net (ablation: the wide branch alone):
    x0_hat = x + branch(x, t), last layer zero-initialised (identity at init)."""
    def __init__(self, ch=128, depth=7, t_dim=128):
        super().__init__()
        self.t_embed = time_mlp(t_dim)
        self.branch = WideResT(ch=ch, depth=depth, t_dim=t_dim, in_ch=1, out_ch=1)
        nn.init.zeros_(self.branch.deconvs[-1].weight); nn.init.zeros_(self.branch.deconvs[-1].bias)

    def forward(self, x, t):
        return x + self.branch(x, self.t_embed(t))


def unet_features(net, x, temb):
    """UNetWithTime.forward up to the penultimate full-resolution features (dec1, base_ch channels)."""
    e1 = net.enc1(x, temb)
    e2 = net.enc2(net.pool(e1), temb)
    e3 = net.enc3(net.pool(e2), temb)
    e4 = net.enc4(net.pool(e3), temb)
    b = net.bottleneck(net.pool(e4), temb)
    d4 = net.dec4(torch.cat([net.up4(b), e4], 1), temb)
    d3 = net.dec3(torch.cat([net.up3(d4), e3], 1), temb)
    d2 = net.dec2(torch.cat([net.up2(d3), e2], 1), temb)
    return net.dec1(torch.cat([net.up1(d2), e1], 1), temb)


class HybridT(nn.Module):
    """Dual-scale t-conditioned network: the residual U-Net (4 levels, context) and the
    full-resolution FiLM(t) wide branch run in parallel on x_t; their full-resolution features
    are fused by a small head that predicts the residual to x0 (zero-initialised => identity
    at init). One time embedding (the U-Net's) is shared by both branches."""
    def __init__(self, base_ch=64, t_dim=128, wide_ch=128, wide_depth=7, fuse_ch=64):
        super().__init__()
        self.unet = UNetWithTime(base_ch=base_ch, t_dim=t_dim)
        self.wide = WideResT(ch=wide_ch, depth=wide_depth, t_dim=t_dim, in_ch=1, out_ch=wide_ch)
        self.fuse = nn.Sequential(nn.Conv2d(base_ch + wide_ch, fuse_ch, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(fuse_ch, 1, 1))
        nn.init.zeros_(self.fuse[-1].weight); nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x, t):
        temb = self.unet.t_embed(t)
        u = unet_features(self.unet, x, temb)
        w = self.wide(x, temb)
        return x + self.fuse(torch.cat([u, w], 1))


class WGANVGG_G(nn.Module):
    """WGAN-VGG generator (Yang et al., TMI 2018): 8 conv layers, 3x3, 32 filters, ReLU; last conv -> 1 channel.
    Trained by train_wgan_vgg.py (adversarial + VGG perceptual loss); blind (no t)."""
    def __init__(self, ch=32, depth=8):
        super().__init__()
        layers = []
        for i in range(depth - 1):
            layers += [nn.Conv2d(1 if i == 0 else ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.relu(self.net(x))


class CPCE2D(nn.Module):
    """Conveying-path-based convolutional encoder-decoder module of MAP-NN (Shan et al., Nat. Mach. Intell. 2019):
    3 conv (3x3, 32) + 3 deconv (3x3) with conveying paths; the module output is added to its input."""
    def __init__(self, ch=32):
        super().__init__()
        self.c1 = nn.Conv2d(1, ch, 3, padding=1); self.c2 = nn.Conv2d(ch, ch, 3, padding=1); self.c3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.d1 = nn.ConvTranspose2d(ch, ch, 3, padding=1)
        self.d2 = nn.ConvTranspose2d(2 * ch, ch, 3, padding=1)
        self.d3 = nn.ConvTranspose2d(2 * ch, 1, 3, padding=1)

    def forward(self, x):
        h1 = F.relu(self.c1(x)); h2 = F.relu(self.c2(h1)); h3 = F.relu(self.c3(h2))
        g = F.relu(self.d1(h3))
        g = F.relu(self.d2(torch.cat([g, h2], 1)))
        return x + self.d3(torch.cat([g, h1], 1))


class MAPNN(nn.Module):
    """MAP-NN: T=5 cascaded CPCE modules (separate weights), supervised on the final output."""
    def __init__(self, T=5, ch=32):
        super().__init__()
        self.mods = nn.ModuleList([CPCE2D(ch) for _ in range(T)])

    def forward(self, x):
        for m in self.mods:
            x = m(x)
        return F.relu(x)


class _LN2d(nn.Module):
    """channel LayerNorm per pixel (Restormer 'WithBias')"""
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim)); self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mu = x.mean(1, keepdim=True); var = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + 1e-5) * self.w[None, :, None, None] + self.b[None, :, None, None]


class _MDTA(nn.Module):
    """multi-Dconv head transposed attention (attention over channels)"""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.temp = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.heads, c // self.heads, h * w); k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temp
        attn = attn.softmax(dim=-1)
        o = (attn @ v).reshape(b, c, h, w)
        return self.out(o)


class _GDFN(nn.Module):
    """gated-Dconv feed-forward network"""
    def __init__(self, dim, expansion=2.66):
        super().__init__()
        hid = int(dim * expansion)
        self.pin = nn.Conv2d(dim, hid * 2, 1, bias=False)
        self.dw = nn.Conv2d(hid * 2, hid * 2, 3, padding=1, groups=hid * 2, bias=False)
        self.pout = nn.Conv2d(hid, dim, 1, bias=False)

    def forward(self, x):
        a, g = self.dw(self.pin(x)).chunk(2, dim=1)
        return self.pout(F.gelu(a) * g)


class _RBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = _LN2d(dim); self.attn = _MDTA(dim, heads); self.n2 = _LN2d(dim); self.ffn = _GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Restormer(nn.Module):
    """Restormer (Zamir et al., CVPR 2022), compact configuration: dim 32, blocks [2,3,3,4], heads [1,2,4,8],
    2 refinement blocks, pixel-(un)shuffle resampling, global residual. ~9M parameters."""
    def __init__(self, dim=32, blocks=(2, 3, 3, 4), heads=(1, 2, 4, 8), n_ref=2):
        super().__init__()
        d = dim
        self.embed = nn.Conv2d(1, d, 3, padding=1, bias=False)
        self.enc1 = nn.Sequential(*[_RBlock(d, heads[0]) for _ in range(blocks[0])])
        self.down1 = nn.Sequential(nn.Conv2d(d, d // 2, 3, padding=1, bias=False), nn.PixelUnshuffle(2))
        self.enc2 = nn.Sequential(*[_RBlock(2 * d, heads[1]) for _ in range(blocks[1])])
        self.down2 = nn.Sequential(nn.Conv2d(2 * d, d, 3, padding=1, bias=False), nn.PixelUnshuffle(2))
        self.enc3 = nn.Sequential(*[_RBlock(4 * d, heads[2]) for _ in range(blocks[2])])
        self.down3 = nn.Sequential(nn.Conv2d(4 * d, 2 * d, 3, padding=1, bias=False), nn.PixelUnshuffle(2))
        self.latent = nn.Sequential(*[_RBlock(8 * d, heads[3]) for _ in range(blocks[3])])
        self.up3 = nn.Sequential(nn.Conv2d(8 * d, 16 * d, 3, padding=1, bias=False), nn.PixelShuffle(2))
        self.red3 = nn.Conv2d(8 * d, 4 * d, 1, bias=False)
        self.dec3 = nn.Sequential(*[_RBlock(4 * d, heads[2]) for _ in range(blocks[2])])
        self.up2 = nn.Sequential(nn.Conv2d(4 * d, 8 * d, 3, padding=1, bias=False), nn.PixelShuffle(2))
        self.red2 = nn.Conv2d(4 * d, 2 * d, 1, bias=False)
        self.dec2 = nn.Sequential(*[_RBlock(2 * d, heads[1]) for _ in range(blocks[1])])
        self.up1 = nn.Sequential(nn.Conv2d(2 * d, 4 * d, 3, padding=1, bias=False), nn.PixelShuffle(2))
        self.dec1 = nn.Sequential(*[_RBlock(2 * d, heads[0]) for _ in range(blocks[0])])
        self.refine = nn.Sequential(*[_RBlock(2 * d, heads[0]) for _ in range(n_ref)])
        self.out = nn.Conv2d(2 * d, 1, 3, padding=1, bias=False)

    def forward(self, x):
        e1 = self.enc1(self.embed(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        l = self.latent(self.down3(e3))
        d3 = self.dec3(self.red3(torch.cat([self.up3(l), e3], 1)))
        d2 = self.dec2(self.red2(torch.cat([self.up2(d3), e2], 1)))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return x + self.out(self.refine(d1))


def build(arch):
    if arch == 'mapnn':
        return MAPNN(), False
    if arch == 'restormer':
        return Restormer(), False
    if arch == 'wganvgg':
        return WGANVGG_G(), False
    if arch == 'hybrid':
        return HybridT(base_ch=64), True
    if arch == 'hybrid96':
        return HybridT(base_ch=96), True
    if arch == 'wide_film':
        return WideFiLM(), True
    if arch == 'unet':
        return UNetWithTime(base_ch=64, t_dim=128), True
    if arch == 'unet_res':
        return UNetRes(), True
    if arch == 'unet_res96':
        return UNetRes(base_ch=96), True
    if arch == 'unet_res48':
        return UNetRes(base_ch=48), True
    if arch == 'unet_res_nn':
        return UNetRes(no_norm=True), True
    if arch == 'unet_res_ref':
        return UNetResRef(), True
    if arch == 'unet_res_ref_w':
        return UNetResRef(ref_ch=128, ref_depth=7), True
    if arch == 'unet_res96_ref':
        return UNetResRef(base_ch=96), True
    if arch == 'redcnn':
        return REDCNN(), False
    if arch == 'redcnn_res0':
        return REDCNNNoReLU(), False
    if arch == 'drunet':
        return TChan(DRUNet()), True
    if arch == 'nafnet':
        return TChan(NAFNet()), True
    if arch == 'drunet_blind':
        return DRUNet(in_ch=1), False
    if arch == 'nafnet_blind':
        return NAFNet(in_ch=1), False
    if arch == 'dncnn':
        return DnCNN(), False
    if arch == 'redcnn_wide':
        return REDCNNWide(), False
    if arch == 'redcnn_t':
        return TChan(REDCNNT()), True
    if arch == 'redcnn_wide_t':
        return TChan(REDCNNWide(in_ch=2)), True
    if arch == 'edcnn':
        return EDCNN(), False
    raise ValueError(arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', required=True,
                    choices=['unet', 'unet_res', 'unet_res_nn', 'redcnn', 'redcnn_res0',
                             'drunet', 'nafnet', 'drunet_blind', 'nafnet_blind', 'dncnn', 'redcnn_wide', 'redcnn_t', 'redcnn_wide_t', 'unet_res96', 'unet_res48', 'edcnn', 'unet_res_ref', 'unet_res_ref_w', 'unet_res96_ref', 'hybrid', 'hybrid96', 'wide_film', 'mapnn', 'restormer', 'wganvgg'])
    ap.add_argument('--mode', required=True, choices=['bridge', 'endpoint', 'interp'],
                    help='interp = non-physical matched-MSE interpolation control (InterpDataset, uniform knots)')
    ap.add_argument('--experiment', required=True, choices=list(ALPHAS.keys()))
    ap.add_argument('--schedule', default='uniform',
                    choices=['uniform', 'geometric', 'equal_improvement'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=None,
                    help='default: 80 for bridge, 400 for endpoint (matched updates)')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--tag', default='')
    ap.add_argument('--endpoint_step', type=int, default=4,
                    help='endpoint mode: which precomputed knot (step0..4) to train on; '
                         '4 = t=1 (low-dose endpoint); <4 trains a dose-specialist at that knot')
    ap.add_argument('--extra_schedule', default='',
                    choices=['', 'uniform', 'equal_improvement', 'geometric'],
                    help='bridge mode: also train on the precompute of this second knot schedule. Its thinnings '
                         'are independent realizations at different doses, so the union doubles both the number of '
                         'noise realizations and the t coverage at no storage cost (LDCT has both dirs on disk).')
    ap.add_argument('--knot_weights', default='',
                    help='bridge mode: comma list of integer repeat factors per knot step0..step4 (e.g. 1,1,1,2,2 '
                         'oversamples the two lowest-dose knots); files are repeated in the training list')
    ap.add_argument('--init_from', default='',
                    help='warm start: load the model weights of this checkpoint before training (fine-tuning)')
    ap.add_argument('--test_pre', default=None,
                    help='precompute dir whose TEST split is used for the report '
                         '(default: the training precompute; LDCT uniform has no test slices, '
                         'so pass the EI dir there)')
    ap.add_argument('--train_frac', type=float, default=1.0,
                    help='fraction of training slices (random subset, --frac_seed); epochs scale by 1/frac '
                         'so the number of gradient updates stays matched to the full-data run')
    ap.add_argument('--frac_seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--test_only', action='store_true', help='skip training; load best.pth and run the test phase')
    ap.add_argument('--aug', action='store_true', help='random dihedral augmentation (flips / 90-degree rotations) per sample')
    ap.add_argument('--ema', type=float, default=0.0, help='EMA decay for the evaluated/saved weights (0 = off, e.g. 0.999)')
    ap.add_argument('--blind_t', action='store_true', help='dose-blind control: feed t=1 to a time-conditioned arch at train AND test time (same data, no dose input)')
    args = ap.parse_args()
    if args.epochs is None:
        args.epochs = 80 if args.mode == 'bridge' else 400

    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    alpha = ALPHAS[args.experiment]
    pre = PRECOMPUTED[(args.experiment, args.schedule)]
    if args.mode == 'interp':
        src = PRECOMPUTED[(args.experiment, 'equal_improvement')]
        train_ds = InterpDataset(src, 'train', args.experiment, alpha)
        val_ds = InterpDataset(src, 'val', args.experiment, alpha)
        pre = src
    else:
        train_ds = BridgeAllStepDataset(pre, 'train', args.experiment)
        val_ds = BridgeAllStepDataset(pre, 'val', args.experiment)
    if args.experiment.startswith('2detect') and args.mode != 'interp':
        canon = [f'slice{i:05d}' for i in range(1, 1001)]
        all_files = sorted(glob(os.path.join(pre, '*_step*.npz')))
        for ds, split in ((train_ds, 'train'), (val_ds, 'val')):
            keep = set(split_slice_ids(canon, '2detect', split))
            n0 = len(ds.files)
            ds.files = [f for f in all_files if os.path.basename(f).split('_step')[0] in keep]
            if len(ds.files) != n0:
                print(f"  canonical split enforced for {split}: {n0} -> {len(ds.files)} files "
                      f"({len({os.path.basename(f).split('_step')[0] for f in ds.files})} slices)", flush=True)
    if args.extra_schedule and args.mode == 'bridge':
        ex = PRECOMPUTED[(args.experiment, args.extra_schedule)]
        for ds, split in ((train_ds, 'train'), (val_ds, 'val')):
            n0 = len(ds.files)
            ds.files = ds.files + BridgeAllStepDataset(ex, split, args.experiment).files
            print(f"  + {args.extra_schedule} precompute for {split}: {n0} -> {len(ds.files)} files", flush=True)
    if args.knot_weights and args.mode == 'bridge':
        w = [int(v) for v in args.knot_weights.split(',')]
        def step_of(f):
            return int(os.path.basename(f).split('_step')[1].split('_')[0].split('.')[0])
        n0 = len(train_ds.files)
        train_ds.files = [f for f in train_ds.files for _ in range(w[step_of(f)])]
        print(f"  knot weights {w}: train {n0} -> {len(train_ds.files)} files", flush=True)
    if args.mode == 'endpoint':
        suffix = f'_step{args.endpoint_step}'
        for ds in (train_ds, val_ds):
            ds.files = [f for f in ds.files if os.path.basename(f).split('.')[0].split('_r')[0].endswith(suffix)]
        print(f"  endpoint-only filter ({suffix}): train {len(train_ds.files)} / val {len(val_ds.files)} pairs")
    if args.train_frac < 1.0:
        sids = sorted({os.path.basename(f).split('_step')[0] for f in train_ds.files})
        rs = np.random.RandomState(args.frac_seed)
        keep = set(np.array(sids)[rs.permutation(len(sids))[:max(1, int(round(args.train_frac * len(sids))))]])
        sub = [f for f in train_ds.files if os.path.basename(f).split('_step')[0] in keep]
        train_ds.files = sub * int(round(1.0 / args.train_frac))
        print(f"  train_frac={args.train_frac}: {len(keep)} slices / {len(sub)} distinct samples x"
              f"{int(round(1.0 / args.train_frac))} repeats = {len(train_ds)} per epoch (frac_seed {args.frac_seed})")
    model, timed = build(args.arch)
    model = model.to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device)
        model.load_state_dict(sd['model'])
        print(f"  warm start from {args.init_from} (epoch {sd.get('epoch')}, val {sd.get('val_loss')})", flush=True)

    name = f'{args.experiment}_{args.arch}_{args.mode}_{args.schedule}'
    if args.mode == 'endpoint' and args.endpoint_step != 4:
        name += f'_k{args.endpoint_step}'
    if args.seed != 42:
        name += f'_seed{args.seed}'
    if args.train_frac < 1.0:
        name += f'_frac{args.train_frac:g}' + (f'fs{args.frac_seed}' if args.frac_seed else '')
    if args.aug:
        name += '_aug'
    if args.ema > 0:
        name += '_ema'
    if args.lr != 1e-4:
        name += f'_lr{args.lr:g}'
    if args.batch_size != 64:
        name += f'_bs{args.batch_size}'
    if args.extra_schedule:
        name += f'_plus{args.extra_schedule[:2]}'
    if args.knot_weights:
        name += '_kw' + args.knot_weights.replace(',', '')
    if args.init_from:
        name += '_ft'
    if args.blind_t:
        name += '_blindt'
    if args.tag:
        name += f'_{args.tag}'
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 70)
    print(f"R6: {name}  alpha={alpha:.4f} epochs={args.epochs} data={pre}")
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  out: {out_dir}")
    print("=" * 70, flush=True)

    if args.smoke:
        train_ds.files = train_ds.files[:64]; val_ds.files = val_ds.files[:16]; args.epochs = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=(len(train_ds) >= 2 * args.batch_size),
                              persistent_workers=True)
    val_every = max(1, args.epochs // 400)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()

    def fwd(x, t, net=None):
        net = model if net is None else net
        if args.blind_t and timed:
            t = torch.ones_like(t)
        return net(x, t) if timed else net(x)

    import copy
    ema_model = copy.deepcopy(model).eval() if args.ema > 0 else None
    if ema_model is not None:
        for q in ema_model.parameters():
            q.requires_grad_(False)

    def ema_update():
        with torch.no_grad():
            for q, p_ in zip(ema_model.parameters(), model.parameters()):
                q.mul_(args.ema).add_(p_.detach(), alpha=1.0 - args.ema)
            for qb, pb in zip(ema_model.buffers(), model.buffers()):
                qb.copy_(pb)

    def dihedral(x_t, x0):
        ks = torch.randint(0, 4, (x_t.shape[0],)); fl = torch.rand(x_t.shape[0]) < 0.5
        xs, ys = [], []
        for i in range(x_t.shape[0]):
            a, b = torch.rot90(x_t[i], int(ks[i]), (1, 2)), torch.rot90(x0[i], int(ks[i]), (1, 2))
            if fl[i]:
                a, b = torch.flip(a, (2,)), torch.flip(b, (2,))
            xs.append(a); ys.append(b)
        return torch.stack(xs), torch.stack(ys)

    eval_net = ema_model if ema_model is not None else model
    best_val, history = float('inf'), []
    if args.test_only:
        args.epochs = 0
        best_val = float(torch.load(os.path.join(out_dir, 'best.pth'), map_location='cpu').get('val_loss', float('nan')))
        print(f"TEST-ONLY: loading existing best.pth (val {best_val:.6f})", flush=True)
    for epoch in range(args.epochs):
        model.train(); tr, n = 0.0, 0
        for batch in train_loader:
            x_t = batch['x_t'].to(device); x0 = batch['x0'].to(device); t = batch['t'].to(device)
            if args.aug:
                x_t, x0 = dihedral(x_t, x0)
            optimizer.zero_grad()
            with autocast():
                pred = fwd(x_t, t)
                loss = F.mse_loss(pred, x0) + 0.1 * F.l1_loss(pred, x0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer); scaler.update()
            if ema_model is not None:
                ema_update()
            tr += loss.item(); n += 1
        scheduler.step()
        if (epoch + 1) % val_every and epoch != args.epochs - 1:
            continue
        model.eval(); vl, m = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred = fwd(batch['x_t'].to(device), batch['t'].to(device), eval_net)
                vl += F.mse_loss(pred, batch['x0'].to(device)).item(); m += 1
        vl /= max(m, 1)
        history.append({'epoch': epoch + 1, 'train': tr / max(n, 1), 'val_mse': vl})
        if epoch % (10 * val_every) == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs}: train={tr/max(n,1):.6f} val={vl:.6f}", flush=True)
        if vl < best_val:
            best_val = vl
            torch.save({'model': eval_net.state_dict(), 'model_raw': model.state_dict() if ema_model is not None else None,
                        'epoch': epoch, 'val_loss': vl, 'aug': args.aug, 'ema': args.ema,
                        'arch': args.arch, 'mode': args.mode, 'experiment': args.experiment,
                        'schedule': args.schedule, 'seed': args.seed},
                       os.path.join(out_dir, 'best.pth'))
    if not args.test_only:
        with open(os.path.join(out_dir, 'train_history.json'), 'w') as f:
            json.dump(history, f)

    model.load_state_dict(torch.load(os.path.join(out_dir, 'best.pth'), map_location=device)['model'])
    model.eval()
    test_pre = args.test_pre or pre
    print(f"TEST precompute: {test_pre}")
    files = sorted(glob(os.path.join(test_pre, '*_step*.npz')))
    by_sid = {}
    for f in files:
        by_sid.setdefault(os.path.basename(f).split('_step')[0], []).append(f)
    exp_key = 'ldct' if args.experiment.startswith('ldct') else '2detect'
    test_ids = split_slice_ids(sorted(by_sid.keys()), exp_key, 'test')
    per_step = {}
    with torch.no_grad():
        for sid in test_ids:
            for f in sorted(by_sid[sid]):
                d = np.load(f)
                x_t, x0, t = d['x_t'].astype(np.float32), d['x0'].astype(np.float32), float(d['t'])
                if not (np.isfinite(x_t).all() and np.isfinite(x0).all()):
                    continue
                xt = torch.from_numpy(x_t)[None, None].to(device)
                tt = torch.tensor([t], dtype=torch.float32).to(device)
                pred = np.clip(fwd(xt, tt).cpu().numpy().squeeze(), 0, 1)
                k = os.path.basename(f).split('_step')[1].split('.')[0]
                e = per_step.setdefault(k, {'t': t, 'psnr': [], 'ssim': [], 'psnr_in': []})
                e['psnr'].append(compute_psnr(pred, x0)); e['ssim'].append(compute_ssim(pred, x0))
                e['psnr_in'].append(compute_psnr(x_t, x0))
    summary = {'name': name, 'best_val': best_val, 'epochs': args.epochs,
               'params_M': sum(p.numel() for p in model.parameters()) / 1e6}
    for k, e in sorted(per_step.items()):
        summary[f'step{k}'] = {'t': e['t'], 'n': len(e['psnr']),
                               'psnr_in': float(np.mean(e['psnr_in'])),
                               'psnr': float(np.mean(e['psnr'])), 'psnr_std': float(np.std(e['psnr'])),
                               'ssim': float(np.mean(e['ssim']))}
        print(f"TEST step{k} t={e['t']:.3f}: in {np.mean(e['psnr_in']):.2f} -> "
              f"{np.mean(e['psnr']):.2f}±{np.std(e['psnr']):.2f} dB, SSIM {np.mean(e['ssim']):.4f}")
    with open(os.path.join(out_dir, 'test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
