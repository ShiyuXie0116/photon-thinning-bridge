"""WGAN-VGG baseline, trained with its published patch recipe."""
import os, sys, json, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from glob import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_eval import compute_psnr, compute_ssim
from baselines.train_baselines import split_slice_ids, OUT_ROOT
from baselines.train_boosters import BridgeAllStepDataset
from baselines.train_bridge import PRECOMPUTED, ALPHAS, WGANVGG_G


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        chs = [(1, 64, 1), (64, 64, 2), (64, 128, 1), (128, 128, 2), (128, 256, 1), (256, 256, 2)]
        self.convs = nn.ModuleList([nn.Conv2d(i, o, 3, stride=s, padding=1) for i, o, s in chs])
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        h = x
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
        h = h.mean((2, 3))                      # global average -> 256 (input size independent)
        return self.fc2(F.leaky_relu(self.fc1(h), 0.2))


class VGGFeat(nn.Module):
    """VGG-19 features up to the output of the 16th conv layer (conv5_4, ImageNet weights)."""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.net = nn.Sequential(*list(vgg.children())[:35]).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                     # x in [0,1], 1 channel
        return self.net((x.repeat(1, 3, 1, 1) - self.mean) / self.std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True, choices=list(ALPHAS.keys()))
    ap.add_argument('--schedule', default='uniform')
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--n_critic', type=int, default=4)
    ap.add_argument('--lambda_gp', type=float, default=10.0)
    ap.add_argument('--lambda_vgg', type=float, default=0.1)
    ap.add_argument('--endpoint_step', type=int, default=4)
    ap.add_argument('--test_pre', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--patch', type=int, default=0, help='train on random PxP patches (paper: 64), 0 = full images')
    ap.add_argument('--patches_per_image', type=int, default=8)
    ap.add_argument('--max_iters', type=int, default=0, help='stop after this many generator iterations (paper: 100k)')
    args = ap.parse_args()
    device = torch.device('cuda')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    pre = PRECOMPUTED[(args.experiment, args.schedule)]
    train_ds = BridgeAllStepDataset(pre, 'train', args.experiment)
    val_ds = BridgeAllStepDataset(pre, 'val', args.experiment)
    suffix = f'_step{args.endpoint_step}'
    for ds in (train_ds, val_ds):
        ds.files = [f for f in ds.files if os.path.basename(f).split('.')[0].split('_r')[0].endswith(suffix)]
    print(f"  endpoint-only ({suffix}): train {len(train_ds.files)} / val {len(val_ds.files)} pairs", flush=True)
    name = f'{args.experiment}_wganvgg_endpoint_{args.schedule}' + (f'_{args.tag}' if args.tag else '')
    out_dir = os.path.join(OUT_ROOT, name); os.makedirs(out_dir, exist_ok=True)
    if args.smoke:
        train_ds.files = train_ds.files[:32]; val_ds.files = val_ds.files[:8]; args.epochs = 2
    print(f"WGAN-VGG: {name}  epochs={args.epochs} lr={args.lr} n_critic={args.n_critic} "
          f"lambda_gp={args.lambda_gp} lambda_vgg={args.lambda_vgg} data={pre}", flush=True)

    G, D, V = WGANVGG_G().to(device), Critic().to(device), VGGFeat().to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                    drop_last=True, persistent_workers=True)
    vl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    def grad_penalty(real, fake):
        a = torch.rand(real.size(0), 1, 1, 1, device=device)
        xi = (a * real + (1 - a) * fake).requires_grad_(True)
        d = D(xi)
        g = torch.autograd.grad(d.sum(), xi, create_graph=True)[0]
        return ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

    best_val, history, it = float('inf'), [], 0
    epoch = -1
    while epoch + 1 < args.epochs:
        epoch += 1
        G.train(); D.train(); lg = ld = lv = 0.0; n = 0
        for batch in tl:
            x1 = batch['x_t'].to(device); x0 = batch['x0'].to(device)
            if args.patch:
                P, ppi = args.patch, args.patches_per_image
                B, _, H, W = x1.shape
                ii = torch.randint(0, H - P + 1, (B * ppi,)); jj = torch.randint(0, W - P + 1, (B * ppi,))
                bb = torch.arange(B).repeat_interleave(ppi)
                x1 = torch.stack([x1[b, :, i:i + P, j:j + P] for b, i, j in zip(bb, ii, jj)])
                x0 = torch.stack([x0[b, :, i:i + P, j:j + P] for b, i, j in zip(bb, ii, jj)])
            for _ in range(args.n_critic):
                with torch.no_grad():
                    fake = G(x1)
                lossD = D(fake).mean() - D(x0).mean() + args.lambda_gp * grad_penalty(x0, fake)
                optD.zero_grad(); lossD.backward(); optD.step()
            fake = G(x1)
            lossV = F.mse_loss(V(fake), V(x0))
            lossG = -D(fake).mean() + args.lambda_vgg * lossV
            optG.zero_grad(); lossG.backward(); optG.step()
            lg += lossG.item(); ld += lossD.item(); lv += lossV.item(); n += 1; it += 1
            if args.max_iters and it >= args.max_iters:
                break
        G.eval(); vm, m = 0.0, 0
        with torch.no_grad():
            for batch in vl:
                vm += F.mse_loss(G(batch['x_t'].to(device)), batch['x0'].to(device)).item(); m += 1
        vm /= max(m, 1)
        history.append({'epoch': epoch + 1, 'lossG': lg / max(n, 1), 'lossD': ld / max(n, 1), 'vgg': lv / max(n, 1), 'val_mse': vm})
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs}: G={lg/max(n,1):.4f} D={ld/max(n,1):.4f} vgg={lv/max(n,1):.4f} val={vm:.6f} (it {it})", flush=True)
        if args.max_iters and it >= args.max_iters and epoch != args.epochs - 1:
            args.epochs = epoch + 1
        if vm < best_val:
            best_val = vm
            torch.save({'model': G.state_dict(), 'critic': D.state_dict(), 'epoch': epoch, 'val_loss': vm,
                        'arch': 'wganvgg', 'mode': 'endpoint', 'experiment': args.experiment, 'schedule': args.schedule},
                       os.path.join(out_dir, 'best.pth'))
    with open(os.path.join(out_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f)

    G.load_state_dict(torch.load(os.path.join(out_dir, 'best.pth'), map_location=device)['model']); G.eval()
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
                pred = np.clip(G(torch.from_numpy(x_t)[None, None].to(device)).cpu().numpy().squeeze(), 0, 1)
                k = os.path.basename(f).split('_step')[1].split('.')[0]
                e = per_step.setdefault(k, {'t': t, 'psnr': [], 'ssim': [], 'psnr_in': []})
                e['psnr'].append(compute_psnr(pred, x0)); e['ssim'].append(compute_ssim(pred, x0))
                e['psnr_in'].append(compute_psnr(x_t, x0))
    summary = {'name': name, 'best_val': best_val, 'epochs': args.epochs,
               'params_M': sum(p.numel() for p in G.parameters()) / 1e6}
    for k, e in sorted(per_step.items()):
        summary[f'step{k}'] = {'t': e['t'], 'n': len(e['psnr']), 'psnr_in': float(np.mean(e['psnr_in'])),
                               'psnr': float(np.mean(e['psnr'])), 'psnr_std': float(np.std(e['psnr'])),
                               'ssim': float(np.mean(e['ssim']))}
        print(f"TEST step{k} t={e['t']:.3f}: in {np.mean(e['psnr_in']):.2f} -> "
              f"{np.mean(e['psnr']):.2f}±{np.std(e['psnr']):.2f} dB, SSIM {np.mean(e['ssim']):.4f}")
    with open(os.path.join(out_dir, 'test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
