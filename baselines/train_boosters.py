"""Datasets for bridge training, blind (dose-free) wrappers and the interpolation control."""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_eval import UNetWithTime, compute_psnr, compute_ssim, get_time_steps
from baselines.train_baselines import REDCNN, split_slice_ids, OUT_ROOT

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BRIDGE_PRECOMPUTED = {
    ('2detect', 'uniform'): f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed",
    ('2detect', 'equal_improvement'): f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed_equal_improvement",
    ('ldct', 'uniform'): f"{DATA_ROOT}/LDCT_dose_bridge/precomputed",
    ('ldct', 'equal_improvement'): f"{DATA_ROOT}/LDCT_dose_bridge/precomputed_equal_improvement",
    ('2detect_realI0', 'geometric'): f"{DATA_ROOT}/2DeteCT_dose_bridge_realI0/precomputed_geometric",
    ('2detect_effI0', 'geometric'): f"{DATA_ROOT}/2DeteCT_dose_bridge_effI0/precomputed_geometric",
}
ENDPOINT_SRC = {
    '2detect': f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed_equal_improvement",
    'ldct': f"{DATA_ROOT}/LDCT_dose_bridge/precomputed_equal_improvement",
}
ALPHAS = {'2detect': 0.01, 'ldct': 0.1, '2detect_realI0': 1500.0 / 53000.0,
          '2detect_effI0': 425.0 / 15000.0}


class BridgeAllStepDataset(Dataset):
    """(x_t, t, x0) for all 5 precomputed steps; split identical to all other models."""
    def __init__(self, pre, split, experiment):
        all_files = sorted(glob(os.path.join(pre, '*_step*.npz')))
        if not all_files:
            raise RuntimeError(f"No data at {pre}")
        by_sid = {}
        for f in all_files:
            by_sid.setdefault(os.path.basename(f).split('_step')[0], []).append(f)
        slice_ids = sorted(by_sid.keys())
        exp_key = 'ldct' if experiment.startswith('ldct') else '2detect'
        selected = split_slice_ids(slice_ids, exp_key, split)
        self.files = []
        for sid in selected:
            self.files.extend(by_sid[sid])
        print(f"  {split}: {len(self.files)} samples ({len(selected)} slices)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return {
            'x_t': torch.from_numpy(d['x_t']).float().unsqueeze(0),
            'x0': torch.from_numpy(d['x0']).float().unsqueeze(0),
            't': torch.tensor(float(d['t']), dtype=torch.float32),
        }


class InterpDataset(Dataset):
    """Matched-MSE linear-interpolation states from (x0, x1) endpoint pairs.

    knots t_k = k/5 (uniform), weights w_k = sqrt((a^{-t_k}-1)/(a^{-1}-1)),
    so E||x_w - x0||^2 matches the bridge degradation MSE at every knot.
    """
    def __init__(self, src, split, experiment, alpha):
        files = sorted(glob(os.path.join(src, '*_step4.npz')))
        if not files:
            raise RuntimeError(f"No data at {src}")
        by_id = {os.path.basename(f).split('_step')[0]: f for f in files}
        slice_ids = sorted(by_id.keys())
        exp_key = 'ldct' if experiment.startswith('ldct') else '2detect'
        selected = split_slice_ids(slice_ids, exp_key, split)
        self.files = [by_id[s] for s in selected]
        self.t_knots = np.array([k / 5.0 for k in range(1, 6)], dtype=np.float32)
        self.w_knots = np.sqrt((alpha ** (-self.t_knots) - 1.0)
                               / (alpha ** (-1.0) - 1.0)).astype(np.float32)
        print(f"  {split}: {len(self.files)} slices x 5 levels; "
              f"w={np.array2string(self.w_knots, precision=3)}")

    def __len__(self):
        return len(self.files) * 5

    def __getitem__(self, idx):
        f = self.files[idx // 5]
        k = idx % 5
        d = np.load(f)
        x0 = d['x0'].astype(np.float32)
        x1 = d['x_t'].astype(np.float32)  # step4 == t=1 state
        w = self.w_knots[k]
        xw = np.clip(x0 + w * (x1 - x0), 0.0, 1.0)
        return {
            'x_t': torch.from_numpy(xw).unsqueeze(0),
            'x0': torch.from_numpy(x0).unsqueeze(0),
            't': torch.tensor(float(self.t_knots[k]), dtype=torch.float32),
        }


class REDCNNTime(nn.Module):
    """RED-CNN with zero-initialized scale-shift (FiLM) time conditioning on
    conv1/3/5 features: out <- out*(1+g(t)) + b(t) before the ReLU, with the
    final MLP layer zero-initialized so the network is EXACTLY the blind
    RED-CNN at initialization and conditioning can only be learned on top.
    ~37K extra params. The endpoint control is the identical net at t=1."""
    def __init__(self, ch=96, t_dim=64):
        super().__init__()
        k, p = 5, 2
        self.conv1 = nn.Conv2d(1, ch, k, padding=p)
        self.conv2 = nn.Conv2d(ch, ch, k, padding=p)
        self.conv3 = nn.Conv2d(ch, ch, k, padding=p)
        self.conv4 = nn.Conv2d(ch, ch, k, padding=p)
        self.conv5 = nn.Conv2d(ch, ch, k, padding=p)
        self.tconv1 = nn.ConvTranspose2d(ch, ch, k, padding=p)
        self.tconv2 = nn.ConvTranspose2d(ch, ch, k, padding=p)
        self.tconv3 = nn.ConvTranspose2d(ch, ch, k, padding=p)
        self.tconv4 = nn.ConvTranspose2d(ch, ch, k, padding=p)
        self.tconv5 = nn.ConvTranspose2d(ch, 1, k, padding=p)
        self.t_mlp = nn.Sequential(nn.Linear(1, t_dim), nn.SiLU(),
                                   nn.Linear(t_dim, 6 * ch))
        nn.init.zeros_(self.t_mlp[-1].weight)
        nn.init.zeros_(self.t_mlp[-1].bias)

    def forward(self, x, t):
        gb = self.t_mlp(t.view(-1, 1).float())
        g1, b1, g2, b2, g3, b3 = [c[:, :, None, None] for c in gb.chunk(6, dim=1)]
        r1 = x
        out = F.relu(self.conv1(x) * (1 + g1) + b1)
        out = F.relu(self.conv2(out))
        r2 = out
        out = F.relu(self.conv3(out) * (1 + g2) + b2)
        out = F.relu(self.conv4(out))
        r3 = out
        out = F.relu(self.conv5(out) * (1 + g3) + b3)
        out = self.tconv1(out)
        out = out + r3
        out = self.tconv2(F.relu(out))
        out = self.tconv3(F.relu(out))
        out = out + r2
        out = self.tconv4(F.relu(out))
        out = self.tconv5(F.relu(out))
        out = out + r1
        return F.relu(out)


class BlindWrap(nn.Module):
    """Give a time input signature to a blind model (ignored)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t=None):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['redcnn_bridge', 'unet_bridge', 'interp',
                             'redcnn_t_bridge', 'redcnn_t_endpoint'])
    ap.add_argument('--experiment', required=True,
                    choices=['2detect', 'ldct', '2detect_realI0', '2detect_effI0'])
    ap.add_argument('--schedule', default='uniform',
                    choices=['uniform', 'geometric', 'equal_improvement'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    alpha = ALPHAS[args.experiment]

    if args.mode == 'interp':
        src = ENDPOINT_SRC[args.experiment]
        train_ds = InterpDataset(src, 'train', args.experiment, alpha)
        val_ds = InterpDataset(src, 'val', args.experiment, alpha)
        name = f'{args.experiment}_interp_unet'
        model = UNetWithTime(base_ch=64, t_dim=128).to(device)
        timed = True
    elif args.mode in ('redcnn_t_bridge', 'redcnn_t_endpoint'):
        pre = BRIDGE_PRECOMPUTED[(args.experiment, args.schedule)]
        train_ds = BridgeAllStepDataset(pre, 'train', args.experiment)
        val_ds = BridgeAllStepDataset(pre, 'val', args.experiment)
        if args.mode == 'redcnn_t_endpoint':
            for ds in (train_ds, val_ds):
                ds.files = [f for f in ds.files if f.endswith('_step4.npz')]
                print(f"  endpoint-only filter: {len(ds.files)} pairs")
            name = f'{args.experiment}_redcnn_t_endpoint'
        else:
            name = f'{args.experiment}_redcnn_t_bridge_{args.schedule}'
        model = REDCNNTime().to(device)
        timed = True
    else:
        pre = BRIDGE_PRECOMPUTED[(args.experiment, args.schedule)]
        train_ds = BridgeAllStepDataset(pre, 'train', args.experiment)
        val_ds = BridgeAllStepDataset(pre, 'val', args.experiment)
        if args.mode == 'redcnn_bridge':
            name = f'{args.experiment}_redcnn_bridge_{args.schedule}'
            model = REDCNN().to(device)
            timed = False
        else:
            name = f'{args.experiment}_unet_bridge_{args.schedule}'
            model = UNetWithTime(base_ch=64, t_dim=128).to(device)
            timed = True
    if args.seed != 42:
        name += f'_seed{args.seed}'
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"Booster: {name}  (mode={args.mode}, schedule={args.schedule}, "
          f"seed={args.seed}, alpha={alpha:.4f})")
    print(f"  out: {out_dir}")
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("=" * 70)

    if args.smoke:
        train_ds.files = train_ds.files[:64]
        val_ds.files = val_ds.files[:16]
        args.epochs = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()

    best_val = float('inf')
    history = []
    for epoch in range(args.epochs):
        model.train()
        tr, n = 0.0, 0
        for batch in train_loader:
            x_t = batch['x_t'].to(device)
            x0 = batch['x0'].to(device)
            t = batch['t'].to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(x_t, t) if timed else model(x_t)
                loss = F.mse_loss(pred, x0) + 0.1 * F.l1_loss(pred, x0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            tr += loss.item(); n += 1
        scheduler.step()

        model.eval()
        vl, m = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred = (model(batch['x_t'].to(device), batch['t'].to(device))
                        if timed else model(batch['x_t'].to(device)))
                vl += F.mse_loss(pred, batch['x0'].to(device)).item(); m += 1
        vl /= max(m, 1)
        history.append({'epoch': epoch + 1, 'val_mse': vl})
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs}: train={tr/max(n,1):.6f} "
                  f"val={vl:.6f}", flush=True)
        if vl < best_val:
            best_val = vl
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': vl,
                        'mode': args.mode, 'experiment': args.experiment,
                        'schedule': args.schedule, 'seed': args.seed},
                       os.path.join(out_dir, 'best.pth'))

    with open(os.path.join(out_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f)

    model.load_state_dict(torch.load(os.path.join(out_dir, 'best.pth'),
                                     map_location=device)['model'])
    model.eval()
    if args.mode == 'interp':
        src = ENDPOINT_SRC[args.experiment]
        files = sorted(glob(os.path.join(src, '*_step4.npz')))
    else:
        pre = BRIDGE_PRECOMPUTED[(args.experiment, args.schedule)]
        files = sorted(glob(os.path.join(pre, '*_step4.npz')))
    by_id = {os.path.basename(f).split('_step')[0]: f for f in files}
    exp_key = 'ldct' if args.experiment.startswith('ldct') else '2detect'
    test_ids = split_slice_ids(sorted(by_id.keys()), exp_key, 'test')
    psnrs, ssims = [], []
    n_skipped = 0
    with torch.no_grad():
        for sid in test_ids:
            d = np.load(by_id[sid])
            x1, x0 = d['x_t'].astype(np.float32), d['x0'].astype(np.float32)
            if not (np.isfinite(x1).all() and np.isfinite(x0).all()):
                n_skipped += 1
                continue
            xt = torch.from_numpy(x1)[None, None].to(device)
            tt = torch.tensor([1.0], dtype=torch.float32).to(device)
            pred = model(xt, tt) if timed else model(xt)
            pred = np.clip(pred.cpu().numpy().squeeze(), 0, 1)
            psnrs.append(compute_psnr(pred, x0))
            ssims.append(compute_ssim(pred, x0))
    summary = {'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
               'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
               'n': len(psnrs), 'n_skipped_nonfinite': n_skipped, 'best_val': best_val}
    print(f"\nTEST single-step ({name}): {summary['psnr_mean']:.2f}"
          f"±{summary['psnr_std']:.2f} dB, SSIM={summary['ssim_mean']:.4f}")
    with open(os.path.join(out_dir, 'test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
