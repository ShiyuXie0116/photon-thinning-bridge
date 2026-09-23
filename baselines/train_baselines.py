"""Single-dose baseline architectures and their training loop."""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_eval import UNetWithTime, compute_psnr, compute_ssim

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRECOMPUTED = {
    '2detect': f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed_equal_improvement",
    'ldct': f"{DATA_ROOT}/LDCT_dose_bridge/precomputed_equal_improvement",
    '2detect_realI0': f"{DATA_ROOT}/2DeteCT_dose_bridge_realI0/precomputed_geometric",
    '2detect_effI0': f"{DATA_ROOT}/2DeteCT_dose_bridge_effI0/precomputed_geometric",
}
OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"
LDCT_TEST_PATIENTS = ['C004', 'C120']


class REDCNN(nn.Module):
    def __init__(self, ch=96):
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

    def forward(self, x):
        r1 = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        r2 = out
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        r3 = out
        out = F.relu(self.conv5(out))
        out = self.tconv1(out)
        out = out + r3
        out = self.tconv2(F.relu(out))
        out = self.tconv3(F.relu(out))
        out = out + r2
        out = self.tconv4(F.relu(out))
        out = self.tconv5(F.relu(out))
        out = out + r1
        return F.relu(out)


class UNetEndpoint(nn.Module):
    """Identical UNetWithTime, always queried at t=1.0."""
    def __init__(self):
        super().__init__()
        self.net = UNetWithTime(base_ch=64, t_dim=128)

    def forward(self, x):
        t = torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
        return self.net(x, t)


def split_slice_ids(slice_ids, experiment, split, seed=42):
    if experiment == 'ldct':
        test_pids = set(LDCT_TEST_PATIENTS)
        train_val = [s for s in slice_ids if s.split('_s')[0] not in test_pids]
        test = [s for s in slice_ids if s.split('_s')[0] in test_pids]
        if split == 'test':
            return test
        np.random.seed(seed)
        perm2 = np.random.permutation(len(train_val))
        n_tr = int(len(train_val) * 0.85)
        idx = perm2[:n_tr] if split == 'train' else perm2[n_tr:]
        return [train_val[i] for i in idx]
    else:  # 2detect / 2detect_realI0: slice-level 80/10/10
        np.random.seed(seed)
        perm = np.random.permutation(len(slice_ids))
        n_train = int(len(slice_ids) * 0.8)
        n_val = int(len(slice_ids) * 0.1)
        if split == 'train':
            idx = perm[:n_train]
        elif split == 'val':
            idx = perm[n_train:n_train + n_val]
        else:
            idx = perm[n_train + n_val:]
        return [slice_ids[i] for i in idx]


class EndpointDataset(Dataset):
    def __init__(self, precomputed_path, split, experiment):
        files = sorted(glob(os.path.join(precomputed_path, '*_step4.npz')))
        if not files:
            raise RuntimeError(f"No step4 data at {precomputed_path}")
        by_id = {os.path.basename(f).split('_step')[0]: f for f in files}
        slice_ids = sorted(by_id.keys())
        selected = split_slice_ids(slice_ids, experiment, split)
        self.files = [by_id[s] for s in selected]
        print(f"  {split}: {len(self.files)} endpoint pairs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return {
            'x1': torch.from_numpy(d['x_t']).float().unsqueeze(0),   # step4 x_t == t=1.0
            'x0': torch.from_numpy(d['x0']).float().unsqueeze(0),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True, choices=list(PRECOMPUTED.keys()))
    ap.add_argument('--model', required=True, choices=['unet', 'redcnn'])
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--smoke', action='store_true', help='2 epochs, tiny subset')
    ap.add_argument('--seed', type=int, default=42,
                    help='Training seed (init + data order); the train/val/test '
                         'split always uses seed 42 so splits stay identical')
    args = ap.parse_args()

    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pre = PRECOMPUTED[args.experiment]
    suffix = '' if args.seed == 42 else f'_seed{args.seed}'
    out_dir = os.path.join(OUT_ROOT, f'{args.experiment}_{args.model}_endpoint{suffix}')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"Endpoint-only baseline: {args.model} on {args.experiment}")
    print(f"  data: {pre}\n  out:  {out_dir}\n  epochs: {args.epochs}")
    print("=" * 70)

    model = (UNetEndpoint() if args.model == 'unet' else REDCNN()).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_ds = EndpointDataset(pre, 'train', args.experiment)
    val_ds = EndpointDataset(pre, 'val', args.experiment)
    if args.smoke:
        train_ds.files = train_ds.files[:128]
        val_ds.files = val_ds.files[:32]
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
        tr_loss, n = 0.0, 0
        for batch in train_loader:
            x1 = batch['x1'].to(device)
            x0 = batch['x0'].to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(x1)
                loss = F.mse_loss(pred, x0) + 0.1 * F.l1_loss(pred, x0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item(); n += 1
        scheduler.step()

        model.eval()
        vl, m = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['x1'].to(device))
                vl += F.mse_loss(pred, batch['x0'].to(device)).item(); m += 1
        vl /= max(m, 1)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}/{args.epochs}: train={tr_loss/max(n,1):.6f} "
                  f"val={vl:.6f} lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        history.append({'epoch': epoch + 1, 'val_mse': vl})

        if vl < best_val:
            best_val = vl
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': vl,
                        'arch': args.model, 'experiment': args.experiment},
                       os.path.join(out_dir, 'best.pth'))

    with open(os.path.join(out_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f)

    model.load_state_dict(torch.load(os.path.join(out_dir, 'best.pth'),
                                     map_location=device)['model'])
    model.eval()
    test_ds = EndpointDataset(pre, 'test', args.experiment)
    psnrs, ssims, ld_psnrs = [], [], []
    with torch.no_grad():
        for f in test_ds.files:
            d = np.load(f)
            x1, x0 = d['x_t'].astype(np.float32), d['x0'].astype(np.float32)
            pred = model(torch.from_numpy(x1)[None, None].to(device))
            pred = np.clip(pred.cpu().numpy().squeeze(), 0, 1)
            psnrs.append(compute_psnr(pred, x0))
            ssims.append(compute_ssim(pred, x0))
            ld_psnrs.append(compute_psnr(x1, x0))
    summary = {
        'low_dose_psnr': float(np.mean(ld_psnrs)),
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'n': len(psnrs), 'best_epoch': int(torch.load(
            os.path.join(out_dir, 'best.pth'), map_location='cpu')['epoch']),
    }
    print(f"\nTEST ({args.experiment}/{args.model}): "
          f"LD={summary['low_dose_psnr']:.2f} -> {summary['psnr_mean']:.2f}"
          f"±{summary['psnr_std']:.2f} dB, SSIM={summary['ssim_mean']:.4f}")
    with open(os.path.join(out_dir, 'test_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
