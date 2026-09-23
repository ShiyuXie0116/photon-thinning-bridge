"""Shared network definitions and image-quality metrics."""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from scipy.ndimage import uniform_filter
import warnings

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')


# Network (same architecture used by all models)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.t_proj = nn.Linear(t_dim, out_ch) if t_dim else None
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb=None):
        h = F.silu(self.norm1(self.conv1(x)))
        if t_emb is not None and self.t_proj is not None:
            h = h + self.t_proj(t_emb)[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)

class UNetWithTime(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64, t_dim=128):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(), nn.Linear(t_dim * 4, t_dim))
        self.enc1 = ConvBlock(in_ch, base_ch, t_dim)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, t_dim)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, t_dim)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, t_dim)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8, t_dim)
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8, t_dim)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4, t_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, t_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, t_dim)
        self.out = nn.Conv2d(base_ch, out_ch, 1)
    def forward(self, x, t):
        t_emb = self.t_embed(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        e4 = self.enc4(self.pool(e3), t_emb)
        b = self.bottleneck(self.pool(e4), t_emb)
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1), t_emb)
        return self.out(d1)


def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    data_range = max(gt.max() - gt.min(), 1e-10)
    return 10.0 * np.log10(data_range ** 2 / mse)

def compute_ssim(pred, gt, win_size=7):
    data_range = max(gt.max() - gt.min(), 1e-10)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = uniform_filter(pred, size=win_size, mode='reflect')
    mu_y = uniform_filter(gt, size=win_size, mode='reflect')
    s_x = np.maximum(uniform_filter(pred ** 2, win_size, mode='reflect') - mu_x ** 2, 0)
    s_y = np.maximum(uniform_filter(gt ** 2, win_size, mode='reflect') - mu_y ** 2, 0)
    s_xy = uniform_filter(pred * gt, win_size, mode='reflect') - mu_x * mu_y
    return float(np.mean(((2*mu_x*mu_y+C1)*(2*s_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(s_x+s_y+C2))))


# Schedule computation

def get_time_steps(alpha, n_steps, schedule):
    N = n_steps
    if schedule == 'geometric':
        u_steps = np.linspace(1.0, alpha, N + 1)
        u_steps = np.clip(u_steps, 1e-15, None)
        t_steps = np.log(u_steps) / np.log(alpha)
    elif schedule == 'equal_improvement':
        s = np.zeros(N + 1)
        for k in range(N + 1):
            u_k = 1.0 + (N - k) / N * (1.0 / alpha - 1.0)
            s[k] = 0.0 if k == N else -np.log(u_k) / np.log(alpha)
        t_steps = s[::-1]
    else:
        t_steps = np.linspace(0.0, 1.0, N + 1)
    return t_steps.astype(np.float32)


def run_multistep(model, x_lowdose, schedule_times, pred_target, device):
    """Run multi-step inference from t=1 to t=0.

    schedule_times: ascending [0, ..., 1], reversed for inference.
    """
    t_reversed = schedule_times[::-1]
    x_current = x_lowdose.copy()

    with torch.no_grad():
        for i in range(len(t_reversed) - 1):
            t_curr = float(t_reversed[i])
            t_next = float(t_reversed[i + 1])

            x_t = torch.from_numpy(x_current).float().unsqueeze(0).unsqueeze(0).to(device)
            t_t = torch.tensor([t_curr], dtype=torch.float32).to(device)
            out = model(x_t, t_t).cpu().numpy().squeeze()

            if pred_target == 'x0':
                x0_pred = np.clip(out, 0.0, 1.0)
                if t_next < 1e-6:
                    x_current = x0_pred
                else:
                    ratio = t_next / t_curr
                    x_current = x0_pred + ratio * (x_current - x0_pred)
            else:
                dt = t_curr - t_next
                x_current = x_current + out * dt

            x_current = np.clip(x_current, 0.0, 1.0)

    return x_current


def run_singlestep(model, x_lowdose, device):
    """Single-step x0 prediction: model(x_lowdose, t=1.0) -> x0_pred."""
    with torch.no_grad():
        x_t = torch.from_numpy(x_lowdose).float().unsqueeze(0).unsqueeze(0).to(device)
        t_t = torch.tensor([1.0], dtype=torch.float32).to(device)
        x0_pred = model(x_t, t_t).cpu().numpy().squeeze()
    return np.clip(x0_pred, 0.0, 1.0)


def get_test_slice_ids(dataset, precomputed_path):
    """Get test slice IDs ensuring no overlap with train/val."""
    all_files = sorted(glob(os.path.join(precomputed_path, '*_step4.npz')))
    slice_ids = sorted(set(os.path.basename(f).split('_step')[0] for f in all_files))

    if dataset == 'ldct':
        test_pids = ['C004', 'C120']
        test_ids = [s for s in slice_ids if s.split('_s')[0] in test_pids]
        train_val_ids = [s for s in slice_ids if s.split('_s')[0] not in test_pids]
    else:
        np.random.seed(42)
        perm = np.random.permutation(len(slice_ids))
        n_train = int(len(slice_ids) * 0.8)
        n_val = int(len(slice_ids) * 0.1)
        test_ids = [slice_ids[i] for i in perm[n_train + n_val:]]
        train_val_ids = [slice_ids[i] for i in perm[:n_train + n_val]]

    assert len(set(test_ids) & set(train_val_ids)) == 0, "Test/train overlap detected!"
    return test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['2detect', 'ldct'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.dataset == '2detect':
        base = f"{DATA_ROOT}/2DeteCT_dose_bridge"
        alpha = 0.01
        ref_precomputed = os.path.join(base, 'precomputed_geometric')

        models_config = [
            ('vel_uniform',  os.path.join(base, 'output/dose_bridge_best.pth'),                       'velocity', 'uniform'),
            ('vel_geo',      os.path.join(base, 'output_geometric/dose_bridge_best.pth'),              'velocity', 'geometric'),
            ('vel_EI',       os.path.join(base, 'output_equal_improvement/dose_bridge_best.pth'),      'velocity', 'equal_improvement'),
            ('x0_uniform',   os.path.join(base, 'output_x0pred/x0pred_best.pth'),                     'x0',       'uniform'),
            ('x0_geo',       os.path.join(base, 'output_geometric_x0pred/dose_bridge_best.pth'),       'x0',       'geometric'),
            ('x0_EI',        os.path.join(base, 'output_equal_improvement_x0pred/dose_bridge_best.pth'),'x0',       'equal_improvement'),
        ]
    else:
        base = f"{DATA_ROOT}/LDCT_dose_bridge"
        alpha = 0.1
        ref_precomputed = os.path.join(base, 'precomputed_geometric')

        models_config = [
            ('vel_uniform',  os.path.join(base, 'output/ldct_dose_bridge_best.pth'),                       'velocity', 'uniform'),
            ('vel_geo',      os.path.join(base, 'output_geometric/ldct_dose_bridge_best.pth'),              'velocity', 'geometric'),
            ('vel_EI',       os.path.join(base, 'output_equal_improvement/ldct_dose_bridge_best.pth'),      'velocity', 'equal_improvement'),
            ('x0_geo',       os.path.join(base, 'output_geometric_x0pred/ldct_dose_bridge_best.pth'),       'x0',       'geometric'),
            ('x0_EI',        os.path.join(base, 'output_equal_improvement_x0pred/ldct_dose_bridge_best.pth'),'x0',       'equal_improvement'),
        ]

    n_steps = 5

    test_ids = get_test_slice_ids(args.dataset, ref_precomputed)
    print(f"\nDataset: {args.dataset}, alpha={alpha}")
    print(f"Test slices: {len(test_ids)}")
    print(f"Reference precomputed: {ref_precomputed}")

    print("\nLoading test data...")
    test_data = {}
    for sid in tqdm(test_ids, desc="Loading"):
        fpath = os.path.join(ref_precomputed, f"{sid}_step4.npz")
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath)
        test_data[sid] = {
            'x_lowdose': d['x_t'].astype(np.float32),
            'x0': d['x0'].astype(np.float32),
        }
    print(f"Loaded {len(test_data)} test samples")

    ld_psnrs, ld_ssims = [], []
    for sid, td in test_data.items():
        ld_psnrs.append(compute_psnr(td['x_lowdose'], td['x0']))
        ld_ssims.append(compute_ssim(td['x_lowdose'], td['x0']))

    all_results = {
        'low_dose': {
            'psnr_mean': float(np.mean(ld_psnrs)),
            'psnr_std': float(np.std(ld_psnrs)),
            'ssim_mean': float(np.mean(ld_ssims)),
            'ssim_std': float(np.std(ld_ssims)),
            'n_samples': len(ld_psnrs),
        }
    }
    print(f"\nLow-dose baseline: PSNR={np.mean(ld_psnrs):.2f}±{np.std(ld_psnrs):.2f}  "
          f"SSIM={np.mean(ld_ssims):.4f}±{np.std(ld_ssims):.4f}")

    for name, ckpt_path, pred_target, schedule in models_config:
        print(f"\n{'='*70}")
        print(f"Evaluating: {name} (target={pred_target}, schedule={schedule})")
        print(f"  Checkpoint: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            print(f"  SKIPPED: checkpoint not found")
            continue

        model = UNetWithTime(base_ch=64, t_dim=128).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        epoch = ckpt.get('epoch', '?')
        print(f"  Loaded (epoch {epoch})")

        times = get_time_steps(alpha, n_steps, schedule)
        print(f"  Schedule: {np.array2string(times, precision=4)}")

        psnrs_multi, ssims_multi = [], []
        psnrs_single, ssims_single = [], []

        for sid in tqdm(test_data, desc=f"  {name}"):
            td = test_data[sid]
            x_ld = td['x_lowdose']
            x0_gt = td['x0']

            x_result = run_multistep(model, x_ld, times, pred_target, device)
            psnrs_multi.append(compute_psnr(x_result, x0_gt))
            ssims_multi.append(compute_ssim(x_result, x0_gt))

            if pred_target == 'x0':
                x_single = run_singlestep(model, x_ld, device)
                psnrs_single.append(compute_psnr(x_single, x0_gt))
                ssims_single.append(compute_ssim(x_single, x0_gt))

        result = {
            'multi_step': {
                'psnr_mean': float(np.mean(psnrs_multi)),
                'psnr_std': float(np.std(psnrs_multi)),
                'ssim_mean': float(np.mean(ssims_multi)),
                'ssim_std': float(np.std(ssims_multi)),
            },
            'pred_target': pred_target,
            'schedule': schedule,
            'epoch': epoch,
            'n_samples': len(psnrs_multi),
        }

        print(f"  Multi-step:  PSNR={np.mean(psnrs_multi):.2f}±{np.std(psnrs_multi):.2f}  "
              f"SSIM={np.mean(ssims_multi):.4f}±{np.std(ssims_multi):.4f}")

        if psnrs_single:
            result['single_step'] = {
                'psnr_mean': float(np.mean(psnrs_single)),
                'psnr_std': float(np.std(psnrs_single)),
                'ssim_mean': float(np.mean(ssims_single)),
                'ssim_std': float(np.std(ssims_single)),
            }
            print(f"  Single-step: PSNR={np.mean(psnrs_single):.2f}±{np.std(psnrs_single):.2f}  "
                  f"SSIM={np.mean(ssims_single):.4f}±{np.std(ssims_single):.4f}")

        all_results[name] = result
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*90}")
    print(f"UNIFIED RESULTS — {args.dataset.upper()} (alpha={alpha}, {len(test_data)} test samples)")
    print(f"{'='*90}")
    print(f"{'Method':<25s} {'Schedule':<18s} {'Target':<10s} {'PSNR (dB)':<18s} {'SSIM':<18s}")
    print(f"{'-'*90}")
    r = all_results['low_dose']
    print(f"{'Low-dose input':<25s} {'—':<18s} {'—':<10s} "
          f"{r['psnr_mean']:.2f}±{r['psnr_std']:.2f}{'':>4s} "
          f"{r['ssim_mean']:.4f}±{r['ssim_std']:.4f}")

    for name in [k for k in all_results if k != 'low_dose']:
        r = all_results[name]
        m = r['multi_step']
        print(f"{name+' (multi)':<25s} {r['schedule']:<18s} {r['pred_target']:<10s} "
              f"{m['psnr_mean']:.2f}±{m['psnr_std']:.2f}{'':>4s} "
              f"{m['ssim_mean']:.4f}±{m['ssim_std']:.4f}")
        if 'single_step' in r:
            s = r['single_step']
            print(f"{name+' (single)':<25s} {r['schedule']:<18s} {r['pred_target']:<10s} "
                  f"{s['psnr_mean']:.2f}±{s['psnr_std']:.2f}{'':>4s} "
                  f"{s['ssim_mean']:.4f}±{s['ssim_std']:.4f}")
    print(f"{'='*90}")

    out_path = os.path.join(base, f'unified_eval_{args.dataset}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
