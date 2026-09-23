"""Photon-thinning bridge on Mayo LDCT: trajectory precomputation and training."""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from glob import glob
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import traceback
import warnings

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

try:
    import astra
    HAS_ASTRA = True
except ImportError:
    HAS_ASTRA = False
    print("WARNING: astra-toolbox not found.")


# Configuration

class Config:
    recon_dir = f"{DATA_ROOT}/LDCT_recon"
    precomputed_path = f"{DATA_ROOT}/LDCT_dose_bridge/precomputed"
    output_path = f"{DATA_ROOT}/LDCT_dose_bridge/output"

    DSO = 595.0              # source-to-origin distance (mm)
    DDO = 490.6
    DU = 1.285839319229126
    n_angles = 1152
    n_det = 736
    VOXEL_SIZE_512 = 0.7421875
    HU_FACTOR = 0.0192

    recon_size = 256
    edge_margin = 30

    I0_high = 1e5
    I0_low = 1e4
    alpha = None
    n_steps = 5
    time_schedule = 'geometric'

    patients_test = ['C004', 'C120']
    patients_train = None

    base_channels = 64
    t_dim = 128

    batch_size = 64
    epochs = 80
    lr = 1e-4
    weight_decay = 1e-4
    train_ratio = 0.85
    val_ratio = 0.15
    num_workers = 4
    use_amp = True
    seed = 42

    def compute_alpha(self):
        self.alpha = self.I0_low / self.I0_high


    def detect_patients(self):
        """Auto-detect all patients with full-dose sinograms."""
        all_pids = []
        for d in sorted(os.listdir(self.recon_dir)):
            if '_full_complete' in d:
                tif = os.path.join(self.recon_dir, d, 'scan_001_flat_fan_projections.tif')
                if os.path.exists(tif):
                    pid = d.replace('_full_complete', '')
                    all_pids.append(pid)
        self.patients_train = [p for p in all_pids if p not in self.patients_test]
        return all_pids


def get_time_steps(config):
    """Return time steps from t=0 (full-dose) to t=1 (low-dose).

    uniform:            uniform in t-space.
    geometric:          uniform in dose-space u=alpha^t.
    equal_improvement:  uniform in MSE-space u=alpha^{-t}, derived from
                        E[||x^(t)-x^(0)||^2] = K*(alpha^{-t}-1).
                        Small dt near t=1 (low-dose), large dt near t=0.
    """
    N = config.n_steps
    alpha = config.alpha
    if config.time_schedule == 'geometric':
        u_steps = np.linspace(1.0, alpha, N + 1)
        u_steps = np.clip(u_steps, 1e-15, None)
        t_steps = np.log(u_steps) / np.log(alpha)
    elif config.time_schedule == 'equal_improvement':
        s = np.zeros(N + 1)
        for k in range(N + 1):
            u_k = 1.0 + (N - k) / N * (1.0 / alpha - 1.0)
            s[k] = 0.0 if k == N else -np.log(u_k) / np.log(alpha)
        t_steps = s[::-1]
    else:
        t_steps = np.linspace(0.0, 1.0, N + 1)
    return t_steps.astype(np.float32)


def load_patient_angles(patient_id, config):
    """Load real angles from rebinning_params.npz for a patient."""
    npz_path = os.path.join(config.recon_dir, f'{patient_id}_full_complete', 'rebinning_params.npz')
    npz = np.load(npz_path, allow_pickle=True)
    angles = npz['angles'].astype(np.float64)[:config.n_angles] + np.pi / 2
    return angles


def correct_fbp(sinogram, img_size, angles, config):
    """FBP reconstruction matching existing pipeline.
    - FOV-adjusted voxel_size for any resolution
    - Siemens detector flip
    - Sinogram scaling by 1/voxel_size
    - Hann filter
    """
    voxel_size = config.VOXEL_SIZE_512 * (512.0 / img_size)
    vox_scaling = 1.0 / voxel_size
    source_dist = vox_scaling * config.DSO
    det_dist = vox_scaling * config.DDO
    det_spacing = vox_scaling * config.DU

    vol_geom = astra.create_vol_geom(img_size, img_size)
    proj_geom = astra.create_proj_geom('fanflat', det_spacing, config.n_det, angles,
                                        source_dist, det_dist)

    sino_flipped = np.flip(sinogram.copy(), axis=1)
    sino_scaled = sino_flipped * vox_scaling

    sino_id = astra.data2d.create('-sino', proj_geom, sino_scaled.astype(np.float32))
    rec_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id
    cfg['FilterType'] = 'hann'

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recon = astra.data2d.get(rec_id).copy()

    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(rec_id)
    return recon


def to_hu(recon, config):
    """Convert attenuation coefficients to HU."""
    return 1000.0 * (recon - config.HU_FACTOR) / config.HU_FACTOR


def load_patient_sinogram(patient_id, dose, config):
    """Load the full 3D fan-beam sinogram for a patient.
    Returns: (n_angles, n_det, n_slices) array.
    """
    d = os.path.join(config.recon_dir, f'{patient_id}_{dose}_complete')
    tif_path = os.path.join(d, 'scan_001_flat_fan_projections.tif')
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"No sinogram at {tif_path}")

    img = Image.open(tif_path)
    n_frames = img.n_frames
    h, w = img.size

    sino_3d = np.zeros((n_frames, w, h), dtype=np.float32)
    for i in range(n_frames):
        img.seek(i)
        frame = np.array(img, dtype=np.float32)
        sino_3d[i] = frame
    img.close()

    return sino_3d


def extract_slice_sinogram(sino_3d, slice_idx):
    """Extract 2D sinogram for one slice: (n_angles, n_det)."""
    return np.ascontiguousarray(sino_3d[:, :, slice_idx].astype(np.float32))


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
    s_x = np.maximum(uniform_filter(pred ** 2, size=win_size, mode='reflect') - mu_x ** 2, 0)
    s_y = np.maximum(uniform_filter(gt ** 2, size=win_size, mode='reflect') - mu_y ** 2, 0)
    s_xy = uniform_filter(pred * gt, size=win_size, mode='reflect') - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * s_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (s_x + s_y + C2))
    return float(np.mean(ssim_map))


def compute_rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def precompute_patient(patient_id, config):
    """Precompute dose bridge for all valid slices of a patient."""
    print(f"\nLoading sinogram for {patient_id}...")
    sino_3d = load_patient_sinogram(patient_id, 'full', config)
    n_angles, n_det, n_slices = sino_3d.shape
    print(f"  Shape: {sino_3d.shape} (angles={n_angles}, det={n_det}, slices={n_slices})")

    angles = load_patient_angles(patient_id, config)
    print(f"  Angles: n={len(angles)}, range=[{np.degrees(angles.min()):.1f}°, {np.degrees(angles.max()):.1f}°]")

    margin = config.edge_margin
    valid_start = margin
    valid_end = n_slices - margin

    alpha = config.alpha
    I0_high = config.I0_high
    n_steps = config.n_steps
    recon_size = config.recon_size
    times = get_time_steps(config)

    total = 0
    for s in tqdm(range(valid_start, valid_end), desc=f"  {patient_id}"):
        check = os.path.join(config.precomputed_path, f"{patient_id}_s{s:04d}_step0.npz")
        if os.path.exists(check):
            total += n_steps
            continue

        try:
            y_full = extract_slice_sinogram(sino_3d, s)

            y_full = np.maximum(y_full, 0.0)

            q_full = np.round(I0_high * np.exp(-y_full)).astype(np.int64)
            q_full = np.maximum(q_full, 1)

            sinograms = {0.0: y_full}
            for i in range(1, n_steps + 1):
                t = times[i]
                alpha_t = alpha ** t
                I0_t = I0_high * alpha_t
                q_t = np.random.binomial(q_full, alpha_t)
                q_t = np.maximum(q_t, 1)
                y_t = -np.log(q_t.astype(np.float64) / I0_t).astype(np.float32)
                sinograms[t] = y_t

            images_hu = {}
            for t in times:
                recon = correct_fbp(sinograms[t], recon_size, angles, config)
                images_hu[t] = to_hu(recon, config)

            x0 = images_hu[0.0]
            vmin = float(x0.min())
            vmax = float(x0.max())
            scale = vmax - vmin + 1e-8

            def normalize(x):
                return np.clip((x - vmin) / scale, 0.0, 1.0).astype(np.float32)

            images_n = {t: normalize(images_hu[t]) for t in times}

            for step in range(n_steps):
                t_curr = times[step + 1]
                t_prev = times[step]
                dt = t_curr - t_prev
                x_curr = images_n[t_curr]
                x_prev = images_n[t_prev]
                velocity = (x_prev - x_curr) / (dt + 1e-8)

                sample_id = f"{patient_id}_s{s:04d}_step{step}"
                save_path = os.path.join(config.precomputed_path, f"{sample_id}.npz")
                np.savez_compressed(
                    save_path,
                    x_t=x_curr, x_t_next=x_prev,
                    velocity=velocity,
                    x0=images_n[0.0], x1=images_n[1.0],
                    t=np.float32(t_curr), t_next=np.float32(t_prev),
                    dt=np.float32(dt),
                    patient_id=patient_id, slice_idx=s,
                    vmin=np.float32(vmin), vmax=np.float32(vmax),
                )
                total += 1

        except Exception as e:
            print(f"  Error {patient_id} slice {s}: {e}")
            traceback.print_exc()

    print(f"  {patient_id}: {total} samples")
    return total


def precompute_all(config, patient_list):
    assert HAS_ASTRA
    os.makedirs(config.precomputed_path, exist_ok=True)

    print(f"Precomputing LDCT dose bridge (FBP + HU)")
    print(f"  Patients: {patient_list}")
    print(f"  I0_high: {config.I0_high:.0f}, I0_low: {config.I0_low:.0f}, alpha: {config.alpha:.4f}")
    print(f"  Steps: {config.n_steps}, Recon: {config.recon_size}")
    print(f"  Output: {config.precomputed_path}")

    grand_total = 0
    for pid in patient_list:
        grand_total += precompute_patient(pid, config)

    print(f"\nDone. Total samples: {grand_total}")


class DoseBridgeDataset(Dataset):
    def __init__(self, config, split='train'):
        self.pred_target = getattr(config, 'pred_target', 'velocity')
        all_files = sorted(glob(os.path.join(config.precomputed_path, '*.npz')))
        if len(all_files) == 0:
            raise RuntimeError(f"No precomputed data at {config.precomputed_path}")

        slice_files = {}
        for f in all_files:
            sid = os.path.basename(f).split('_step')[0]
            slice_files.setdefault(sid, []).append(f)

        slice_ids = sorted(slice_files.keys())

        test_pids = set(config.patients_test)
        train_val_ids = [s for s in slice_ids if s.split('_s')[0] not in test_pids]
        test_ids = [s for s in slice_ids if s.split('_s')[0] in test_pids]

        if split == 'test':
            selected = test_ids
        else:
            np.random.seed(config.seed)
            perm = np.random.permutation(len(train_val_ids))
            n_train = int(len(train_val_ids) * config.train_ratio / (config.train_ratio + config.val_ratio))
            if split == 'train':
                selected = [train_val_ids[i] for i in perm[:n_train]]
            else:
                selected = [train_val_ids[i] for i in perm[n_train:]]

        self.files = []
        for sid in selected:
            self.files.extend(slice_files[sid])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x_t = data['x_t']
        t = float(data['t'])
        dt = float(data['dt'])
        target = data['x0'] if self.pred_target == 'x0' else data['velocity']
        return {
            'x_t': torch.from_numpy(x_t).float().unsqueeze(0),
            'target': torch.from_numpy(target).float().unsqueeze(0),
            't': torch.tensor([t], dtype=torch.float32),
            'dt': torch.tensor([dt], dtype=torch.float32),
        }


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


def train_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        x_t = batch['x_t'].to(device)
        target = batch['target'].to(device)
        t = batch['t'].squeeze(-1).to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                out = model(x_t, t)
                loss = F.mse_loss(out, target) + 0.1 * F.l1_loss(out, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x_t, t)
            loss = F.mse_loss(out, target) + 0.1 * F.l1_loss(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def validate(model, loader, device):
    model.eval()
    total_loss, n = 0, 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch['x_t'].to(device), batch['t'].squeeze(-1).to(device))
            target = batch['target'].to(device)
            total_loss += F.mse_loss(out, target).item()
            n += 1
    return total_loss / max(n, 1)


def inference_on_slice(model, sinogram_2d, angles, config, device):
    """Run dose bridge inference: simulate low-dose from sinogram, bridge back."""
    alpha = config.alpha
    I0_high = config.I0_high
    recon_size = config.recon_size

    y_full = np.maximum(sinogram_2d, 0.0)
    x_full_hu = to_hu(correct_fbp(y_full, recon_size, angles, config), config)

    q_full = np.round(I0_high * np.exp(-y_full)).astype(np.int64)
    q_full = np.maximum(q_full, 1)
    q_low = np.random.binomial(q_full, alpha)
    q_low = np.maximum(q_low, 1)
    I0_t = I0_high * alpha
    y_low = -np.log(q_low.astype(np.float64) / I0_t).astype(np.float32)
    x_low_hu = to_hu(correct_fbp(y_low, recon_size, angles, config), config)

    vmin = float(x_full_hu.min())
    vmax = float(x_full_hu.max())
    scale = vmax - vmin + 1e-8

    def normalize(x):
        return np.clip((x - vmin) / scale, 0.0, 1.0).astype(np.float32)

    x0 = normalize(x_full_hu)
    x1 = normalize(x_low_hu)

    model.eval()
    t_steps = get_time_steps(config)
    t_reversed = t_steps[::-1]
    x_current = x1.copy()
    trajectory = [(float(t_reversed[0]), x_current.copy())]
    pred_target = getattr(config, 'pred_target', 'velocity')

    with torch.no_grad():
        for i in range(len(t_reversed) - 1):
            t_curr = float(t_reversed[i])
            t_next = float(t_reversed[i + 1])

            x_t_tensor = torch.from_numpy(x_current).float().unsqueeze(0).unsqueeze(0).to(device)
            t_tensor = torch.tensor([t_curr], dtype=torch.float32).to(device)
            out = model(x_t_tensor, t_tensor).cpu().numpy().squeeze()

            if pred_target == 'x0':
                x0_pred = out
                if t_next < 1e-6:
                    x_current = x0_pred
                else:
                    ratio = t_next / t_curr
                    x_current = x0_pred + ratio * (x_current - x0_pred)
            else:
                dt = t_curr - t_next
                x_current = x_current + out * dt

            x_current = np.clip(x_current, 0.0, 1.0)
            trajectory.append((t_next, x_current.copy()))

    return x0, x1, x_current, trajectory


def quick_evaluate(model, config, device, epoch, n_samples=5):
    net = model.module if hasattr(model, 'module') else model

    metrics = {'low_dose': [], 'dose_bridge': []}
    rng = np.random.RandomState(epoch)

    for pid in config.patients_test[:1]:
        angles = load_patient_angles(pid, config)
        sino_3d = load_patient_sinogram(pid, 'full', config)
        n_slices = sino_3d.shape[2]
        margin = config.edge_margin
        valid = list(range(margin, n_slices - margin))
        chosen = rng.choice(valid, min(n_samples, len(valid)), replace=False)

        for s in chosen:
            try:
                sino_2d = extract_slice_sinogram(sino_3d, s)
                x0, x1, x_db, _ = inference_on_slice(net, sino_2d, angles, config, device)
                metrics['low_dose'].append((compute_psnr(x1, x0), compute_ssim(x1, x0)))
                metrics['dose_bridge'].append((compute_psnr(x_db, x0), compute_ssim(x_db, x0)))
            except Exception as e:
                print(f"  Eval error {pid} s{s}: {e}")

    if not metrics['dose_bridge']:
        return None

    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1} eval ({len(metrics['dose_bridge'])} samples)")
    for k in ['low_dose', 'dose_bridge']:
        p = np.mean([m[0] for m in metrics[k]])
        s = np.mean([m[1] for m in metrics[k]])
        print(f"  {k:15s}: PSNR={p:.2f}  SSIM={s:.4f}")
    db_p = np.mean([m[0] for m in metrics['dose_bridge']])
    ld_p = np.mean([m[0] for m in metrics['low_dose']])
    print(f"  DB vs LD: {db_p - ld_p:+.2f} dB")
    print(f"{'='*70}\n")
    return {'epoch': epoch + 1, 'low_dose_psnr': float(ld_p), 'dose_bridge_psnr': float(db_p)}


def full_evaluate(model, config, device, n_samples=50, n_vis=8):
    os.makedirs(config.output_path, exist_ok=True)
    rng = np.random.RandomState(config.seed)

    results = {m: {'psnr': [], 'ssim': [], 'rmse': []} for m in ['low_dose', 'dose_bridge']}
    vis_data = []

    for pid in config.patients_test:
        print(f"Evaluating {pid}...")
        angles = load_patient_angles(pid, config)
        sino_3d = load_patient_sinogram(pid, 'full', config)
        n_slices = sino_3d.shape[2]
        margin = config.edge_margin
        valid = list(range(margin, n_slices - margin))
        chosen = rng.choice(valid, min(n_samples, len(valid)), replace=False)

        for s in tqdm(chosen, desc=f"  {pid}"):
            try:
                sino_2d = extract_slice_sinogram(sino_3d, s)
                x0, x1, x_db, traj = inference_on_slice(model, sino_2d, angles, config, device)
                for m, x in [('low_dose', x1), ('dose_bridge', x_db)]:
                    results[m]['psnr'].append(compute_psnr(x, x0))
                    results[m]['ssim'].append(compute_ssim(x, x0))
                    results[m]['rmse'].append(compute_rmse(x, x0))
                if len(vis_data) < n_vis:
                    vis_data.append({'gt': x0, 'low_dose': x1, 'db': x_db,
                                     'slice': f"{pid}_s{s}", 'trajectory': traj})
            except Exception as e:
                print(f"  Skip {pid} s{s}: {e}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (LDCT Dose Bridge)")
    print("=" * 80)
    for m in ['low_dose', 'dose_bridge']:
        if results[m]['psnr']:
            p = np.mean(results[m]['psnr'])
            ps = np.std(results[m]['psnr'])
            ss = np.mean(results[m]['ssim'])
            print(f"  {m:20s}: PSNR={p:.2f}+/-{ps:.2f}  SSIM={ss:.4f}")
    print("=" * 80)

    if vis_data:
        nv = min(n_vis, len(vis_data))
        fig, axes = plt.subplots(nv, 4, figsize=(16, 4 * nv))
        if nv == 1:
            axes = axes.reshape(1, -1)
        for i in range(nv):
            d = vis_data[i]
            vm, vx = d['gt'].min(), d['gt'].max()
            p_ld = compute_psnr(d['low_dose'], d['gt'])
            p_db = compute_psnr(d['db'], d['gt'])
            axes[i, 0].imshow(d['gt'], cmap='gray', vmin=vm, vmax=vx)
            axes[i, 0].set_title(f"Full-dose ({d['slice']})")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(d['low_dose'], cmap='gray', vmin=vm, vmax=vx)
            axes[i, 1].set_title(f"Low-dose {p_ld:.1f}dB")
            axes[i, 1].axis('off')
            color = 'green' if p_db > p_ld else 'red'
            axes[i, 2].imshow(d['db'], cmap='gray', vmin=vm, vmax=vx)
            axes[i, 2].set_title(f"Dose Bridge {p_db:.1f}dB ({p_db-p_ld:+.1f})",
                                 color=color, fontweight='bold')
            axes[i, 2].axis('off')
            err = np.abs(d['db'] - d['gt'])
            axes[i, 3].imshow(err, cmap='hot', vmin=0, vmax=max(err.max() * 0.5, 0.01))
            axes[i, 3].set_title("Error (DB)")
            axes[i, 3].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(config.output_path, 'ldct_comparison.png'), dpi=150)
        plt.close()

    summary = {}
    for m in ['low_dose', 'dose_bridge']:
        if results[m]['psnr']:
            summary[m] = {
                'psnr_mean': float(np.mean(results[m]['psnr'])),
                'psnr_std': float(np.std(results[m]['psnr'])),
                'ssim_mean': float(np.mean(results[m]['ssim'])),
            }
    summary['config'] = {
        'recon_size': config.recon_size,
        'I0_high': config.I0_high,
        'I0_low': config.I0_low,
        'n_steps': config.n_steps,
        'patients_test': config.patients_test,
    }
    with open(os.path.join(config.output_path, 'ldct_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {config.output_path}")


def main():
    parser = argparse.ArgumentParser(description='LDCT Dose Bridge (Binomial Thinning)')
    parser.add_argument('--precompute', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--patient_group', type=int, default=None,
                        help='Split train patients into 4 groups (0-3) for parallel precompute')
    parser.add_argument('--patients', type=str, nargs='+', default=None,
                        help='Specific patients to precompute')
    parser.add_argument('--recon_size', type=int, default=256)
    parser.add_argument('--I0_high', type=float, default=1e5)
    parser.add_argument('--I0_low', type=float, default=1e4)
    parser.add_argument('--n_steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--schedule', type=str, default='geometric',
                        choices=['uniform', 'geometric', 'equal_improvement'],
                        help='Time step schedule (default: geometric)')
    parser.add_argument('--target', type=str, default='velocity',
                        choices=['velocity', 'x0'],
                        help='Prediction target: velocity or x0 (default: velocity)')
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Override base directory for precomputed and output paths')
    args = parser.parse_args()

    config = Config()
    config.recon_size = args.recon_size
    config.I0_high = args.I0_high
    config.I0_low = args.I0_low
    config.compute_alpha()
    config.n_steps = args.n_steps
    config.time_schedule = args.schedule
    config.pred_target = args.target
    config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    if args.base_dir:
        config.precomputed_path = os.path.join(args.base_dir, 'precomputed')
        config.output_path = os.path.join(args.base_dir, 'output')

    if config.time_schedule != 'uniform':
        config.precomputed_path = config.precomputed_path.rstrip('/') + f'_{config.time_schedule}'
        config.output_path = config.output_path.rstrip('/') + f'_{config.time_schedule}'

    if config.pred_target == 'x0':
        config.output_path = config.output_path.rstrip('/') + '_x0pred'

    all_pids = config.detect_patients()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    times = get_time_steps(config)
    print("=" * 70)
    print(f"LDCT Dose Bridge (Binomial Thinning) [{config.time_schedule} schedule]")
    print("=" * 70)
    print(f"  Recon: {config.recon_size}")
    print(f"  I0_high: {config.I0_high:.0f}, I0_low: {config.I0_low:.0f}, alpha: {config.alpha:.6f}")
    print(f"  Schedule: {config.time_schedule}")
    print(f"  Prediction target: {config.pred_target}")
    print(f"  Time grid: {np.array2string(times, precision=3)}")
    print(f"  dt per step: {np.array2string(np.diff(times), precision=3)}")
    print(f"  Dose levels: {[f'{config.I0_high * config.alpha**t:.0f}' for t in times]}")
    print(f"  Precomputed: {config.precomputed_path}")
    print(f"  Output: {config.output_path}")
    print(f"  Train patients ({len(config.patients_train)}): {config.patients_train}")
    print(f"  Test patients: {config.patients_test}")
    print("=" * 70)

    if args.precompute:
        if args.patients:
            patient_list = args.patients
        elif args.patient_group is not None:
            n = len(config.patients_train)
            groups = [config.patients_train[i::4] for i in range(4)]
            patient_list = groups[args.patient_group]
            print(f"Patient group {args.patient_group}: {patient_list}")
        else:
            patient_list = config.patients_train
        precompute_all(config, patient_list)
        return

    if args.eval:
        model = UNetWithTime(base_ch=config.base_channels, t_dim=config.t_dim).to(device)
        ckpt_path = args.ckpt or os.path.join(config.output_path, 'ldct_dose_bridge_best.pth')
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
            print(f"Loaded: {ckpt_path}")
        full_evaluate(model, config, device, n_samples=50, n_vis=8)
        return

    if args.train:
        os.makedirs(config.output_path, exist_ok=True)
        model = UNetWithTime(base_ch=config.base_channels, t_dim=config.t_dim).to(device)
        print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

        start_epoch = 0
        if args.ckpt and os.path.exists(args.ckpt):
            ckpt = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(ckpt['model'])
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resuming from epoch {start_epoch}")

        try:
            train_ds = DoseBridgeDataset(config, 'train')
            val_ds = DoseBridgeDataset(config, 'val')
        except RuntimeError as e:
            print(f"ERROR: {e}\nRun --precompute first!")
            return

        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
        scaler = GradScaler() if config.use_amp else None

        best_val_loss = float('inf')
        eval_history = []

        for _ in range(start_epoch):
            scheduler.step()

        print("\nStarting training...")
        for epoch in range(start_epoch, config.epochs):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, config.use_amp)
            val_loss = validate(model, val_loader, device)
            scheduler.step()

            print(f"Epoch {epoch+1}/{config.epochs}: train={train_loss:.6f} val={val_loss:.6f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss},
                           os.path.join(config.output_path, 'ldct_dose_bridge_best.pth'))
                print(f"  -> Best saved")

            if (epoch + 1) % 10 == 0:
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss},
                           os.path.join(config.output_path, f'ldct_dose_bridge_epoch{epoch+1}.pth'))

            if HAS_ASTRA and ((epoch + 1) % 10 == 0 or epoch == 0):
                res = quick_evaluate(model, config, device, epoch, n_samples=5)
                if res:
                    eval_history.append(res)
                    with open(os.path.join(config.output_path, 'eval_history.json'), 'w') as f:
                        json.dump(eval_history, f, indent=2)

        if HAS_ASTRA:
            print("\nFinal evaluation...")
            best_path = os.path.join(config.output_path, 'ldct_dose_bridge_best.pth')
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device)['model'])
            full_evaluate(model, config, device, n_samples=50, n_vis=8)

    if not (args.precompute or args.train or args.eval):
        print("No action. Use --precompute, --train, or --eval")


if __name__ == "__main__":
    main()
