"""Reconstruction and evaluation helpers for the real low-dose scans."""

import os, json, argparse, traceback, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

try:
    import astra
    HAS_ASTRA = True
except ImportError:
    HAS_ASTRA = False


# Network (identical to training scripts)

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
    if mse < 1e-10: return 100.0
    data_range = max(gt.max() - gt.min(), 1e-10)
    return 10.0 * np.log10(data_range ** 2 / mse)

def compute_ssim(pred, gt, win_size=7):
    data_range = max(gt.max() - gt.min(), 1e-10)
    C1, C2 = (0.01 * data_range)**2, (0.03 * data_range)**2
    mu_x = uniform_filter(pred, size=win_size, mode='reflect')
    mu_y = uniform_filter(gt, size=win_size, mode='reflect')
    s_x = np.maximum(uniform_filter(pred**2, win_size, mode='reflect') - mu_x**2, 0)
    s_y = np.maximum(uniform_filter(gt**2, win_size, mode='reflect') - mu_y**2, 0)
    s_xy = uniform_filter(pred * gt, win_size, mode='reflect') - mu_x * mu_y
    return float(np.mean(((2*mu_x*mu_y+C1)*(2*s_xy+C2)) / ((mu_x**2+mu_y**2+C1)*(s_x+s_y+C2))))


def get_ei_schedule(alpha, n_steps):
    N = n_steps
    s = np.zeros(N + 1)
    for k in range(N + 1):
        u_k = 1.0 + (N - k) / N * (1.0 / alpha - 1.0)
        s[k] = 0.0 if k == N else -np.log(u_k) / np.log(alpha)
    return s[::-1].astype(np.float32)  # ascending [0, ..., 1]

def get_geometric_schedule(alpha, n_steps):
    u_steps = np.linspace(1.0, alpha, n_steps + 1)
    u_steps = np.clip(u_steps, 1e-15, None)
    return (np.log(u_steps) / np.log(alpha)).astype(np.float32)

def run_trajectory(model, x_lowdose, times, device):
    """Run x0-prediction multi-step, return trajectory at each step."""
    t_reversed = times[::-1]
    x_current = x_lowdose.copy()
    trajectory = [(float(t_reversed[0]), x_current.copy(), None)]

    with torch.no_grad():
        for i in range(len(t_reversed) - 1):
            t_curr = float(t_reversed[i])
            t_next = float(t_reversed[i + 1])

            x_t = torch.from_numpy(x_current).float().unsqueeze(0).unsqueeze(0).to(device)
            t_t = torch.tensor([t_curr], dtype=torch.float32).to(device)
            x0_pred = model(x_t, t_t).cpu().numpy().squeeze()
            x0_pred = np.clip(x0_pred, 0.0, 1.0)

            if t_next < 1e-6:
                x_current = x0_pred
            else:
                ratio = t_next / t_curr
                x_current = x0_pred + ratio * (x_current - x0_pred)

            x_current = np.clip(x_current, 0.0, 1.0)
            trajectory.append((t_next, x_current.copy(), x0_pred.copy()))

    return trajectory


class Config2DeteCT:
    sino_path = f"{DATA_ROOT}/2DeteCT"
    DET_PIX = 0.0748
    SOD = 431.019989
    SDD = 529.000488
    recon_size = 256
    n_angles_full = 3600
    lsmr_iter = 50
    lsmr_damp = 1e-2

def preprocess_sinogram(slice_idx, mode, config):
    """Load and preprocess 2DeteCT sinogram."""
    from scipy.interpolate import interp1d
    from PIL import Image

    base = os.path.join(config.sino_path, f'slice{slice_idx:05d}', f'mode{mode}')
    sino = np.array(Image.open(os.path.join(base, 'sinogram.tif')), dtype=np.float32)
    dark = np.array(Image.open(os.path.join(base, 'dark.tif')), dtype=np.float32)
    flat1 = np.array(Image.open(os.path.join(base, 'flat1.tif')), dtype=np.float32)
    flat2 = np.array(Image.open(os.path.join(base, 'flat2.tif')), dtype=np.float32)
    flat = (flat1 + flat2) / 2.0

    sino = sino[:, 0::2] + sino[:, 1::2]
    dark = dark[0, 0::2] + dark[0, 1::2]
    flat = flat[0, 0::2] + flat[0, 1::2]

    data = ((sino - dark) / (flat - dark))[:-1, :]

    shift = config.DET_PIX if (slice_idx in range(1, 2831) or slice_idx in range(5521, 5871)) else 0.0
    if shift != 0.0:
        det_pix_sz = 2 * config.DET_PIX
        grid = np.arange(956) * det_pix_sz
        data = interp1d(grid, data, axis=1, kind='linear',
                        bounds_error=False, fill_value='extrapolate')(grid + shift)

    data = np.clip(data, 1e-6, None)
    data = -np.log(data)
    return np.ascontiguousarray(data.astype(np.float32))


class AstraProjector2D:
    def __init__(self, config):
        self.n = config.recon_size
        self.na = config.n_angles_full
        self.nd = 956

        det_pix_sz = 2 * config.DET_PIX
        vox = det_pix_sz * self.nd * config.SOD / config.SDD / self.n

        angles = np.linspace(0, 2 * np.pi, self.na, endpoint=False)
        self.pg = astra.create_proj_geom(
            'fanflat', det_pix_sz / vox, self.nd,
            angles, config.SOD / vox, (config.SDD - config.SOD) / vox
        )
        self.vg = astra.create_vol_geom(self.n, self.n)
        self.proj_id = astra.create_projector('cuda', self.pg, self.vg)
        self.img_size = self.n * self.n

    def forward(self, x):
        img = x.reshape(self.n, self.n).astype(np.float32)
        vid = astra.data2d.create('-vol', self.vg, img)
        sid = astra.data2d.create('-sino', self.pg, 0)
        cfg = astra.astra_dict('FP_CUDA')
        cfg['ProjectorId'] = self.proj_id
        cfg['VolumeDataId'] = vid
        cfg['ProjectionDataId'] = sid
        aid = astra.algorithm.create(cfg)
        astra.algorithm.run(aid)
        sino = astra.data2d.get(sid).flatten()
        astra.algorithm.delete(aid)
        astra.data2d.delete(vid)
        astra.data2d.delete(sid)
        return sino.astype(np.float64)

    def adjoint(self, y):
        sino = y.reshape(self.na, self.nd).astype(np.float32)
        sid = astra.data2d.create('-sino', self.pg, sino)
        vid = astra.data2d.create('-vol', self.vg, 0)
        cfg = astra.astra_dict('BP_CUDA')
        cfg['ProjectorId'] = self.proj_id
        cfg['ProjectionDataId'] = sid
        cfg['ReconstructionDataId'] = vid
        aid = astra.algorithm.create(cfg)
        astra.algorithm.run(aid)
        img = astra.data2d.get(vid).flatten()
        astra.algorithm.delete(aid)
        astra.data2d.delete(sid)
        astra.data2d.delete(vid)
        return img.astype(np.float64)

    def cleanup(self):
        astra.projector.delete(self.proj_id)


def lsmr_recon(projector, sinogram, config):
    from scipy.sparse.linalg import lsmr as scipy_lsmr
    from scipy.sparse.linalg import LinearOperator

    na, nd = sinogram.shape
    A_op = LinearOperator(
        shape=(na * nd, projector.img_size),
        matvec=lambda x: projector.forward(x),
        rmatvec=lambda y: projector.adjoint(y),
        dtype=np.float64
    )
    rhs = sinogram.flatten().astype(np.float64)
    result = scipy_lsmr(A_op, rhs, damp=config.lsmr_damp,
                        maxiter=config.lsmr_iter, atol=1e-6, btol=1e-6, show=False)
    x = result[0].reshape(config.recon_size, config.recon_size)
    return np.maximum(x, 0).astype(np.float32)


def get_2detect_test_slices():
    """Get test slices using same split as training (seed=42, 80/10/10)."""
    all_slices = list(range(1, 1001))
    np.random.seed(42)
    perm = np.random.permutation(len(all_slices))
    n_train = int(len(all_slices) * 0.8)
    n_val = int(len(all_slices) * 0.1)
    return [all_slices[i] for i in perm[n_train + n_val:]]


def eval_2detect_real(model, device, args):
    """Evaluate on 2DeteCT real low-dose (mode1) vs full-dose (mode2)."""
    config = Config2DeteCT()
    projector = AstraProjector2D(config)
    alpha = args.I0_low / args.I0_high
    n_steps = 5
    times = get_geometric_schedule(alpha, n_steps)

    test_slices = get_2detect_test_slices()
    print(f"2DeteCT real low-dose eval: {len(test_slices)} test slices")
    print(f"  alpha={alpha}, schedule={times}")

    rng = np.random.RandomState(42)
    vis_slices = rng.choice(test_slices, min(args.n_vis, len(test_slices)), replace=False)

    all_metrics = {'lowdose': {'psnr': [], 'ssim': []},
                   'single_step': {'psnr': [], 'ssim': []},
                   'multi_step': {'psnr': [], 'ssim': []}}
    all_step_psnrs = [[] for _ in range(n_steps + 1)]
    all_step_ssims = [[] for _ in range(n_steps + 1)]
    vis_data = []

    for s in tqdm(test_slices, desc="Evaluating"):
        try:
            y_full = preprocess_sinogram(s, 2, config)
            y_low = preprocess_sinogram(s, 1, config)

            x_full = lsmr_recon(projector, y_full, config)
            x_low = lsmr_recon(projector, y_low, config)

            vmin = float(x_full.min())
            vmax = float(x_full.max())
            scale = vmax - vmin + 1e-8
            x0 = np.clip((x_full - vmin) / scale, 0.0, 1.0).astype(np.float32)
            x1 = np.clip((x_low - vmin) / scale, 0.0, 1.0).astype(np.float32)

            all_metrics['lowdose']['psnr'].append(compute_psnr(x1, x0))
            all_metrics['lowdose']['ssim'].append(compute_ssim(x1, x0))

            with torch.no_grad():
                x_t = torch.from_numpy(x1).float().unsqueeze(0).unsqueeze(0).to(device)
                t_t = torch.tensor([1.0], dtype=torch.float32).to(device)
                x0_single = model(x_t, t_t).cpu().numpy().squeeze()
                x0_single = np.clip(x0_single, 0.0, 1.0)
            all_metrics['single_step']['psnr'].append(compute_psnr(x0_single, x0))
            all_metrics['single_step']['ssim'].append(compute_ssim(x0_single, x0))

            traj = run_trajectory(model, x1, times, device)
            x_final = traj[-1][1]
            all_metrics['multi_step']['psnr'].append(compute_psnr(x_final, x0))
            all_metrics['multi_step']['ssim'].append(compute_ssim(x_final, x0))

            for j, (t_val, x_img, _) in enumerate(traj):
                all_step_psnrs[j].append(compute_psnr(x_img, x0))
                all_step_ssims[j].append(compute_ssim(x_img, x0))

            if s in vis_slices:
                vis_data.append({'gt': x0, 'ld': x1, 'traj': traj, 'slice': s})

        except Exception as e:
            print(f"  Error slice {s}: {e}")
            traceback.print_exc()

    projector.cleanup()
    return all_metrics, all_step_psnrs, all_step_ssims, vis_data, times, n_steps


class ConfigLDCT:
    recon_dir = f"{DATA_ROOT}/LDCT_recon"
    DSO = 595.0
    DDO = 490.6
    DU = 1.285839319229126
    n_angles = 1152
    n_det = 736
    VOXEL_SIZE_512 = 0.7421875
    HU_FACTOR = 0.0192
    recon_size = 256
    edge_margin = 30
    patients_test = ['C004', 'C120']


def load_patient_sinogram(patient_id, dose, config):
    from PIL import Image
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
        sino_3d[i] = np.array(img, dtype=np.float32)
    img.close()
    return sino_3d


def load_patient_angles(patient_id, config):
    npz_path = os.path.join(config.recon_dir, f'{patient_id}_full_complete', 'rebinning_params.npz')
    npz = np.load(npz_path, allow_pickle=True)
    angles = npz['angles'].astype(np.float64)[:config.n_angles] + np.pi / 2
    return angles


def correct_fbp(sinogram, img_size, angles, config):
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
    return 1000.0 * (recon - config.HU_FACTOR) / config.HU_FACTOR


def eval_ldct_real(model, device, args):
    """Evaluate on LDCT real low-dose vs full-dose (test patients)."""
    config = ConfigLDCT()
    alpha = args.I0_low / args.I0_high
    n_steps = 5
    times = get_geometric_schedule(alpha, n_steps)

    print(f"LDCT real low-dose eval: patients={config.patients_test}")
    print(f"  alpha={alpha}, schedule={times}")

    all_metrics = {'lowdose': {'psnr': [], 'ssim': []},
                   'single_step': {'psnr': [], 'ssim': []},
                   'multi_step': {'psnr': [], 'ssim': []}}
    all_step_psnrs = [[] for _ in range(n_steps + 1)]
    all_step_ssims = [[] for _ in range(n_steps + 1)]
    vis_data = []
    rng = np.random.RandomState(42)

    for pid in config.patients_test:
        print(f"\nLoading {pid}...")
        try:
            angles = load_patient_angles(pid, config)
            sino_full = load_patient_sinogram(pid, 'full', config)
            sino_low = load_patient_sinogram(pid, 'low', config)
        except Exception as e:
            print(f"  Error loading {pid}: {e}")
            continue

        n_slices = sino_full.shape[2]
        margin = config.edge_margin
        valid = list(range(margin, n_slices - margin))
        chosen = rng.choice(valid, min(50, len(valid)), replace=False)
        vis_chosen = set(chosen[:args.n_vis // len(config.patients_test) + 1])

        print(f"  {pid}: {len(chosen)} slices (of {len(valid)} valid)")

        for s in tqdm(chosen, desc=f"  {pid}"):
            try:
                y_full = np.ascontiguousarray(sino_full[:, :, s].astype(np.float32))
                y_low = np.ascontiguousarray(sino_low[:, :, s].astype(np.float32))

                y_full = np.maximum(y_full, 0.0)
                y_low = np.maximum(y_low, 0.0)

                x_full_hu = to_hu(correct_fbp(y_full, config.recon_size, angles, config), config)
                x_low_hu = to_hu(correct_fbp(y_low, config.recon_size, angles, config), config)

                vmin = float(x_full_hu.min())
                vmax = float(x_full_hu.max())
                scale = vmax - vmin + 1e-8
                x0 = np.clip((x_full_hu - vmin) / scale, 0.0, 1.0).astype(np.float32)
                x1 = np.clip((x_low_hu - vmin) / scale, 0.0, 1.0).astype(np.float32)

                all_metrics['lowdose']['psnr'].append(compute_psnr(x1, x0))
                all_metrics['lowdose']['ssim'].append(compute_ssim(x1, x0))

                with torch.no_grad():
                    x_t = torch.from_numpy(x1).float().unsqueeze(0).unsqueeze(0).to(device)
                    t_t = torch.tensor([1.0], dtype=torch.float32).to(device)
                    x0_single = model(x_t, t_t).cpu().numpy().squeeze()
                    x0_single = np.clip(x0_single, 0.0, 1.0)
                all_metrics['single_step']['psnr'].append(compute_psnr(x0_single, x0))
                all_metrics['single_step']['ssim'].append(compute_ssim(x0_single, x0))

                traj = run_trajectory(model, x1, times, device)
                x_final = traj[-1][1]
                all_metrics['multi_step']['psnr'].append(compute_psnr(x_final, x0))
                all_metrics['multi_step']['ssim'].append(compute_ssim(x_final, x0))

                for j, (t_val, x_img, _) in enumerate(traj):
                    all_step_psnrs[j].append(compute_psnr(x_img, x0))
                    all_step_ssims[j].append(compute_ssim(x_img, x0))

                if s in vis_chosen and len(vis_data) < args.n_vis:
                    vis_data.append({'gt': x0, 'ld': x1, 'traj': traj,
                                     'slice': f"{pid}_s{s}"})

            except Exception as e:
                print(f"  Error {pid} s{s}: {e}")
                traceback.print_exc()

    return all_metrics, all_step_psnrs, all_step_ssims, vis_data, times, n_steps


def plot_trajectory(vis_data, n_steps, dataset, alpha, output_dir,
                    win_lo=1.0, win_hi=99.0):
    """Render the per-sample trajectory PNG using a percentile-based display window."""
    os.makedirs(output_dir, exist_ok=True)
    if not vis_data:
        return None

    n_cols = n_steps + 3
    nv = len(vis_data)
    fig, axes = plt.subplots(nv, n_cols, figsize=(2.5 * n_cols, 2.8 * nv))
    if nv == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        f"Real Low-Dose Eval: x0-Pred + Geometric ({dataset.upper()}, α={alpha:.2f}) "
        f"[window p{win_lo:g}–p{win_hi:g}]",
        fontsize=13, fontweight='bold', y=1.01)

    for row, d in enumerate(vis_data):
        x0_gt = d['gt']
        traj = d['traj']
        vm, vx = np.percentile(x0_gt, [win_lo, win_hi])
        if vx - vm < 1e-6:
            vm, vx = float(x0_gt.min()), float(x0_gt.max())

        axes[row, 0].imshow(x0_gt, cmap='gray', vmin=vm, vmax=vx)
        axes[row, 0].set_title("GT (full-dose)", fontsize=9, fontweight='bold')
        axes[row, 0].axis('off')

        for j, (t_val, x_img, x0_pred) in enumerate(traj):
            col = j + 1
            p = compute_psnr(x_img, x0_gt)
            s = compute_ssim(x_img, x0_gt)

            axes[row, col].imshow(x_img, cmap='gray', vmin=vm, vmax=vx)

            if j == 0:
                label = f"Real LD\nt={t_val:.2f}"
                color = 'red'
            elif j == len(traj) - 1:
                label = f"Step {j}\nt={t_val:.2f}"
                color = 'green'
            else:
                label = f"Step {j}\nt={t_val:.2f}"
                color = 'blue'

            axes[row, col].set_title(f"{label}\n{p:.1f}dB / {s:.3f}",
                                     fontsize=7, color=color, fontweight='bold')
            axes[row, col].axis('off')

        final_img = traj[-1][1]
        err = np.abs(final_img - x0_gt)
        axes[row, -1].imshow(err, cmap='hot', vmin=0, vmax=max(err.max() * 0.5, 0.01))
        axes[row, -1].set_title("|Error|", fontsize=9)
        axes[row, -1].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'real_lowdose_trajectory_{dataset}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path


def visualize_and_save(all_metrics, all_step_psnrs, all_step_ssims, vis_data,
                       times, n_steps, dataset, alpha, output_dir,
                       win_lo=1.0, win_hi=99.0):
    os.makedirs(output_dir, exist_ok=True)

    plot_trajectory(vis_data, n_steps, dataset, alpha, output_dir,
                    win_lo=win_lo, win_hi=win_hi)

    if vis_data:
        cache_path = os.path.join(output_dir, f'vis_data_{dataset}.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump({'vis_data': vis_data, 'times': times, 'n_steps': n_steps,
                         'dataset': dataset, 'alpha': float(alpha)}, f)
        print(f"Saved: {cache_path}")

    if all_step_psnrs[0]:
        t_reversed = times[::-1]
        step_means_psnr = [np.mean(p) for p in all_step_psnrs]
        step_means_ssim = [np.mean(s) for s in all_step_ssims]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Real Low-Dose: Per-Step Metrics\n"
                     f"{dataset.upper()}, α={alpha:.2f}, {len(all_step_psnrs[0])} test slices",
                     fontsize=12, fontweight='bold')

        step_labels = ["LD"] + [f"S{i}" for i in range(1, n_steps + 1)]
        t_labels = [f"{float(t_reversed[j]):.3f}" for j in range(n_steps + 1)]
        x_pos = range(n_steps + 1)

        ax1.plot(x_pos, step_means_psnr, 'bo-', linewidth=2, markersize=8)
        for i, (x, y) in enumerate(zip(x_pos, step_means_psnr)):
            ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=9, fontweight='bold')
        ax1.set_xticks(list(x_pos))
        ax1.set_xticklabels([f"{sl}\n(t={tl})" for sl, tl in zip(step_labels, t_labels)], fontsize=8)
        ax1.set_ylabel("PSNR (dB)", fontsize=11)
        ax1.set_xlabel("Inference Step", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("PSNR trajectory", fontsize=11)

        if len(step_means_psnr) > 1:
            total_gain = step_means_psnr[-1] - step_means_psnr[0]
            ax1.annotate(f"Total: +{total_gain:.2f} dB",
                         xy=(n_steps, step_means_psnr[-1]),
                         xytext=(n_steps - 1.5, step_means_psnr[0] + 0.5),
                         fontsize=10, color='green', fontweight='bold',
                         arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

        ax2.plot(x_pos, step_means_ssim, 'ro-', linewidth=2, markersize=8)
        for i, (x, y) in enumerate(zip(x_pos, step_means_ssim)):
            ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=9, fontweight='bold')
        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels([f"{sl}\n(t={tl})" for sl, tl in zip(step_labels, t_labels)], fontsize=8)
        ax2.set_ylabel("SSIM", fontsize=11)
        ax2.set_xlabel("Inference Step", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("SSIM trajectory", fontsize=11)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'real_lowdose_perstep_{dataset}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")

    print("\n" + "=" * 70)
    print(f"REAL LOW-DOSE RESULTS ({dataset.upper()}, α={alpha:.2f})")
    print("=" * 70)
    for method in ['lowdose', 'single_step', 'multi_step']:
        if all_metrics[method]['psnr']:
            pm = np.mean(all_metrics[method]['psnr'])
            ps = np.std(all_metrics[method]['psnr'])
            sm = np.mean(all_metrics[method]['ssim'])
            ss = np.std(all_metrics[method]['ssim'])
            n = len(all_metrics[method]['psnr'])
            print(f"  {method:15s}: PSNR={pm:.2f}±{ps:.2f}  SSIM={sm:.4f}±{ss:.4f}  (n={n})")
    print("=" * 70)

    if all_step_psnrs[0]:
        t_reversed = times[::-1]
        print(f"\nPer-step metrics ({len(all_step_psnrs[0])} samples):")
        print(f"{'Step':<8} {'t':<8} {'PSNR (dB)':<18} {'SSIM':<18}")
        print("-" * 55)
        for j in range(n_steps + 1):
            t_val = float(t_reversed[j]) if j < len(t_reversed) else 0.0
            pm = np.mean(all_step_psnrs[j])
            ps = np.std(all_step_psnrs[j])
            sm = np.mean(all_step_ssims[j])
            ss = np.std(all_step_ssims[j])
            label = "LD input" if j == 0 else f"Step {j}"
            print(f"{label:<8} {t_val:<8.4f} {pm:.2f}±{ps:.2f}{'':>4} {sm:.4f}±{ss:.4f}")

    results = {
        'dataset': dataset,
        'alpha': float(alpha),
        'n_test': len(all_metrics['lowdose']['psnr']),
    }
    for method in ['lowdose', 'single_step', 'multi_step']:
        if all_metrics[method]['psnr']:
            results[method] = {
                'psnr_mean': float(np.mean(all_metrics[method]['psnr'])),
                'psnr_std': float(np.std(all_metrics[method]['psnr'])),
                'ssim_mean': float(np.mean(all_metrics[method]['ssim'])),
                'ssim_std': float(np.std(all_metrics[method]['ssim'])),
            }
    if all_step_psnrs[0]:
        t_reversed = times[::-1]
        results['per_step'] = []
        for j in range(n_steps + 1):
            results['per_step'].append({
                'step': j,
                't': float(t_reversed[j]) if j < len(t_reversed) else 0.0,
                'psnr_mean': float(np.mean(all_step_psnrs[j])),
                'psnr_std': float(np.std(all_step_psnrs[j])),
                'ssim_mean': float(np.mean(all_step_ssims[j])),
                'ssim_std': float(np.std(all_step_ssims[j])),
            })

    json_path = os.path.join(output_dir, f'real_lowdose_results_{dataset}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate dose bridge on real low-dose data')
    parser.add_argument('--dataset', required=True, choices=['2detect', 'ldct'])
    parser.add_argument('--ckpt', help='Path to model checkpoint (not needed for --replot_only)')
    parser.add_argument('--base_dir', required=True, help='Output directory for results')
    parser.add_argument('--I0_high', type=float, default=1e5,
                        help='Full-dose photon intensity (default: 1e5)')
    parser.add_argument('--I0_low', type=float,
                        help='Low-dose intensity matching training (e.g. 3000 or 25000)')
    parser.add_argument('--n_vis', type=int, default=8, help='Number of samples to visualize')
    parser.add_argument('--win_lo', type=float, default=1.0,
                        help='Lower percentile for trajectory display window (default 1.0)')
    parser.add_argument('--win_hi', type=float, default=99.0,
                        help='Upper percentile for trajectory display window (default 99.0)')
    parser.add_argument('--replot_only', action='store_true',
                        help='Skip eval; reload vis_data_<dataset>.pkl and redraw trajectory PNG only')
    args = parser.parse_args()

    if args.replot_only:
        cache_path = os.path.join(args.base_dir, f'vis_data_{args.dataset}.pkl')
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache {cache_path} not found. Run a full eval once to populate it.")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        plot_trajectory(cache['vis_data'], cache['n_steps'], cache['dataset'],
                        cache['alpha'], args.base_dir,
                        win_lo=args.win_lo, win_hi=args.win_hi)
        return

    if args.ckpt is None or args.I0_low is None:
        parser.error("--ckpt and --I0_low are required when not using --replot_only")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha = args.I0_low / args.I0_high

    print(f"Loading checkpoint: {args.ckpt}")
    model = UNetWithTime(base_ch=64, t_dim=128).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

    if args.dataset == '2detect':
        metrics, step_p, step_s, vis, times, ns = eval_2detect_real(model, device, args)
    else:
        metrics, step_p, step_s, vis, times, ns = eval_ldct_real(model, device, args)

    visualize_and_save(metrics, step_p, step_s, vis, times, ns,
                       args.dataset, alpha, args.base_dir,
                       win_lo=args.win_lo, win_hi=args.win_hi)


if __name__ == "__main__":
    main()
