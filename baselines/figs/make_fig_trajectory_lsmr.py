"""Renders the forward-process trajectory figure."""
import os, sys, argparse
import numpy as np
import torch
try:
    import astra
except Exception:
    astra = None
from scipy.sparse.linalg import lsmr
from scipy.ndimage import uniform_filter, binary_fill_holes, label as cc_label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, f"{CODE_ROOT}")
import train_ldct_dose_bridge as T
from unified_eval import compute_psnr

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRE = f"{DATA_ROOT}/LDCT_dose_bridge/precomputed_equal_improvement"
plt.rcParams['font.family'] = 'serif'; plt.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'; plt.rcParams['pdf.fonttype'] = 42


def body_bbox(x0, thr=-500.0, pad=3):
    """common crop box from the full-dose image: HU > thr, binary_fill_holes, largest connected component, +pad px.
    returns (r0, r1, c0, c1) as half-open slices"""
    m = binary_fill_holes(x0 > thr)
    lab, nlab = cc_label(m)
    if nlab > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0; m = lab == int(np.argmax(sizes))
    rows = np.flatnonzero(m.any(axis=1)); cols = np.flatnonzero(m.any(axis=0))
    r0, r1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, m.shape[0])
    c0, c1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, m.shape[1])
    return r0, r1, c0, c1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sids', required=True)
    ap.add_argument('--recon', default='lsmr', choices=['lsmr', 'fbp'])
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--window', default='-160,240', help='display window in HU')
    ap.add_argument('--err_smooth', type=int, default=7)
    ap.add_argument('--err_max_hu', type=float, default=0.0)
    ap.add_argument('--out_dir', required=True); ap.add_argument('--out_suffix', default='')
    ap.add_argument('--no_err', action='store_true', help='image row only (no error-map row)')
    ap.add_argument('--no_title', action='store_true', help='no header line above the panels')
    ap.add_argument('--from_npz', default='', help='re-layout from a saved npz (keys imgs, times); skips astra/LSMR (CPU only)')
    ap.add_argument('--crop', action='store_true', help='crop all panels to the body bbox of imgs[0] (HU>-500, fill holes, largest CC, +crop_pad px)')
    ap.add_argument('--crop_thr', type=float, default=-500.0); ap.add_argument('--crop_pad', type=int, default=3)
    ap.add_argument('--label_w', type=float, default=0.28, help='width (in) reserved for the rotated left label; 0 drops the label')
    ap.add_argument('--title_pad', type=float, default=2.0, help='title pad in pt'); ap.add_argument('--title_fs', type=float, default=6.5, help='title font size in pt')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    config = T.Config(); config.compute_alpha(); alpha = config.alpha; n = config.recon_size
    vs = 1.0 / (config.VOXEL_SIZE_512 * (512.0 / n))
    lo, hi = [float(v) for v in args.window.split(',')]
    cache = {}
    for sid in args.sids.split(','):
        pid, s = sid.split('_s')[0], int(sid.split('_s')[1])
        if args.from_npz:
            d = np.load(args.from_npz); imgs = [x for x in d['imgs'].astype(np.float32)]; times = [float(t) for t in d['times']]
            noise = None; rr = cc = -1
        else:
            if astra is None:
                raise RuntimeError('astra import failed; the full pipeline needs a GPU node (or use --from_npz)')
            vol_geom = astra.create_vol_geom(n, n)
            if pid not in cache:
                cache = {pid: (T.load_patient_sinogram(pid, 'full', config), T.load_patient_angles(pid, config))}
            sino3d, angles = cache[pid]
            y_full = np.maximum(T.extract_slice_sinogram(sino3d, s), 0.0)
            ts = [float(np.load(f'{PRE}/{sid}_step{k}.npz')['t']) for k in range(5)]
            times = [0.0] + ts
            rng = np.random.RandomState(args.seed + s)
            q_full = np.maximum(np.round(config.I0_high * np.exp(-y_full)).astype(np.int64), 1)
            sinos = [y_full]
            for t in ts:
                a_t = alpha ** t; q_t = np.maximum(rng.binomial(q_full, a_t), 1)
                sinos.append((-np.log(q_t.astype(np.float64) / (config.I0_high * a_t))).astype(np.float32))
            if args.recon == 'lsmr':
                pg = astra.create_proj_geom('fanflat', vs * config.DU, config.n_det, angles, vs * config.DSO, vs * config.DDO)
                pid_ = astra.create_projector('cuda', pg, vol_geom); A = astra.OpTomo(pid_)
                def R(y):
                    rhs = (np.flip(y, axis=1) * vs).astype(np.float64).ravel()
                    x = lsmr(A, rhs, damp=0.0, maxiter=args.iters, atol=1e-8, btol=1e-8)[0].reshape(n, n)
                    return T.to_hu(np.maximum(x, 0).astype(np.float32), config)
            else:
                def R(y):
                    return T.to_hu(T.correct_fbp(y, n, angles, config), config)
            imgs = [R(y) for y in sinos]
            fbp0 = T.to_hu(T.correct_fbp(y_full, n, angles, config), config)
            soft = (fbp0 > -150) & (fbp0 < 250); ls = np.sqrt(np.maximum(uniform_filter(fbp0 ** 2, 15) - uniform_filter(fbp0, 15) ** 2, 0))
            ls[uniform_filter(soft.astype(np.float32), 15) < 0.999] = np.inf; rr, cc = np.unravel_index(np.argmin(ls), ls.shape)
            noise = [float(np.std(x[rr - 7:rr + 8, cc - 7:cc + 8])) for x in imgs]
            if args.recon == 'lsmr':
                astra.projector.delete(pid_)
        x0 = imgs[0]; scale = float(x0.max() - x0.min()) + 1e-8; vmin = float(x0.min())
        nrm = lambda x: np.clip((x - vmin) / scale, 0, 1).astype(np.float32)
        psnr = [compute_psnr(nrm(x), nrm(x0)) for x in imgs]
        def err(x):
            e = x - x0
            return np.sqrt(uniform_filter(e ** 2, args.err_smooth)) if args.err_smooth > 1 else np.abs(e)
        emax = args.err_max_hu or float(np.round(np.percentile(err(imgs[-1]), 99) / 10) * 10)
        m = len(imgs); W = 7.0; lw = max(float(args.label_w), 0.0); pw = (W - lw) / m
        if args.crop:
            r0, r1, c0, c1 = body_bbox(x0, thr=args.crop_thr, pad=args.crop_pad); ph = pw * (r1 - r0) / (c1 - c0)
        else:
            r0, r1, c0, c1 = 0, x0.shape[0], 0, x0.shape[1]; ph = pw
        crop = lambda x: x[r0:r1, c0:c1]
        nrow = 1 if args.no_err else 2; top = 0.24 if args.no_title else 0.36; H = nrow * ph + top + 0.14
        fig = plt.figure(figsize=(W, H)); left = lw / W
        cols = ['#1f5fbf'] + ['k'] * (m - 2) + ['#c62828']
        for c in range(m):
            t = times[c]; d = 100 * alpha ** t
            ax = fig.add_axes([left + c * pw / W, 1 - (top + ph) / H, pw / W * 0.985, ph / H * 0.985]); ax.axis('off')
            ax.imshow(np.clip((crop(imgs[c]) - lo) / (hi - lo), 0, 1), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            lab = '$t=0$ (full dose)' if c == 0 else f'$t=1$ ({d:.0f}% dose)' if c == m - 1 else f'$t={t:.2f}$ | {d:.1f}% dose'
            ax.set_title(lab + ('' if c == 0 else f'\n{psnr[c]:.1f} dB'), fontsize=args.title_fs, color=cols[c], fontweight='bold', pad=args.title_pad, linespacing=1.1)
            if not args.no_err:
                ax2 = fig.add_axes([left + c * pw / W, 1 - (top + 2 * ph) / H, pw / W * 0.985, ph / H * 0.985]); ax2.axis('off')
                ax2.imshow(crop(err(imgs[c])), cmap='hot', vmin=0, vmax=emax, interpolation='nearest')
        if lw > 0:
            lx = min(0.012, 0.5 * lw / W)
            fig.text(lx, 1 - (top + 0.5 * ph) / H, 'reconstruction $\\mathcal{R}(y^{(t)})$', rotation=90, ha='center', va='center', fontsize=6.5)
            if not args.no_err:
                fig.text(lx, 1 - (top + 1.5 * ph) / H, f'local RMSE (0–{emax:.0f} HU)', rotation=90, ha='center', va='center', fontsize=6.5)
        if not args.no_title:
            fig.text(0.5 + left / 2, 1 - 0.05 / H, r'forward process: binomial photon thinning of the detected counts, dose $I_0\alpha^t$ decreases with $t$ $\longrightarrow$', ha='center', va='center', fontsize=6.5)
        tag = f'{args.recon}{args.iters if args.recon == "lsmr" else ""}'
        base = os.path.join(args.out_dir, f'fig_traj_{tag}_ldct_{sid}{args.out_suffix}')
        fig.savefig(base + '.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02); fig.savefig(base + '.png', dpi=220, bbox_inches='tight', pad_inches=0.02); plt.close(fig)
        np.savez_compressed(base + '.npz', imgs=np.stack(imgs).astype(np.float32), times=np.array(times), bbox=np.array([r0, r1, c0, c1]))
        noise_s = (f'noise std (HU) in flat ROI ({rr},{cc}) per t: ' + ' '.join(f'{v:.1f}' for v in noise)) if noise is not None else 'noise: n/a (from_npz)'
        print(f'{sid} {tag}: emax {emax:.0f} HU | PSNR ' + ' '.join(f'{p:.1f}' for p in psnr[1:]) + ' (' + ' '.join(f'{p:.3f}' for p in psnr[1:]) + ') | ' + noise_s
              + f' | bbox r{r0}:{r1} c{c0}:{c1} ({r1 - r0}x{c1 - c0}) | label_w {lw:.2f} | panel {pw:.3f}x{ph:.3f} in | canvas {W:.2f}x{H:.2f} in', flush=True)


if __name__ == '__main__':
    main()
