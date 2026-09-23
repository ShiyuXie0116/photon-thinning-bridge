"""Evaluation on the real low-dose scans of both datasets."""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_real_lowdose as ER
from baselines.train_baselines import REDCNN, UNetEndpoint
from baselines.train_boosters import REDCNNTime
from baselines.eval_all import try_lpips, lpips_val
from baselines.train_bridge import UNetRes, REDCNNWide, EDCNN
from unified_eval import get_time_steps

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ER.Config2DeteCT.sino_path = f"{DATA_ROOT}/2DeteCT_raw"

OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"
CACHE = os.path.join(OUT_ROOT, 'real_recon_cache')


def cache_ldct():
    """FBP-reconstruct all valid test slices once; cache normalized pairs."""
    os.makedirs(CACHE, exist_ok=True)
    cfg = ER.ConfigLDCT()
    pairs = {}
    for pid in cfg.patients_test:
        cpath = os.path.join(CACHE, f'ldct_{pid}.npz')
        if os.path.exists(cpath):
            d = np.load(cpath)
            for key in d.files:
                if key.endswith('_x0'):
                    sid = key[:-3]
                    pairs[sid] = (d[sid + '_x1'], d[sid + '_x0'])
            print(f"{pid}: {sum(1 for k in pairs if k.startswith(pid))} cached slices")
            continue
        angles = ER.load_patient_angles(pid, cfg)
        sino_full = ER.load_patient_sinogram(pid, 'full', cfg)
        sino_low = ER.load_patient_sinogram(pid, 'low', cfg)
        n_slices = sino_full.shape[2]
        valid = list(range(cfg.edge_margin, n_slices - cfg.edge_margin))
        print(f"{pid}: reconstructing {len(valid)} slices")
        store = {}
        for s in tqdm(valid, desc=pid):
            y_full = np.maximum(np.ascontiguousarray(sino_full[:, :, s].astype(np.float32)), 0)
            y_low = np.maximum(np.ascontiguousarray(sino_low[:, :, s].astype(np.float32)), 0)
            xf = ER.to_hu(ER.correct_fbp(y_full, cfg.recon_size, angles, cfg), cfg)
            xl = ER.to_hu(ER.correct_fbp(y_low, cfg.recon_size, angles, cfg), cfg)
            vmin, vmax = float(xf.min()), float(xf.max())
            scale = vmax - vmin + 1e-8
            x0 = np.clip((xf - vmin) / scale, 0, 1).astype(np.float32)
            x1 = np.clip((xl - vmin) / scale, 0, 1).astype(np.float32)
            sid = f'{pid}_s{s:04d}'
            store[sid + '_x0'] = x0
            store[sid + '_x1'] = x1
            pairs[sid] = (x1, x0)
        np.savez_compressed(cpath, **store)
    return pairs


def cache_2detect():
    os.makedirs(CACHE, exist_ok=True)
    cfg = ER.Config2DeteCT()
    cpath = os.path.join(CACHE, '2detect_real.npz')
    pairs = {}
    if os.path.exists(cpath):
        d = np.load(cpath)
        for key in d.files:
            if key.endswith('_x0'):
                sid = key[:-3]
                pairs[sid] = (d[sid + '_x1'], d[sid + '_x0'])
        print(f"2detect: {len(pairs)} cached slices")
        return pairs
    projector = ER.AstraProjector2D(cfg)
    test_slices = ER.get_2detect_test_slices()
    print(f"2detect: reconstructing {len(test_slices)} test slices (LSMR x2 each)")
    store = {}
    for s in tqdm(test_slices):
        y_full = ER.preprocess_sinogram(s, 2, cfg)
        y_low = ER.preprocess_sinogram(s, 1, cfg)
        xf = ER.lsmr_recon(projector, y_full, cfg)
        xl = ER.lsmr_recon(projector, y_low, cfg)
        vmin, vmax = float(xf.min()), float(xf.max())
        scale = vmax - vmin + 1e-8
        x0 = np.clip((xf - vmin) / scale, 0, 1).astype(np.float32)
        x1 = np.clip((xl - vmin) / scale, 0, 1).astype(np.float32)
        sid = f'slice{s:05d}'
        store[sid + '_x0'] = x0
        store[sid + '_x1'] = x1
        pairs[sid] = (x1, x0)
    projector.cleanup()
    np.savez_compressed(cpath, **store)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ldct', '2detect'])
    ap.add_argument('--save_pred', action='store_true',
                    help='also write every final prediction image to preds_real.npz (keys "<method>|real|<sid>") '
                         'so they can be re-scored against alternative references')
    args = ap.parse_args()
    device = torch.device('cuda')
    out_dir = os.path.join(OUT_ROOT, f'eval_real_{args.dataset}')
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == 'ldct':
        alpha = 0.1
        bridge_ckpt = f"{DATA_ROOT}/LDCT_dose_bridge/output_geometric_x0pred/ldct_dose_bridge_best.pth"
        ep_unet = f'{OUT_ROOT}/ldct_unet_endpoint/best.pth'
        ep_red = f'{OUT_ROOT}/ldct_redcnn_endpoint/best.pth'
        boosters = [
            ('bridge_redcnn', f'{OUT_ROOT}/ldct_redcnn_bridge_uniform/best.pth',
             'redcnn_blind', alpha),
            ('bridge_redcnn_t',
             f'{OUT_ROOT}/ldct_redcnn_t_bridge_equal_improvement/best.pth',
             'redcnn_t', alpha),
            ('endpoint_redcnn_t', f'{OUT_ROOT}/ldct_redcnn_t_endpoint/best.pth',
             'redcnn_t1', None),
            ('bridge_unet_res', f'{OUT_ROOT}/ldct_unet_res_bridge_uniform/best.pth',
             'unet_res_t:uniform', alpha),
            ('bridge_unet_res_EI', f'{OUT_ROOT}/ldct_unet_res_bridge_equal_improvement/best.pth',
             'unet_res_t:equal_improvement', alpha),
            ('endpoint_unet_res', f'{OUT_ROOT}/ldct_unet_res_endpoint_uniform/best.pth',
             'unet_res_ep', None),
            ('bridge_redcnn_EI_ep200', f'{OUT_ROOT}/ldct_redcnn_bridge_equal_improvement_ep200/best.pth',
             'redcnn_blind', alpha),
            ('endpoint_redcnn_ep1000', f'{OUT_ROOT}/ldct_redcnn_endpoint_equal_improvement_ep1000/best.pth',
             'redcnn_endpoint', None),
            ('endpoint_redcnn_r6ep400', f'{OUT_ROOT}/ldct_redcnn_endpoint_equal_improvement/best.pth',
             'redcnn_endpoint', None),
            ('endpoint_redcnn_r6ep100', f'{OUT_ROOT}/ldct_redcnn_endpoint_equal_improvement_ep100/best.pth',
             'redcnn_endpoint', None),
            ('endpoint_unet_res_ep100', f'{OUT_ROOT}/ldct_unet_res_endpoint_equal_improvement_ep100/best.pth',
             'unet_res_ep', None),
            ('endpoint_dncnn_ep100', f'{OUT_ROOT}/ldct_dncnn_endpoint_equal_improvement_ep100/best.pth', 'r6:dncnn', None),
            ('endpoint_edcnn_ep100', f'{OUT_ROOT}/ldct_edcnn_endpoint_equal_improvement_ep100/best.pth', 'r6:edcnn', None),
            ('endpoint_nafnet_ep100', f'{OUT_ROOT}/ldct_nafnet_endpoint_equal_improvement_ep100/best.pth', 'r6:nafnet', None),
            ('endpoint_drunet_ep100', f'{OUT_ROOT}/ldct_drunet_endpoint_equal_improvement_ep100/best.pth', 'r6:drunet', None),
            ('endpoint_mapnn_ep100', f'{OUT_ROOT}/ldct_mapnn_endpoint_equal_improvement_ep100/best.pth', 'r6:mapnn', None),
            ('endpoint_restormer_ep100', f'{OUT_ROOT}/ldct_restormer_endpoint_equal_improvement_ep100/best.pth', 'r6:restormer', None),
            ('endpoint_wganvgg_pub', f'{OUT_ROOT}/ldct_wganvgg_endpoint_equal_improvement_pub/best.pth', 'r6:wganvgg', None),
            ('bridge_redcnn_wide_EI_ep200', f'{OUT_ROOT}/ldct_redcnn_wide_bridge_equal_improvement_ep200/best.pth',
             'wide_blind', alpha),
            ('ours_unet_res_aug', f'{OUT_ROOT}/ldct_unet_res_bridge_uniform_aug_ema_ep160/best.pth',
             'unet_res_t:uniform', alpha),
            ('ours_unet_res_aug_lr2e4', f'{OUT_ROOT}/ldct_unet_res_bridge_uniform_aug_ema_lr0.0002_ep160/best.pth',
             'unet_res_t:uniform', alpha),
            ('ours_unet96_aug_lr2e4', f'{OUT_ROOT}/ldct_unet_res96_bridge_uniform_aug_ema_lr0.0002_ep160/best.pth',
             'unet_res96_t:uniform', alpha),
            ('ours_hybrid_rep4', f'{OUT_ROOT}/ldct_rep4_hybrid_bridge_uniform_aug_ema_lr0.0002_ep21/best.pth',
             'hybrid_t:uniform', alpha),
            ('ours_hybrid_rep1', f'{OUT_ROOT}/ldct_hybrid_bridge_uniform_aug_ema_lr0.0002_ep85/best.pth',
             'hybrid_t:uniform', alpha),
            ('endpoint_unet_res_aug_ep1000', f'{OUT_ROOT}/ldct_unet_res_endpoint_uniform_aug_ema_ep1000/best.pth',
             'unet_res_ep', None),
            ('endpoint_redcnn_aug_ep1000', f'{OUT_ROOT}/ldct_redcnn_endpoint_equal_improvement_aug_ema_ep1000/best.pth',
             'redcnn_endpoint', None),
            ('endpoint_edcnn_aug_ep1000', f'{OUT_ROOT}/ldct_edcnn_endpoint_equal_improvement_aug_ema_ep1000/best.pth',
             'edcnn_endpoint', None),
            ('endpoint_redcnn_wide_aug_ep1000', f'{OUT_ROOT}/ldct_redcnn_wide_endpoint_equal_improvement_aug_ema_ep1000/best.pth',
             'wide_endpoint', None),
            ('endpoint_redcnn_wide_ep1000', f'{OUT_ROOT}/ldct_redcnn_wide_endpoint_equal_improvement_ep1000/best.pth',
             'wide_endpoint', None),
        ]
        pairs = cache_ldct()
    else:
        alpha = 1500.0 / 53000.0
        bridge_ckpt = f"{DATA_ROOT}/2DeteCT_dose_bridge_realI0/output_geometric_x0pred/dose_bridge_best.pth"
        ep_unet = f'{OUT_ROOT}/2detect_realI0_unet_endpoint/best.pth'
        ep_red = f'{OUT_ROOT}/2detect_realI0_redcnn_endpoint/best.pth'
        boosters = [
            ('bridge_redcnn',
             f'{OUT_ROOT}/2detect_realI0_redcnn_bridge_geometric/best.pth',
             'redcnn_blind', alpha),
            ('bridge_x0_effI0',
             f"{DATA_ROOT}/2DeteCT_dose_bridge_effI0/output_geometric_x0pred/dose_bridge_best.pth",
             'unet_t', 425.0 / 15000.0),
            ('endpoint_unet_effI0', f'{OUT_ROOT}/2detect_effI0_unet_endpoint/best.pth',
             'unet_endpoint', None),
            ('endpoint_redcnn_effI0', f'{OUT_ROOT}/2detect_effI0_redcnn_endpoint/best.pth',
             'redcnn_endpoint', None),
            ('bridge_redcnn_effI0',
             f'{OUT_ROOT}/2detect_effI0_redcnn_bridge_geometric/best.pth',
             'redcnn_blind', 425.0 / 15000.0),
            ('bridge_unet_res', f'{OUT_ROOT}/2detect_realI0_unet_res_bridge_geometric/best.pth',
             'unet_res_t:geometric', alpha),
            ('endpoint_unet_res', f'{OUT_ROOT}/2detect_realI0_unet_res_endpoint_geometric/best.pth',
             'unet_res_ep', None),
            ('bridge_unet_res_effI0', f'{OUT_ROOT}/2detect_effI0_unet_res_bridge_geometric/best.pth',
             'unet_res_t:geometric', 425.0 / 15000.0),
            ('endpoint_unet_res_effI0', f'{OUT_ROOT}/2detect_effI0_unet_res_endpoint_geometric/best.pth',
             'unet_res_ep', None),
            ('endpoint_unet_res_effI0_ep100', f'{OUT_ROOT}/2detect_effI0_unet_res_endpoint_geometric_ep100/best.pth',
             'unet_res_ep', None),
            ('endpoint_redcnn_effI0_ep100', f'{OUT_ROOT}/2detect_effI0_redcnn_endpoint_geometric_ep100/best.pth',
             'redcnn_endpoint', None),
            ('endpoint_dncnn_effI0_ep100', f'{OUT_ROOT}/2detect_effI0_dncnn_endpoint_geometric_ep100/best.pth', 'r6:dncnn', None),
            ('endpoint_edcnn_effI0_ep100', f'{OUT_ROOT}/2detect_effI0_edcnn_endpoint_geometric_ep100/best.pth', 'r6:edcnn', None),
            ('endpoint_nafnet_effI0_ep100', f'{OUT_ROOT}/2detect_effI0_nafnet_endpoint_geometric_ep100/best.pth', 'r6:nafnet', None),
            ('endpoint_drunet_effI0_ep100', f'{OUT_ROOT}/2detect_effI0_drunet_endpoint_geometric_ep100/best.pth', 'r6:drunet', None),
            ('endpoint_mapnn_effI0_ep100', f'{OUT_ROOT}/2detect_effI0_mapnn_endpoint_geometric_ep100/best.pth', 'r6:mapnn', None),
            ('endpoint_restormer_effI0_ep100', f'{OUT_ROOT}/2detect_effI0_restormer_endpoint_geometric_ep100/best.pth', 'r6:restormer', None),
            ('endpoint_wganvgg_effI0_pub', f'{OUT_ROOT}/2detect_effI0_wganvgg_endpoint_geometric_pub/best.pth', 'r6:wganvgg', None),
            ('bridge_redcnn_effI0_ep200', f'{OUT_ROOT}/2detect_effI0_redcnn_bridge_geometric_ep200/best.pth',
             'redcnn_blind', 425.0 / 15000.0),
            ('endpoint_redcnn_effI0_ep1000', f'{OUT_ROOT}/2detect_effI0_redcnn_endpoint_geometric_ep1000/best.pth',
             'redcnn_endpoint', None),
            ('bridge_redcnn_wide_effI0_ep200', f'{OUT_ROOT}/2detect_effI0_redcnn_wide_bridge_geometric_ep200/best.pth',
             'wide_blind', 425.0 / 15000.0),
            ('endpoint_redcnn_wide_effI0_ep1000', f'{OUT_ROOT}/2detect_effI0_redcnn_wide_endpoint_geometric_ep1000/best.pth',
             'wide_endpoint', None),
            ('ours_unet_res_effI0_aug', f'{OUT_ROOT}/2detect_effI0_unet_res_bridge_geometric_aug_ema_ep160/best.pth',
             'unet_res_t:geometric', 425.0 / 15000.0),
            ('ours_unet96_effI0_aug_lr2e4', f'{OUT_ROOT}/2detect_effI0_unet_res96_bridge_geometric_aug_ema_lr0.0002_ep160/best.pth',
             'unet_res96_t:geometric', 425.0 / 15000.0),
            ('ours_unet96ref_effI0_aug_lr2e4', f'{OUT_ROOT}/2detect_effI0_unet_res96_ref_bridge_geometric_aug_ema_lr0.0002_ep160/best.pth',
             'unet_res96_ref_t:geometric', 425.0 / 15000.0),
            ('ours_hybrid96_effI0_aug_lr2e4', f'{OUT_ROOT}/2detect_effI0_hybrid96_bridge_geometric_aug_ema_lr0.0002_ep160/best.pth',
             'hybrid96_t:geometric', 425.0 / 15000.0),
            ('endpoint_unet_res_effI0_aug_ep1000', f'{OUT_ROOT}/2detect_effI0_unet_res_endpoint_geometric_aug_ema_ep1000/best.pth',
             'unet_res_ep', None),
            ('endpoint_redcnn_effI0_aug_ep1000', f'{OUT_ROOT}/2detect_effI0_redcnn_endpoint_geometric_aug_ema_ep1000/best.pth',
             'redcnn_endpoint', None),
            ('endpoint_redcnn_wide_effI0_aug_ep1000', f'{OUT_ROOT}/2detect_effI0_redcnn_wide_endpoint_geometric_aug_ema_ep1000/best.pth',
             'wide_endpoint', None),
        ]
        pairs = cache_2detect()

    sids = sorted(pairs.keys())
    print(f"{args.dataset}: {len(sids)} real test slices, alpha={alpha:.4f}")
    times = ER.get_geometric_schedule(alpha, 5)

    lp = try_lpips(device)
    per = {}
    cur = {'sid': None}
    saved = {}
    import re as _re

    def record(method, pred, gt):
        if args.save_pred and cur['sid'] is not None and not _re.search(r'_step[1-4]$', method):
            saved[f'{method}|real|{cur["sid"]}'] = np.asarray(pred, dtype=np.float32)
        e = per.setdefault(method, {'psnr': [], 'ssim': [], 'lpips': []})
        e['psnr'].append(ER.compute_psnr(pred, gt))
        e['ssim'].append(ER.compute_ssim(pred, gt))
        v = lpips_val(lp, pred, gt, device)
        if v is not None:
            e['lpips'].append(v)

    for sid in sids:
        x1, x0 = pairs[sid]
        cur['sid'] = sid
        record('low_dose', x1, x0)

    # bridge model (dose-uniform x0): single + multi + per-step
    if os.path.exists(bridge_ckpt):
        model = ER.UNetWithTime(base_ch=64, t_dim=128).to(device)
        model.load_state_dict(torch.load(bridge_ckpt, map_location=device)['model'])
        model.eval()
        for sid in tqdm(sids, desc='bridge_x0'):
            x1, x0 = pairs[sid]
            cur['sid'] = sid
            with torch.no_grad():
                xt = torch.from_numpy(x1).float()[None, None].to(device)
                tt = torch.tensor([1.0], dtype=torch.float32).to(device)
                single = np.clip(model(xt, tt).cpu().numpy().squeeze(), 0, 1)
            record('bridge_x0_single', single, x0)
            traj = ER.run_trajectory(model, x1, times, device)
            for j in range(1, len(traj)):
                record(f'bridge_x0_step{j}', traj[j][1], x0)
        del model
        torch.cuda.empty_cache()
    else:
        print(f"MISSING bridge ckpt: {bridge_ckpt}")

    for name, path, arch in [('endpoint_unet', ep_unet, 'unet'),
                             ('endpoint_redcnn', ep_red, 'redcnn')]:
        if not os.path.exists(path):
            print(f"SKIP {name}: {path}")
            continue
        model = (UNetEndpoint() if arch == 'unet' else REDCNN()).to(device)
        model.load_state_dict(torch.load(path, map_location=device)['model'])
        model.eval()
        for sid in tqdm(sids, desc=name):
            x1, x0 = pairs[sid]
            cur['sid'] = sid
            with torch.no_grad():
                pred = model(torch.from_numpy(x1).float()[None, None].to(device))
            record(name, np.clip(pred.cpu().numpy().squeeze(), 0, 1), x0)
        del model
        torch.cuda.empty_cache()

    class BlindWrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x, t=None):
            return self.net(x)

    for name, path, kind, a_roll in boosters:
        if not os.path.exists(path):
            print(f"SKIP {name}: {path}")
            continue
        try:
            sd = torch.load(path, map_location=device)
            if kind == 'redcnn_blind':
                model = BlindWrap(REDCNN()).to(device)
                model.net.load_state_dict(sd['model'])
            elif kind == 'wide_blind':
                model = BlindWrap(REDCNNWide()).to(device)
                model.net.load_state_dict(sd['model'])
            elif kind == 'wide_endpoint':
                model = REDCNNWide().to(device)
                model.load_state_dict(sd['model'])
            elif kind == 'edcnn_endpoint':
                model = EDCNN().to(device)
                model.load_state_dict(sd['model'])
            elif kind == 'unet_t':
                model = ER.UNetWithTime(base_ch=64, t_dim=128).to(device)
                model.load_state_dict(sd['model'])
            elif kind in ('redcnn_t', 'redcnn_t1'):
                model = REDCNNTime().to(device)
                model.load_state_dict(sd['model'])
            elif kind == 'unet_endpoint':
                model = UNetEndpoint().to(device)
                model.load_state_dict(sd['model'])
            elif kind.startswith('unet_res') and '_ref' in kind.split(':')[0]:
                from baselines.train_bridge import build as _build_bridge
                model = _build_bridge(kind.split('_t:')[0])[0].to(device)   # e.g. unet_res_ref_t:uniform
                model.load_state_dict(sd['model'])
            elif kind.split('_t:')[0] in ('hybrid', 'hybrid96', 'wide_film'):
                from baselines.train_bridge import build as _build_bridge
                model = _build_bridge(kind.split('_t:')[0])[0].to(device)
                model.load_state_dict(sd['model'])
            elif kind.startswith('unet_res'):
                model = UNetRes(base_ch=96 if kind.startswith('unet_res96') else 64).to(device)
                model.load_state_dict(sd['model'])
            elif kind.startswith('r6:'):
                from baselines.train_bridge import build as _build_bridge
                model, _r6_timed = _build_bridge(kind[3:]); model = model.to(device)
                model.load_state_dict(sd['model'])
            else:
                model = REDCNN().to(device)
                model.load_state_dict(sd['model'])
        except Exception as e:
            print(f"SKIP {name}: load failed ({e})")
            continue
        model.eval()
        _r6_timed = locals().get('_r6_timed', False) if kind.startswith('r6:') else False
        r9_t = kind.split('_t:')[0] in ('hybrid', 'hybrid96', 'wide_film') and '_t:' in kind
        multi = kind in ('redcnn_blind', 'wide_blind', 'unet_t', 'redcnn_t') or kind.startswith('unet_res_t') or r9_t
        if kind.startswith('unet_res_t') or r9_t:
            roll_times = get_time_steps(a_roll, 5, kind.split(':')[1])
        else:
            roll_times = ER.get_geometric_schedule(a_roll, 5) if multi else None
        for sid in tqdm(sids, desc=name):
            x1, x0 = pairs[sid]
            cur['sid'] = sid
            with torch.no_grad():
                xt = torch.from_numpy(x1).float()[None, None].to(device)
                if kind in ('redcnn_blind', 'wide_blind', 'unet_t', 'redcnn_t', 'redcnn_t1') or kind.startswith('unet_res') or r9_t or (kind.startswith('r6:') and _r6_timed):
                    tt = torch.tensor([1.0], dtype=torch.float32).to(device)
                    pred = model(xt, tt)
                else:
                    pred = model(xt)
            record(f'{name}_single' if multi else name,
                   np.clip(pred.cpu().numpy().squeeze(), 0, 1), x0)
            if multi:
                traj = ER.run_trajectory(model, x1, roll_times, device)
                for j in range(1, len(traj)):
                    record(f'{name}_step{j}', traj[j][1], x0)
        del model
        torch.cuda.empty_cache()

    if saved:
        fn = os.path.join(out_dir, 'preds_real.npz')
        np.savez_compressed(fn, **saved)
        print(f"saved {len(saved)} prediction images -> {fn}")
    np.savez_compressed(
        os.path.join(out_dir, 'per_slice.npz'),
        slice_ids=np.array(sids),
        **{f'{m}__{k}': np.array(v[k]) for m, v in per.items() for k in v if v[k]})

    from scipy.stats import wilcoxon
    summary, pvals = {}, {}
    for m, v in sorted(per.items()):
        summary[m] = {
            'psnr_mean': float(np.mean(v['psnr'])), 'psnr_std': float(np.std(v['psnr'])),
            'ssim_mean': float(np.mean(v['ssim'])), 'ssim_std': float(np.std(v['ssim'])),
            'n': len(v['psnr'])}
        if v['lpips']:
            summary[m]['lpips_mean'] = float(np.mean(v['lpips']))
    finals = [m for m in per if not ('_step' in m and not m.endswith('_step5'))]
    for i, m1 in enumerate(sorted(finals)):
        for m2 in sorted(finals)[i + 1:]:
            a, b = np.array(per[m1]['psnr']), np.array(per[m2]['psnr'])
            if len(a) == len(b):
                try:
                    pvals[f'{m1}|{m2}'] = {
                        'psnr_p': float(wilcoxon(a, b).pvalue),
                        'psnr_diff': float(np.mean(a) - np.mean(b))}
                except Exception:
                    pass

    result = {'dataset': args.dataset, 'alpha': alpha, 'n_test': len(sids),
              'summary': summary, 'wilcoxon': pvals}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nREAL EVAL v2 — {args.dataset} ({len(sids)} slices)")
    for m in sorted(summary):
        s = summary[m]
        line = f"{m:<22s}{s['psnr_mean']:>8.2f}±{s['psnr_std']:<5.2f}{s['ssim_mean']:>8.4f}"
        if 'lpips_mean' in s:
            line += f"{s['lpips_mean']:>8.4f}"
        print(line)
    print(f"saved -> {out_dir}")


if __name__ == '__main__':
    main()
