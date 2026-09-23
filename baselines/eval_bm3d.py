"""BM3D baseline with the dose-matched noise level."""
import os, sys, json, argparse
import numpy as np
import bm3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_eval import compute_psnr, compute_ssim
from baselines.train_baselines import split_slice_ids
from baselines.eval_dose_sweep import KNOTS
from baselines.eval_all import try_lpips, lpips_val

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"


def _bm3d_one(a):   # worker: (npz path, sigma) -> (sid, den, x0)
    path, sigma = a
    z = np.load(path); xt, x0 = z['x_t'].astype(np.float32), z['x0'].astype(np.float32)
    den = np.clip(bm3d.bm3d(xt, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES), 0, 1).astype(np.float32)
    return os.path.basename(path).split('_step')[0], den, x0, float(z['t'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['2detect', 'ldct'])
    ap.add_argument('--family', default=None, help='knot family: uniform (2detect) / EI (ldct)')
    ap.add_argument('--n_train_sigma', type=int, default=100)
    ap.add_argument('--n_test', type=int, default=0)
    ap.add_argument('--sweep_dir', default=None)
    ap.add_argument('--steps', default='0,1,2,3,4', help='knot steps to run')
    ap.add_argument('--lpips', action='store_true', help='also compute LPIPS (CPU torch)')
    ap.add_argument('--tag', default='', help='suffix for the output json (e.g. _lpips)')
    ap.add_argument('--save_pred', default='', help='round 16: npz path to store the denoised test images (keys bm3d|<key>|<sid>)')
    ap.add_argument('--workers', type=int, default=1, help='round 16: multiprocessing workers for BM3D')
    ap.add_argument('--knot_dir', default='', help='round 16: evaluate the {sid}_step<k>.npz files of this dir instead of the sweep knots')
    ap.add_argument('--all_slices', action='store_true', help='round 16: use every slice of --knot_dir for both the oracle sigma and the test set')
    args = ap.parse_args()
    import torch
    lp = try_lpips(torch.device('cpu')) if args.lpips else None
    ds = args.dataset
    fam = args.family or ('uniform' if ds == '2detect' else 'EI')
    d = args.knot_dir or KNOTS[ds][fam]
    sweep_dir = args.sweep_dir or f'{OUT_ROOT}/eval_sweep_{ds}_v1'
    files = sorted(f for f in os.listdir(d) if f.endswith('_step4.npz'))
    sids = sorted({f.split('_step')[0] for f in files})
    if args.all_slices:
        train, test = list(sids), list(sids)
    else:
        train = split_slice_ids(sids, ds, 'train')[:args.n_train_sigma]
        test = split_slice_ids(sids, ds, 'test')
    if args.n_test:
        test = test[:args.n_test]
    res = {'bm3d': {}}
    knot_t = {}
    saved = {}
    for k in [int(x) for x in args.steps.split(',')]:
        key = f'{fam}_step{k}' if not args.knot_dir else f'real_step{k}'
        # dose-oracle sigma from training slices
        se, n = 0.0, 0
        for sid in train:
            z = np.load(os.path.join(d, f'{sid}_step{k}.npz'))
            se += float(np.mean((z['x_t'] - z['x0']) ** 2)); n += 1
        sigma = float(np.sqrt(se / max(n, 1)))
        ps, ss, lps = [], [], []
        jobs = [(os.path.join(d, f'{sid}_step{k}.npz'), sigma) for sid in test]
        if args.workers > 1:
            from multiprocessing import Pool
            with Pool(args.workers) as pool:
                outs = pool.map(_bm3d_one, jobs)
        else:
            outs = [_bm3d_one(j) for j in jobs]
        for sid, den, x0, tk in outs:
            knot_t[key] = tk
            if args.save_pred:
                saved[f'bm3d|{key}|{sid}'] = den
            ps.append(compute_psnr(den, x0)); ss.append(compute_ssim(den, x0))
            if lp is not None:
                v = lpips_val(lp, den, x0, torch.device('cpu'))
                if v is not None:
                    lps.append(v)
        res['bm3d'][key] = {'psnr': float(np.mean(ps)), 'psnr_std': float(np.std(ps)),
                            'ssim': float(np.mean(ss)), 'n': len(ps), 'sigma': sigma}
        if lps:
            res['bm3d'][key]['lpips'] = float(np.mean(lps))
        print(f"{key} t={knot_t[key]:.3f} sigma={sigma:.4f}: BM3D {np.mean(ps):.2f}±{np.std(ps):.2f} dB SSIM {np.mean(ss):.3f}", flush=True)
        np.save(os.path.join(sweep_dir, f'bm3d_{key}_psnr.npy'), np.array(ps))
    out = os.path.join(sweep_dir, f'extra_bm3d{args.tag}.json')
    if args.tag and os.path.exists(os.path.join(sweep_dir, 'extra_bm3d.json')):
        full = json.load(open(os.path.join(sweep_dir, 'extra_bm3d.json')))
        for key, v in res['bm3d'].items():
            full['models']['bm3d'].setdefault(key, {}).update(v)
        with open(os.path.join(sweep_dir, 'extra_bm3d.json'), 'w') as f:
            json.dump(full, f, indent=1)
    with open(out, 'w') as f:
        json.dump({'models': res, 'knot_t': knot_t}, f, indent=1)
    print('written', out)
    if args.save_pred:
        np.savez_compressed(args.save_pred, **saved); print('saved preds ->', args.save_pred)


if __name__ == '__main__':
    main()
