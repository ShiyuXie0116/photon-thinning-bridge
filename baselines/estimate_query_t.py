"""Reference-free estimate of the query time from the input noise level."""
import os, sys, json, argparse
import numpy as np
from glob import glob
from skimage.restoration import estimate_sigma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.train_baselines import split_slice_ids

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE = f"{DATA_ROOT}/dose_bridge_baselines/real_recon_cache"


def sig(x):
    return float(estimate_sigma(x.astype(np.float64), channel_axis=None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ldct', choices=['ldct', '2detect'])
    ap.add_argument('--n_sim', type=int, default=200, help='simulated test slices used for the line fit')
    ap.add_argument('--pre', default=None, help='override the simulated precompute dir used for the fit (e.g. the alpha=0.01 2DeteCT EI dir)')
    ap.add_argument('--out_tag', default='', help='suffix for the output json name')
    ap.add_argument('--split', default='test', choices=['test', 'val', 'train'], help='simulated split used for the noise->t line fit (val avoids touching test slices)')
    args = ap.parse_args()
    if args.dataset == 'ldct':
        pre = args.pre or f"{DATA_ROOT}/LDCT_dose_bridge/precomputed_equal_improvement"
        real = {p: np.load(os.path.join(CACHE, f'ldct_{p}.npz')) for p in ('C004', 'C120')}
    else:
        pre = args.pre or f"{DATA_ROOT}/2DeteCT_dose_bridge_effI0/precomputed_geometric"
        real = {'all': np.load(os.path.join(CACHE, '2detect_real.npz'))}
    sids = sorted({os.path.basename(f).split('_step')[0] for f in glob(os.path.join(pre, '*_step4.npz'))})
    test = split_slice_ids(sids, 'ldct' if args.dataset == 'ldct' else '2detect', args.split)
    rng = np.random.RandomState(0); sub = [test[i] for i in rng.permutation(len(test))[:args.n_sim]]
    ts, ls = [], []
    for sid in sub:
        for k in range(5):
            z = np.load(os.path.join(pre, f'{sid}_step{k}.npz'))
            ts.append(float(z['t'])); ls.append(np.log(sig(z['x_t'])))
    ts, ls = np.array(ts), np.array(ls)
    A = np.stack([ts, np.ones_like(ts)], 1); (m, c), *_ = np.linalg.lstsq(A, ls, rcond=None)
    resid = ls - (m * ts + c)
    print(f"{args.dataset}: fit log sigma = {m:.4f} t + {c:.4f} on {len(sub)} sim slices x 5 knots; resid std {resid.std():.3f}")
    for t in sorted(set(np.round(ts, 3))):
        sel = np.abs(ts - t) < 1e-3
        print(f"   knot t={t:.3f}: sigma {np.exp(ls[sel].mean()):.5f} (fit {np.exp(m*t+c):.5f})")
    out = {'fit': {'slope': m, 'intercept': c, 'resid_std': float(resid.std())}, 'per_patient': {}}
    for pid, d in real.items():
        th = []
        for key in d.files:
            if key.endswith('_x1'):
                th.append((np.log(sig(d[key])) - c) / m)
        th = np.array(th)
        out['per_patient'][pid] = {'t_hat_median': float(np.median(th)), 't_hat_mean': float(th.mean()),
                                   't_hat_q25': float(np.percentile(th, 25)), 't_hat_q75': float(np.percentile(th, 75)), 'n': len(th)}
        print(f"REAL {pid}: n={len(th)}  t_hat median {np.median(th):.3f}  mean {th.mean():.3f}  IQR [{np.percentile(th,25):.3f}, {np.percentile(th,75):.3f}]")
    fn = f"{DATA_ROOT}/dose_bridge_baselines/eval_real_ldct_tsweep/t_hat_{args.dataset}" + ('' if args.split == 'test' else f'_{args.split}') + (f'_{args.out_tag}' if args.out_tag else '') + '.json'
    os.makedirs(os.path.dirname(fn), exist_ok=True); json.dump(out, open(fn, 'w'), indent=1); print('written', fn)


if __name__ == '__main__':
    main()
