"""Precomputes thinning realizations of the 2DeteCT trajectories."""
import os, sys, argparse, traceback
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_2detect_dose_bridge as T
from baselines.train_baselines import split_slice_ids

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

T_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=1); ap.add_argument('--end', type=int, default=1000)
    ap.add_argument('--n_rep', type=int, default=4)
    ap.add_argument('--sino_path', default=f"{DATA_ROOT}/2DeteCT_raw")
    ap.add_argument('--out', default=f"{DATA_ROOT}/2DeteCT_dose_bridge_rep4/precomputed")
    ap.add_argument('--I0_high', type=float, default=1e5); ap.add_argument('--alpha', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=9191)
    args = ap.parse_args()
    config = T.Config(); config.sino_path = args.sino_path; config.I0_high = args.I0_high
    config.alpha = args.alpha; config.I0_low = args.I0_high * args.alpha
    os.makedirs(args.out, exist_ok=True)
    projector = T.AstraProjector2D(config)
    sids = [f'slice{i:05d}' for i in range(1, 1001)]
    test = set(split_slice_ids(sids, '2detect', 'test'))
    todo = [i for i in range(args.start, args.end + 1) if f'slice{i:05d}' not in test]
    print(f"rep4 precompute: slices {args.start}-{args.end} ({len(todo)} non-test), knots {T_LIST}, n_rep={args.n_rep}", flush=True)
    for n, slice_idx in enumerate(todo):
        sid = f'slice{slice_idx:05d}'
        if all(os.path.exists(os.path.join(args.out, f"{sid}_step{k}_r{r}.npz")) for k in range(len(T_LIST)) for r in range(args.n_rep)):
            continue
        try:
            rng = np.random.default_rng(args.seed * 100000 + slice_idx)
            y_full = T.preprocess_sinogram(slice_idx, config.mode, config)
            q_full = np.maximum(np.round(config.I0_high * np.exp(-y_full)).astype(np.int64), 1)
            x0_raw = T.lsmr_recon(projector, y_full, config)
            vmin, vmax = float(x0_raw.min()), float(x0_raw.max()); scale = vmax - vmin + 1e-8
            norm = lambda x: np.clip((x - vmin) / scale, 0.0, 1.0).astype(np.float32)
            x0 = norm(x0_raw)
            for k, t in enumerate(T_LIST):
                a_t = config.alpha ** t; It = config.I0_high * a_t
                for r in range(args.n_rep):
                    qt = np.maximum(rng.binomial(q_full, a_t), 1)
                    yt = (-np.log(qt.astype(np.float64) / It)).astype(np.float32)
                    xt = norm(T.lsmr_recon(projector, yt, config))
                    np.savez_compressed(os.path.join(args.out, f"{sid}_step{k}_r{r}.npz"),
                                        x_t=xt, x0=x0, t=np.float32(t), slice_idx=slice_idx,
                                        vmin=np.float32(vmin), vmax=np.float32(vmax), rep=r)
            if n % 10 == 0:
                print(f"  {n}/{len(todo)} {sid} done", flush=True)
        except Exception as e:
            print(f"  Error {sid}: {e}"); traceback.print_exc()
    print("Done.")


if __name__ == '__main__':
    main()
