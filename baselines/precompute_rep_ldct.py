"""Precomputes thinning realizations of the LDCT trajectories."""
import os, sys, argparse, hashlib, time, traceback
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ldct_dose_bridge as T

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = f"{DATA_ROOT}/LDCT_dose_bridge/precomputed"   # canonical uniform-knot realization (r0)
T_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]
TEST_PIDS = ('C004', 'C120')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', required=True, help='comma list of patient ids')
    ap.add_argument('--reps', default='1,2,3')
    ap.add_argument('--out', default=f"{DATA_ROOT}/LDCT_dose_bridge_rep4/precomputed")
    ap.add_argument('--seed', type=int, default=7373)
    ap.add_argument('--limit', type=int, default=0, help='smoke: only this many slices per patient')
    args = ap.parse_args()
    config = T.Config()
    config.compute_alpha()
    assert abs(config.alpha - 0.1) < 1e-9 and config.I0_high == 1e5
    os.makedirs(args.out, exist_ok=True)
    reps = [int(r) for r in args.reps.split(',') if r.strip()]
    for pid in args.patients.split(','):
        pid = pid.strip()
        assert pid not in TEST_PIDS, 'test patients are never generated'
        step0 = sorted(f for f in os.listdir(SRC) if f.startswith(pid + '_s') and f.endswith('_step0.npz'))
        sids = [f.replace('_step0.npz', '') for f in step0]
        if args.limit:
            sids = sids[:args.limit]
        for sid in sids:                                   # r0 = canonical realization (symlinks)
            for k in range(5):
                dst = os.path.join(args.out, f'{sid}_step{k}_r0.npz')
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(SRC, f'{sid}_step{k}.npz'), dst)
        todo = [sid for sid in sids if not all(os.path.exists(os.path.join(args.out, f'{sid}_step{k}_r{r}.npz'))
                                               for k in range(5) for r in reps)]
        if not todo:
            print(f'{pid}: already complete', flush=True)
            continue
        t0 = time.time()
        print(f'{pid}: {len(sids)} slices, {len(todo)} to generate, reps {reps}', flush=True)
        sino3d = T.load_patient_sinogram(pid, 'full', config)
        angles = T.load_patient_angles(pid, config)
        checked = False
        for n, sid in enumerate(todo):
            try:
                s = int(sid.split('_s')[1])
                z0 = np.load(os.path.join(SRC, f'{sid}_step0.npz'))
                x0 = z0['x0'].astype(np.float32)
                vmin, vmax = float(z0['vmin']), float(z0['vmax'])
                scale = vmax - vmin + 1e-8
                y_full = np.maximum(T.extract_slice_sinogram(sino3d, s), 0.0)
                if not checked:
                    x0_re = T.to_hu(T.correct_fbp(y_full, config.recon_size, angles, config), config)
                    x0_re = np.clip((x0_re - vmin) / scale, 0.0, 1.0).astype(np.float32)
                    err = float(np.abs(x0_re - x0).max())
                    print(f'  consistency {sid}: max|FBP(full) - stored x0| = {err:.2e}', flush=True)
                    if err > 1e-3:
                        raise RuntimeError('canonical x0 not reproduced (geometry/angles mismatch) -- aborting')
                    checked = True
                q_full = np.maximum(np.round(config.I0_high * np.exp(-y_full)).astype(np.int64), 1)
                x0_16 = x0.astype(np.float16)
                for r in reps:
                    h = int(hashlib.md5(f'{sid}_r{r}'.encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng([args.seed, r, h])
                    for k, t in enumerate(T_LIST):
                        a_t = config.alpha ** t
                        It = config.I0_high * a_t
                        qt = np.maximum(rng.binomial(q_full, a_t), 1)
                        yt = (-np.log(qt.astype(np.float64) / It)).astype(np.float32)
                        xt = T.to_hu(T.correct_fbp(yt, config.recon_size, angles, config), config)
                        xt = np.clip((xt - vmin) / scale, 0.0, 1.0).astype(np.float16)
                        np.savez_compressed(os.path.join(args.out, f'{sid}_step{k}_r{r}.npz'),
                                            x_t=xt, x0=x0_16, t=np.float32(t), vmin=np.float32(vmin),
                                            vmax=np.float32(vmax), rep=r, patient_id=pid, slice_idx=s)
                if n % 50 == 0:
                    print(f'  {n}/{len(todo)} {sid}  ({(time.time() - t0) / 60:.1f} min)', flush=True)
            except Exception as e:
                print(f'  Error {sid}: {e}')
                traceback.print_exc()
        del sino3d
        print(f'  {pid} done in {(time.time() - t0) / 60:.1f} min', flush=True)
    print('Done.')


if __name__ == '__main__':
    main()
