"""Diffusion prior baseline on Mayo LDCT."""
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("DIFFUSION_BASELINES", "external/diffusion_baselines")
sys.path.insert(0, os.path.join(DB, 'DDS'))
sys.path.insert(0, DB)
sys.path.insert(0, os.path.dirname(HERE))
from guided_diffusion.script_util import create_model
from utils import get_beta_schedule
from unified_eval import compute_psnr, compute_ssim
from baselines.train_baselines import split_slice_ids
from baselines.eval_all import try_lpips, lpips_val

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"
PRE_EI = f"{DATA_ROOT}/LDCT_dose_bridge/precomputed_equal_improvement"
PRIOR = f"{DATA_ROOT}/diffusion_priors/ldct256_x0/ema_latest.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--knots', default='EI_step4,EI_step3,EI_step2,EI_step1,EI_step0')
    ap.add_argument('--T_sampling', type=int, default=50)
    ap.add_argument('--n_test', type=int, default=0)
    ap.add_argument('--n_train_sigma', type=int, default=200)
    ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--sweep_dir', default=f'{OUT_ROOT}/eval_sweep_ldct')
    ap.add_argument('--knot_dir', default='', help='round 16: dir of {sid}_step4.npz files (e.g. the real pairs) used instead of PRE_EI')
    ap.add_argument('--all_slices', action='store_true', help='round 16: every slice of --knot_dir is both the sigma set and the test set')
    ap.add_argument('--save_pred', default='', help='round 16: npz path to store the predictions (keys <mode>|<key>|<sid>)')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    device = torch.device('cuda')
    os.makedirs(args.sweep_dir, exist_ok=True)

    model = create_model(image_size=256, num_channels=128, num_res_blocks=2, attention_resolutions='16,8',
                         num_head_channels=64, learn_sigma=True, use_scale_shift_norm=True,
                         resblock_updown=True, in_channels=1).to(device).eval()
    sd = torch.load(args.prior, map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=True)
    betas = torch.from_numpy(get_beta_schedule('linear', beta_start=1e-4, beta_end=2e-2,
                                               num_diffusion_timesteps=1000)).float().to(device)
    abar = torch.cumprod(1 - betas, 0)
    lp = try_lpips(device)

    PRE_DIR = args.knot_dir or PRE_EI
    sids = sorted({f.split('_step')[0] for f in os.listdir(PRE_DIR) if f.endswith('_step4.npz')})
    if args.all_slices:
        train, test = list(sids), list(sids)
    else:
        train = split_slice_ids(sids, 'ldct', 'train')[:args.n_train_sigma]
        test = split_slice_ids(sids, 'ldct', 'test')
    if args.n_test:
        test = test[:args.n_test]
    print(f"prior {args.prior}; {len(train)} train slices for sigma, {len(test)} test slices", flush=True)

    def eps_pred(x, t_int):
        t = torch.full((x.shape[0],), float(t_int), device=device)
        with torch.no_grad():
            et = model(x, t)
        return et[:, :et.size(1) // 2]

    def ddim_from(x_start_m, tau, steps):
        ts = np.linspace(tau, 0, steps + 1).round().astype(int)
        x = x_start_m
        for i in range(len(ts) - 1):
            t, tn = int(ts[i]), int(ts[i + 1])
            at = abar[t]; et = eps_pred(x, t)
            x0 = ((x - (1 - at).sqrt() * et) / at.sqrt()).clamp(-1, 1)
            x = x0 if tn == 0 else abar[tn].sqrt() * x0 + (1 - abar[tn]).sqrt() * et
        return x

    per = {}

    saved = {}

    def record(m, key, pred, gt, sid):
        e = per.setdefault(m, {}).setdefault(key, {'psnr': [], 'ssim': [], 'lpips': [], 'sid': []})
        e['psnr'].append(compute_psnr(pred, gt)); e['ssim'].append(compute_ssim(pred, gt))
        v = lpips_val(lp, pred, gt, device)
        if v is not None:
            e['lpips'].append(v)
        e['sid'].append(sid)
        if args.save_pred:
            saved[f'{m}|{key}|{sid}'] = np.asarray(pred, np.float32)

    knot_t = {}
    for key in args.knots.split(','):
        step = key.split('_step')[1]
        se, n = 0.0, 0
        for sid in train:
            z = np.load(os.path.join(PRE_DIR, f'{sid}_step{step}.npz'))
            se += float(np.mean((z['x_t'] - z['x0']) ** 2)); n += 1
            knot_t[key] = float(z['t'])
        sigma_m = 2.0 * float(np.sqrt(se / n))
        snr = ((1 - abar) / abar).cpu().numpy()
        tau = int(np.argmin(np.abs(snr - sigma_m ** 2)))
        steps = max(5, min(args.T_sampling, tau))
        print(f"[ddim_denoise] {key}: t={knot_t[key]:.3f} sigma_m={sigma_m:.4f} -> tau={tau}, steps={steps}", flush=True)
        for sid in test:
            f = os.path.join(PRE_DIR, f'{sid}_step{step}.npz')
            if not os.path.exists(f):
                continue
            z = np.load(f)
            x1, x0 = z['x_t'].astype(np.float32), z['x0'].astype(np.float32)
            xm = torch.from_numpy(x1 * 2 - 1)[None, None].to(device)
            out = ddim_from(abar[tau].sqrt() * xm, tau, steps)
            pred = ((out + 1) / 2).clamp(0, 1).cpu().numpy()[0, 0]
            record('ddim_denoise', key, pred, x0, sid)
        e = per['ddim_denoise'][key]
        print(f"  -> {np.mean(e['psnr']):.2f}±{np.std(e['psnr']):.2f} dB, SSIM {np.mean(e['ssim']):.3f}, "
              f"LPIPS {np.mean(e['lpips']) if e['lpips'] else float('nan'):.4f}", flush=True)

    out = {'models': {}, 'knot_t': knot_t}
    for m, byk in per.items():
        out['models'][m] = {}
        for key, e in byk.items():
            s = {'psnr': float(np.mean(e['psnr'])), 'psnr_std': float(np.std(e['psnr'])),
                 'ssim': float(np.mean(e['ssim'])), 'n': len(e['psnr'])}
            if e['lpips']:
                s['lpips'] = float(np.mean(e['lpips']))
            out['models'][m][key] = s
            np.save(os.path.join(args.sweep_dir, f'{m}_{key}_psnr.npy'), np.array(e['psnr']))
    with open(os.path.join(args.sweep_dir, f'extra_diffusion{args.tag}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('written', os.path.join(args.sweep_dir, f'extra_diffusion{args.tag}.json'))
    if args.save_pred:
        np.savez_compressed(args.save_pred, **saved); print('saved preds ->', args.save_pred)


if __name__ == '__main__':
    main()
