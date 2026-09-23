"""Image-domain DDIM denoiser built on a diffusion prior."""
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("DIFFUSION_BASELINES", "external/diffusion_baselines")
sys.path.insert(0, os.path.join(DB, 'DDS'))
sys.path.insert(0, DB)
sys.path.insert(0, os.path.dirname(HERE))
from guided_diffusion.script_util import create_model
from utils import CG, get_beta_schedule
import train_2detect_dose_bridge as T
from unified_eval import compute_psnr, compute_ssim
from baselines.train_baselines import split_slice_ids
from baselines.eval_all import try_lpips, lpips_val

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"
PRE = f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed"
PRE_EI = f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed_equal_improvement"
ALT = f"{DATA_ROOT}/2DeteCT_dose_bridge/alt_refs"
PRIOR = f"{DATA_ROOT}/diffusion_priors/2detect256_lsmr50/ema_latest.pt"


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='ddim_denoise,dds')
    ap.add_argument('--knots', default='uniform_step4,uniform_step2,uniform_step0',
                    help='knots for ddim_denoise on realization A')
    ap.add_argument('--T_sampling', type=int, default=50)
    ap.add_argument('--CG_iter', type=int, default=5)
    ap.add_argument('--eta', type=float, default=0.85)
    ap.add_argument('--n_test', type=int, default=0)
    ap.add_argument('--sweep_dir', default=f'{OUT_ROOT}/eval_sweep_2detect_v1')
    ap.add_argument('--knot_dir', default='', help='round 16: dir of {sid}_step4.npz files (e.g. the real pairs) used for every knot in --knots')
    ap.add_argument('--all_slices', action='store_true', help='round 16: every slice of --knot_dir is both the sigma set and the test set')
    ap.add_argument('--save_pred', default='', help='round 16: npz path to store the predictions (keys <mode>|<key>|<sid>)')
    ap.add_argument('--tag', default='', help='suffix of the output json (extra_diffusion<tag>.json)')
    args = ap.parse_args()
    device = torch.device('cuda')

    model = create_model(image_size=256, num_channels=128, num_res_blocks=2, attention_resolutions='16,8',
                         num_head_channels=64, learn_sigma=True, use_scale_shift_norm=True,
                         resblock_updown=True, in_channels=1).to(device).eval()
    sd = torch.load(PRIOR, map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=True)
    betas = torch.from_numpy(get_beta_schedule('linear', beta_start=1e-4, beta_end=2e-2,
                                               num_diffusion_timesteps=1000)).float().to(device)
    abar = torch.cumprod(1 - betas, 0)
    lp = try_lpips(device)

    sids = sorted({f.split('_step')[0] for f in os.listdir(args.knot_dir or PRE) if f.endswith('_step4.npz')})
    if args.all_slices:
        train, test = list(sids), list(sids)
    else:
        train = split_slice_ids(sids, '2detect', 'train')[:100]
        test = split_slice_ids(sids, '2detect', 'test')
    if args.n_test:
        test = test[:args.n_test]

    def eps_pred(x, t_int):
        t = torch.full((x.shape[0],), float(t_int), device=device)
        with torch.no_grad():
            et = model(x, t)
        return et[:, :et.size(1) // 2]

    def ddim_from(x_start_m, tau, steps):
        """Deterministic DDIM (eta=0) from state x_tau (model space) to 0."""
        ts = np.linspace(tau, 0, steps + 1).round().astype(int)
        x = x_start_m
        for i in range(len(ts) - 1):
            t, tn = int(ts[i]), int(ts[i + 1])
            at = abar[t]; et = eps_pred(x, t)
            x0 = (x - (1 - at).sqrt() * et) / at.sqrt()
            x0 = x0.clamp(-1, 1)
            if tn == 0:
                x = x0
            else:
                atn = abar[tn]
                x = atn.sqrt() * x0 + (1 - atn).sqrt() * et
        return x

    res, per = {}, {}

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

    if 'ddim_denoise' in args.modes.split(','):
        jobs = [(k, args.knot_dir or (PRE_EI if k.startswith('EI') else PRE), 'x_t', 'x0') for k in args.knots.split(',')]
        if os.path.isdir(ALT) and 'B' in args.knots.split(','):
            jobs.append(('B_step4', ALT, 'x1B', 'x0_pre'))
        for key, d, kin, kgt in jobs:
            step = key.split('_step')[1]
            se, n = 0.0, 0
            for sid in train:
                f = os.path.join(d if d != ALT else PRE, f'{sid}_step{step}.npz')   # d = --knot_dir when given
                z = np.load(f); se += float(np.mean((z['x_t'] - z['x0']) ** 2)); n += 1
            sigma_m = 2.0 * float(np.sqrt(se / n))
            # matching diffusion time: (1-abar)/abar = sigma^2
            snr = ((1 - abar) / abar).cpu().numpy()
            tau = int(np.argmin(np.abs(snr - sigma_m ** 2)))
            steps = max(5, min(args.T_sampling, tau))
            print(f"[ddim_denoise] {key}: sigma_m={sigma_m:.4f} -> tau={tau}, steps={steps}", flush=True)
            for sid in test:
                f = os.path.join(d, f'{sid}.npz') if d == ALT else os.path.join(d, f'{sid}_step{step}.npz')
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                x1, x0 = z[kin].astype(np.float32), z[kgt].astype(np.float32)
                xm = torch.from_numpy(x1 * 2 - 1)[None, None].to(device)
                x_tau = abar[tau].sqrt() * xm
                out = ddim_from(x_tau, tau, steps)
                pred = ((out + 1) / 2).clamp(0, 1).cpu().numpy()[0, 0]
                record('ddim_denoise', key, pred, x0, sid)
            e = per['ddim_denoise'][key]
            print(f"  -> {np.mean(e['psnr']):.2f}±{np.std(e['psnr']):.2f} dB, SSIM {np.mean(e['ssim']):.3f}", flush=True)

    if 'dds' in args.modes.split(',') and os.path.isdir(ALT):
        cfg = T.Config(); proj = T.AstraProjector2D(cfg)
        NA, ND = proj.na, proj.nd

        class Op:
            def A(self, x):
                xs = x.detach().cpu().numpy().reshape(-1, 256, 256)
                ys = np.stack([proj.forward(xi).reshape(NA, ND) for xi in xs]).astype(np.float32)
                return torch.from_numpy(ys)[:, None].to(x.device)

            def AT(self, y):
                ys = y.detach().cpu().numpy().reshape(-1, NA, ND)
                xs = np.stack([proj.adjoint(yi).reshape(256, 256) for yi in ys]).astype(np.float32)
                return torch.from_numpy(xs)[:, None].to(y.device)
        op = Op()
        Acg = lambda x: op.AT(op.A(x))
        ones = torch.ones(1, 1, 256, 256, device=device)
        A_ones = op.A(ones)
        skip = 1000 // args.T_sampling
        times = list(range(0, 1000, skip)); times_next = [-1] + times[:-1]
        checked = False
        for sid in test:
            f = os.path.join(ALT, f'{sid}.npz')
            if not os.path.exists(f):
                continue
            z = np.load(f)
            x0 = z['x0_pre'].astype(np.float32); vmin, vmax = float(z['vmin']), float(z['vmax'])
            scale = vmax - vmin + 1e-8
            y1 = torch.from_numpy(z['y1B'].astype(np.float32))[None, None].to(device)
            if y1.shape[-2] != NA:
                y1 = y1[..., :NA, :]
            y01 = (y1 - vmin * A_ones) / scale
            if not checked:
                yf = torch.from_numpy(z['y_full'].astype(np.float32))[None, None].to(device)[..., :NA, :]
                Ax0 = op.A(torch.from_numpy(x0)[None, None].to(device)) * scale + vmin * A_ones
                rel = float((Ax0 - yf).norm() / yf.norm())
                print(f"[dds] operator check: ||A x0 - y_full||/||y_full|| = {rel:.3f} (LSMR-50 image; expect ~0.05-0.2)", flush=True)
                checked = True
            y = 2.0 * y01 - A_ones
            bcg = op.AT(y)
            x = torch.randn(1, 1, 256, 256, device=device)
            with torch.no_grad():
                for i, j in zip(reversed(times), reversed(times_next)):
                    t = torch.full((1,), float(i), device=device); nt = torch.full((1,), float(j), device=device)
                    at = compute_alpha(betas, t.long()); at_next = compute_alpha(betas, nt.long())
                    et = model(x, t); et = et[:, :et.size(1) // 2]
                    x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
                    x0_hat = CG(Acg, bcg, x0_t, n_inner=args.CG_iter)
                    c2 = (1 - at_next).sqrt() * ((1 - args.eta ** 2) ** 0.5)
                    c1 = (1 - at_next).sqrt() * args.eta
                    x = at_next.sqrt() * x0_hat + c1 * torch.randn_like(x0_hat) + c2 * et if j != 0 else x0_hat
            pred = ((x + 1) / 2).clamp(0, 1).cpu().numpy()[0, 0]
            record('dds', 'B_step4', pred, x0, sid)
        e = per['dds']['B_step4']
        print(f"[dds] B: {np.mean(e['psnr']):.2f}±{np.std(e['psnr']):.2f} dB, SSIM {np.mean(e['ssim']):.3f}", flush=True)

    out = {'models': {}, 'knot_t': {}}
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
