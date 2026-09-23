"""Plots PSNR against dose for every method."""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"
ALPHA = {'2detect': 0.01, 'ldct': 0.1}
C = {'bridge_unet': '#2a78d6', 'bridge_redcnn': '#eb6834', 'ep_redcnn': '#1baf7a', 'ep_unet': '#eda100',
     'spec': '#0b0b0b', 'grid': '#e6e5e1', 'text': '#0b0b0b', 'muted': '#52514e'}

SERIES = [
    (((os.environ['OURS2D_KEY'] + '|') if os.environ.get('OURS2D_KEY') else '') + 'rep4_unet_res96_bridge_uniform_aug_ema_lr0.0002v2@t|rep4_unet_res_bridge_uniform_aug_ema_lr0.0002v2@t|unet_res96_bridge_uniform_aug_ema_lr0.0002@t|unet_res_bridge_uniform_aug_ema@t|unet_res_bridge_uniform@t',
     'Ours: one bridge checkpoint', C['bridge_unet'], '-', 'o'),
    ('redcnn_wide_endpoint_uniform_aug_ema|redcnn_wide_endpoint_uniform', 'RED-CNN-wide, single dose', C['bridge_redcnn'], '--', 's'),
    ('redcnn_endpoint_uniform_aug_ema|redcnn_endpoint_uniform_alt|redcnn_endpoint_uniform', 'RED-CNN, single dose', C['ep_redcnn'], '--', '^'),
    ('unet_res_endpoint_uniform_aug_ema@t1_alt|unet_res_endpoint_uniform_aug_ema@t1|unet_res_endpoint_uniform@t1', 'U-Net, single dose', C['ep_unet'], '--', 'v'),
    ('bm3d', 'BM3D (dose-oracle $\\sigma$)', C['muted'], ':', 'd'),
]
SERIES_LDCT = [
    (((os.environ['OURSLD_KEY'] + '|') if os.environ.get('OURSLD_KEY') else '') + 'unet_res_bridge_uniform_aug_ema_lr0.0002@t|unet_res_bridge_uniform_aug_ema@t|unet_res_bridge_uniform@t',
     'Ours: one bridge checkpoint', C['bridge_unet'], '-', 'o'),
    ('redcnn_wide_endpoint_equal_improvement_aug_ema|redcnn_wide_endpoint_equal_improvement', 'RED-CNN-wide, single dose', C['bridge_redcnn'], '--', 's'),
    ('redcnn_endpoint_equal_improvement_aug_ema|redcnn_endpoint_equal_improvement_alt|endpoint_redcnn', 'RED-CNN, single dose', C['ep_redcnn'], '--', '^'),
    ('unet_res_endpoint_uniform_aug_ema@t1|unet_res_endpoint_uniform@t1', 'U-Net, single dose', C['ep_unet'], '--', 'v'),
    ('bm3d', 'BM3D (dose-oracle $\\sigma$)', C['muted'], ':', 'd'),
]

if os.environ.get('BASE_PUB') == '1':
    def _pub(series, ds):
        out = []
        for key, lab, col, ls, mk in series:
            if 'RED-CNN-wide' in lab:
                continue
            tag = os.environ.get('BASE_TAG', '')
            sched = 'uniform' if ds == '2detect' else 'equal_improvement'
            if lab.startswith('RED-CNN'):
                key = (f'redcnn_endpoint_{sched}_{tag}' if tag else
                       'redcnn_endpoint_uniform' if ds == '2detect' else 'redcnn_endpoint_equal_improvement|redcnn_endpoint_equal_improvement_alt')
            elif lab.startswith('U-Net'):
                key = f'unet_res_endpoint_{sched}_{tag}@t1' if tag else 'unet_res_endpoint_uniform@t1'
            out.append((key, lab, col, ls, mk))
        return out
    SERIES = _pub(SERIES, '2detect'); SERIES_LDCT = _pub(SERIES_LDCT, 'ldct')


def load(ds, path):
    s = json.load(open(path))
    from glob import glob
    for ef in sorted(glob(os.path.join(os.path.dirname(path), 'extra_*.json'))):
        E = json.load(open(ef))
        for m, byk in E['models'].items():
            s['models'].setdefault(m, {}).update(byk)
    knots = sorted(s['knot_t'].items(), key=lambda kv: kv[1])
    return s, knots


def panel(ax, ds, s, knots, series, spec_prefix, title, family='uniform'):
    a = ALPHA[ds]
    sched = {'uniform': 'uniform', 'EI': 'equal_improvement'}[family]
    knots = [(k, t) for k, t in knots if k.startswith(family + '_step') or (family == 'uniform' and k.startswith('offgrid_step'))]
    knots = sorted(knots, key=lambda kt: kt[1])
    tvals = np.array([t for _, t in knots])
    dose = 100 * a ** tvals
    inp = np.array([s['models']['input'][k]['psnr'] for k, _ in knots])
    ax.axhline(0, color=C['muted'], lw=1, ls=':', zorder=1)
    ax.text(dose.max() * 0.55, -0.25, 'input', color=C['muted'], fontsize=7, ha='center', va='top')
    for key, label, col, ls, mk in series:
        key = next((k for k in key.split('|') if k in s['models']), None)
        if key is None:
            continue
        y = np.array([s['models'][key][k]['psnr'] if k in s['models'][key] else np.nan for k, _ in knots]) - inp
        ax.plot(dose, y, ls, color=col, lw=1.6, marker=mk, ms=4, mfc='white', mew=1.2, label=label, zorder=3)
    sx, sy = [], []
    for k, t in knots:
        if not k.startswith(family + '_step'):
            continue
        step = k.split('_step')[1]
        cand = [m for m in s['models'] if (f'_spec_{sched}_k{step}' in m or f'_endpoint_{sched}_k{step}' in m)
                and (os.environ.get('BASE_PUB') != '1' or 'aug' not in m)
                and '@t1' not in m and k in s['models'][m]]
        if not cand:
            continue
        m = max(cand, key=lambda mm: s['models'][mm][k]['psnr'])
        if k in s['models'][m]:
            sx.append(100 * a ** t); sy.append(s['models'][m][k]['psnr'] - s['models']['input'][k]['psnr'])
    fam_knots = [(k, t) for k, t in knots if k.startswith(family + '_step')]
    kmin = fam_knots[-1][0]
    cands = [next((k for k in ser[0].split('|') if k in s['models']), None) for ser in series[1:4]]
    cands = [c for c in cands if c and kmin in s['models'][c]]
    if cands:
        ep = max(cands, key=lambda m: s['models'][m][kmin]['psnr'])
        sx.append(100 * a ** fam_knots[-1][1]); sy.append(s['models'][ep][kmin]['psnr'] - s['models']['input'][kmin]['psnr'])
    if any(k.startswith('offgrid_step') for k, _ in knots):
        ax.axvspan(100 * a ** 1.3, 100 * a ** 1.0, color=C['grid'], alpha=0.6, lw=0, zorder=0)
        ax.text(100 * a ** 1.12, 0.6, 'below\ntraining\nrange', fontsize=6, color=C['muted'], ha='center', va='bottom')
    if sx:
        ax.scatter(sx, sy, s=34, facecolors='none', edgecolors=C['spec'], lw=1.2, zorder=4,
                   label='Per-dose specialist network (one model per dose)')
    ax.set_xscale('log')
    ax.set_xlabel('dose (% of full dose)')
    ax.set_ylabel(r'$\Delta$PSNR = output $-$ input (dB)')
    ax.set_title(title, fontsize=9, loc='left')
    ticks = sorted(set(np.round(dose, 1)))
    if len(ticks) > 7:   # thin the tick labels when off-grid doses are present
        ticks = [d for d in ticks if d in (0.3, 1.0, 2.5, 6.3, 15.8, 39.8)] or ticks[::2]
    ax.set_xticks(ticks); ax.set_xticklabels([f'{d:g}' for d in ticks], fontsize=7)
    ax.minorticks_off()
    ax.grid(True, axis='y', color=C['grid'], lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep2d', default=f'{OUT_ROOT}/eval_sweep_2detect_v1/summary.json')
    ap.add_argument('--sweepld', default=f'{OUT_ROOT}/eval_sweep_ldct_v1/summary.json')
    ap.add_argument('--out', default=f"{CODE_ROOT}/ICASSP2027/figs/dose_sweep")
    args = ap.parse_args()
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5})
    have_ld = os.path.exists(args.sweepld)
    fig, axes = plt.subplots(1, 2 if have_ld else 1, figsize=(7.0 if have_ld else 3.5, 2.3), squeeze=False)
    s, knots = load('2detect', args.sweep2d)
    panel(axes[0, 0], '2detect', s, knots, SERIES, 'redcnn_spec_', r'2DeteCT (trained for $100\times$: 1% dose)', family='uniform')
    if have_ld:
        s2, knots2 = load('ldct', args.sweepld)
        panel(axes[0, 1], 'ldct', s2, knots2, SERIES_LDCT, 'redcnn_spec_', r'Mayo LDCT (trained for $10\times$: 10% dose)', family='EI')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3 if have_ld else 2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12 if have_ld else 0.2, 1, 1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + '.pdf', bbox_inches='tight')
    fig.savefig(args.out + '_preview.png', dpi=200, bbox_inches='tight')
    print('written', args.out + '.pdf')


if __name__ == '__main__':
    main()
