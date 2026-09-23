"""Renders the reconstruction comparison figures."""
import os, sys, json, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, binary_fill_holes, label as cc_label
sys.path.insert(0, f"{CODE_ROOT}")
from unified_eval import compute_psnr, compute_ssim

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B = f"{DATA_ROOT}/dose_bridge_baselines"
TAG = os.environ.get('BASE_TAG', 'ep100')
CYAN, MAG = (0, 229, 255), (255, 0, 255)
CFG = {
    'ldct': dict(pre=f"{DATA_ROOT}/LDCT_dose_bridge/precomputed_equal_improvement", hu=(-1090.1, 1264.6),
                 win_main='hu:-160,240', win_cyan='hu:-160,240', win_mag='hu:-60,160', label='Mayo LDCT, 10% dose',
                 methods=[('Input', None, None),
                          ('BM3D', f'{B}/eval_sim_metrics_ldct/preds_bm3d.npz', 'bm3d'),
                          ('DDIM prior', f'{B}/eval_sim_metrics_ldct/preds_ddim.npz', 'ddim_denoise'),
                          ('WGAN-VGG', f'{B}/eval_sweep_ldct/preds/wganvgg_endpoint_equal_improvement_pub.npz', 'wganvgg_endpoint_equal_improvement_pub'),
                          ('MAP-NN', f'{B}/eval_sweep_ldct/preds/mapnn_endpoint_equal_improvement_{TAG}.npz', f'mapnn_endpoint_equal_improvement_{TAG}'),
                          ('EDCNN', f'{B}/eval_sweep_ldct/preds/edcnn_endpoint_equal_improvement_{TAG}.npz', f'edcnn_endpoint_equal_improvement_{TAG}'),
                          ('DnCNN', f'{B}/eval_sweep_ldct/preds/dncnn_endpoint_equal_improvement_{TAG}.npz', f'dncnn_endpoint_equal_improvement_{TAG}'),
                          ('Reference', None, None),
                          ('NAFNet', f'{B}/eval_sweep_ldct/preds/nafnet_endpoint_equal_improvement_{TAG}.npz', f'nafnet_endpoint_equal_improvement_{TAG}@t1'),
                          ('Restormer', f'{B}/eval_sweep_ldct/preds/restormer_endpoint_equal_improvement_{TAG}.npz', f'restormer_endpoint_equal_improvement_{TAG}'),
                          ('U-Net', f'{B}/eval_sweep_ldct/preds/unet_res_endpoint_equal_improvement_{TAG}.npz', f'unet_res_endpoint_equal_improvement_{TAG}@t1'),
                          ('RED-CNN', f'{B}/eval_sweep_ldct/preds/redcnn_endpoint_equal_improvement_{TAG}.npz', f'redcnn_endpoint_equal_improvement_{TAG}'),
                          ('DRUNet', f'{B}/eval_sweep_ldct/preds/drunet_endpoint_equal_improvement_{TAG}.npz', f'drunet_endpoint_equal_improvement_{TAG}@t1'),
                          ('CoreDiff', f'{B}/eval_sweep_ldct_v2/preds/coredif_aug.npz', 'coredif_aug'),
                          ('Ours', f'{B}/eval_sweep_ldct_v5/preds/ourshyb4_samplers.npz', 'ourshyb4_pmavg_EI5'),
                          ('Reference', None, None)]),
    '2detect': dict(pre=f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed_equal_improvement", hu=None,
                    win_main='p1,p97', win_cyan='p1,p97', win_mag='p5,p90', label='2DeteCT, 1% dose',
                    methods=[('Input', None, None),
                             ('BM3D', f'{B}/eval_sim_metrics_2detect/preds_bm3d.npz', 'bm3d'),
                             ('DDIM prior', f'{B}/eval_sim_metrics_2detect/preds_ddim.npz', 'ddim_denoise'),
                             ('MAP-NN', f'{B}/eval_sweep_2detect/preds/mapnn_endpoint_uniform_{TAG}.npz', f'mapnn_endpoint_uniform_{TAG}'),
                             ('WGAN-VGG', f'{B}/eval_sweep_2detect/preds/wganvgg_endpoint_uniform_pub.npz', 'wganvgg_endpoint_uniform_pub'),
                             ('NAFNet', f'{B}/eval_sweep_2detect/preds/nafnet_endpoint_uniform_{TAG}.npz', f'nafnet_endpoint_uniform_{TAG}@t1'),
                             ('Restormer', f'{B}/eval_sweep_2detect/preds/restormer_endpoint_uniform_{TAG}.npz', f'restormer_endpoint_uniform_{TAG}'),
                             ('Reference', None, None),
                             ('DnCNN', f'{B}/eval_sweep_2detect/preds/dncnn_endpoint_uniform_{TAG}.npz', f'dncnn_endpoint_uniform_{TAG}'),
                             ('EDCNN', f'{B}/eval_sweep_2detect/preds/edcnn_endpoint_uniform_{TAG}.npz', f'edcnn_endpoint_uniform_{TAG}'),
                             ('DRUNet', f'{B}/eval_sweep_2detect/preds/drunet_endpoint_uniform_{TAG}.npz', f'drunet_endpoint_uniform_{TAG}@t1'),
                             ('RED-CNN', f'{B}/eval_sweep_2detect/preds/redcnn_endpoint_uniform_{TAG}.npz', f'redcnn_endpoint_uniform_{TAG}'),
                             ('U-Net', f'{B}/eval_sweep_2detect/preds/unet_res_endpoint_uniform_{TAG}.npz', f'unet_res_endpoint_uniform_{TAG}@t1'),
                             ('CoreDiff', f'{B}/eval_sweep_2detect_v2/preds/coredif_aug.npz', 'coredif_aug'),
                             ('Ours', f'{B}/eval_sweep_2detect_v5/preds/ourshyb96_samplers.npz', 'ourshyb96_pmavg_EI5'),
                             ('Reference', None, None)]),
}
ROI_H, ROI_W, ZOOM, FRAME = 20, 40, 3, 4


class Preds:
    def __init__(self, path, name):
        self.z = np.load(path); self.files = set(self.z.files); self.name = name
    def sids(self):
        return sorted({f.split('|')[-1] for f in self.files if '|EI_step4|' in f and f.startswith(self.name)})
    def get(self, sid):
        k = f'{self.name}|EI_step4|{sid}'
        if k not in self.files:
            c = sorted(f for f in self.files if f.endswith(f'|EI_step4|{sid}') and f.startswith(self.name))
            if not c:
                raise KeyError(f'{k} not in {self.z}')
            k = c[0]
        return self.z[k].astype(np.float32)


def parse_window(spec, x0, hu):
    if spec.startswith('hu:'):
        lo, hi = [float(v) for v in spec[3:].split(',')]
        return (lo - hu[0]) / (hu[1] - hu[0]), (hi - hu[0]) / (hu[1] - hu[0])
    a, b = spec.split(',')
    return float(np.percentile(x0, float(a[1:]))), float(np.percentile(x0, float(b[1:])))


def to_u8(x, w):
    return (np.clip((x - w[0]) / (w[1] - w[0]), 0, 1) * 255).astype(np.uint8)


def body_mask(x0, cfg):
    """'valid' pixels for ROI placement: soft tissue only (no air/lung/bone on LDCT; no background / saturated
    objects on 2DeteCT), so the crops show texture where the methods actually differ."""
    if cfg['hu']:
        hu = (x0 * (cfg['hu'][1] - cfg['hu'][0]) + cfg['hu'][0])
        return (hu > -150) & (hu < 250)
    p1, p97 = np.percentile(x0, [1, 97]); xw = (x0 - p1) / (p97 - p1)
    return (xw > 0.10) & (xw < 0.85)


def best_window(score, mask, h, w, margin=6, exclude=None, min_frac=0.97):
    H, W = score.shape
    mf = uniform_filter(mask.astype(np.float32), size=(h, w), mode='constant')
    sc = uniform_filter(score.astype(np.float32), size=(h, w), mode='constant')
    best, arg = -np.inf, None
    for frac in (min_frac, 0.9, 0.8, 0.6, 0.0):
        for r in range(margin + h // 2, H - margin - h // 2, 2):
            for c in range(margin + w // 2, W - margin - w // 2, 2):
                if mf[r, c] < frac:
                    continue
                if exclude is not None and abs(r - exclude[0]) < h + 6 and abs(c - exclude[1]) < w + 6:
                    continue
                if sc[r, c] > best:
                    best, arg = sc[r, c], (r - h // 2, c - w // 2)
        if arg is not None:
            break
    return arg


def draw_box(rgb, r, c, h, w, col, t=2):
    r0, c0, r1, c1 = max(r - t, 0), max(c - t, 0), min(r + h + t, rgb.shape[0]), min(c + w + t, rgb.shape[1])
    rgb[r0:r0 + t, c0:c1] = col; rgb[r1 - t:r1, c0:c1] = col; rgb[r0:r1, c0:c0 + t] = col; rgb[r0:r1, c1 - t:c1] = col


def crop_zoom(x, w, roi, col):
    r, c = roi
    p = to_u8(x[r:r + ROI_H, c:c + ROI_W], w)
    p = np.array(Image.fromarray(p).resize((ROI_W * ZOOM, ROI_H * ZOOM), Image.LANCZOS))
    rgb = np.stack([p] * 3, -1)
    out = np.zeros((rgb.shape[0] + 2 * FRAME, rgb.shape[1] + 2 * FRAME, 3), np.uint8); out[...] = col
    out[FRAME:-FRAME, FRAME:-FRAME] = rgb
    return out


def crop_bbox(x0, cfg, pad=3):
    """Common per-slice bounding box (r0, r1, c0, c1) of the body, computed ONCE from the pipeline reference x0:
    LDCT: HU > -500 (cfg['hu'] scaling); 2DeteCT: pixels above 10% of the p1-p97 window.  Then binary_fill_holes,
    largest connected component, bbox + pad px (clipped to the image)."""
    if cfg['hu']:
        hu = x0 * (cfg['hu'][1] - cfg['hu'][0]) + cfg['hu'][0]; m = hu > -500
    else:
        p1, p97 = np.percentile(x0, [1, 97]); m = (x0 - p1) / (p97 - p1) > 0.10
    m = binary_fill_holes(m)
    lab, n = cc_label(m)
    if n > 1:
        m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    rr, cc = np.where(m)
    H, W = x0.shape
    return (max(int(rr.min()) - pad, 0), min(int(rr.max()) + 1 + pad, H), max(int(cc.min()) - pad, 0), min(int(cc.max()) + 1 + pad, W))


def cell(x, wm, wc, wg, rc, rm, bbox=None):
    """bbox=None: original layout (full 256x256 main + 256-wide crop strip).  bbox=(r0,r1,c0,c1): the main image is
    cropped to the box (ROI boxes shifted into it); the zoomed crops are cut from the UNCROPPED x exactly as before and
    the strip is LANCZOS-resized to the cropped main width so the cell stays a rectangle."""
    strip = np.concatenate([crop_zoom(x, wg, rm, MAG), crop_zoom(x, wc, rc, CYAN)], axis=1)
    if bbox is None:
        rgb = np.stack([to_u8(x, wm)] * 3, -1).copy()
        draw_box(rgb, *rc, ROI_H, ROI_W, CYAN); draw_box(rgb, *rm, ROI_H, ROI_W, MAG)
        return np.concatenate([rgb, strip], axis=0)
    r0, r1, c0, c1 = bbox
    rgb = np.stack([to_u8(x[r0:r1, c0:c1], wm)] * 3, -1).copy()
    draw_box(rgb, rc[0] - r0, rc[1] - c0, ROI_H, ROI_W, CYAN); draw_box(rgb, rm[0] - r0, rm[1] - c0, ROI_H, ROI_W, MAG)
    w = rgb.shape[1]; h = int(round(strip.shape[0] * w / strip.shape[1]))
    strip = np.array(Image.fromarray(strip).resize((w, h), Image.LANCZOS))
    return np.concatenate([rgb, strip], axis=0)


def make_row(ds, sid, args, out_dir):
    cfg = CFG[ds]
    d = np.load(f"{cfg['pre']}/{sid}_step4.npz"); x0 = d['x0'].astype(np.float32); xt = d['x_t'].astype(np.float32)
    imgs, names = [], []
    last_ref = max(i for i, m in enumerate(cfg['methods']) if m[0] == 'Reference')   # index of the final Reference entry
    for mi, (name, path, key) in enumerate(cfg['methods']):
        if args.methods and name not in args.methods.split(',') and name not in ('Input', 'Reference'):
            continue
        if args.single_ref and name == 'Reference' and mi != last_ref:   # --single_ref: drop the mid-row Reference
            continue
        if name == 'Input':
            imgs.append(xt)
        elif name == 'Reference':
            if args.ref_override:
                path, key = args.ref_override.split(':'); hu = np.load(path)[key].astype(np.float32)
                if float(hu.max()) <= 1.5:
                    imgs.append(np.clip(hu, 0, 1).astype(np.float32))
                else:
                    vmin, vmax = float(d['vmin']), float(d['vmax']); imgs.append(np.clip((hu - vmin) / (vmax - vmin + 1e-8), 0, 1).astype(np.float32))
            else:
                imgs.append(x0)
        else:
            imgs.append(np.clip(Preds(path, key).get(sid), 0, 1))
        names.append(name)
    wm = parse_window(args.win_main or cfg['win_main'], x0, cfg['hu'])
    wc = parse_window(args.win_cyan or cfg['win_cyan'], x0, cfg['hu'])
    wg = parse_window(args.win_mag or cfg['win_mag'], x0, cfg['hu'])
    mask = body_mask(x0, cfg)
    xw = np.clip((x0 - wm[0]) / (wm[1] - wm[0]), 0, 1)
    std = np.sqrt(np.maximum(uniform_filter(xw ** 2, 9) - uniform_filter(xw, 9) ** 2, 0))
    rc = tuple(int(v) for v in args.roi_cyan.split(',')) if args.roi_cyan else best_window(std, mask, ROI_H, ROI_W)
    base_names = [n for n in names if n not in ('Input', 'Reference', 'Ours')]
    best_base = max(base_names, key=lambda n: compute_psnr(imgs[names.index(n)], x0))
    diff = np.abs(imgs[names.index(best_base)] - imgs[names.index('Ours')])
    rm = tuple(int(v) for v in args.roi_mag.split(',')) if args.roi_mag else best_window(diff, mask, ROI_H, ROI_W, exclude=(rc[0] + ROI_H // 2, rc[1] + ROI_W // 2))
    bbox = crop_bbox(x0, cfg) if args.crop else None
    cells = [cell(x, wm, wc, wg, rc, rm, bbox) for x in imgs]
    metrics = [(compute_psnr(x, x0), compute_ssim(x, x0)) if n != 'Reference' else None for x, n in zip(imgs, names)]
    for n in ('Ours', best_base):
        pass
    def roi_psnr(x, roi):
        r, c = roi; return compute_psnr(x[r:r + ROI_H, c:c + ROI_W], x0[r:r + ROI_H, c:c + ROI_W])
    vis = {n: (roi_psnr(x, rm), roi_psnr(x, rc)) for x, n in zip(imgs, names) if n != 'Reference'}
    vis['_best_base'] = best_base
    ncol = args.ncol; ch, cw = cells[0].shape[:2]; nrow = int(np.ceil(len(cells) / ncol))
    gap = 0.02; top = args.top; bot = args.bot
    cell_w = args.cell_w if args.cell_w > 0 else (7.0 - (ncol - 1) * gap) / ncol
    cell_h = cell_w * ch / cw
    W = ncol * cell_w + (ncol - 1) * gap; H = nrow * (top + cell_h + bot)
    if args.max_h > 0 and H > args.max_h:
        s = max((args.max_h / nrow - cell_h) / (top + bot), 0.0); top, bot = max(top * s, 0.10), max(bot * s, 0.11)
        H = nrow * (top + cell_h + bot)
        print(f'{sid}: figure taller than --max_h {args.max_h}: bands top/bot -> {top:.3f}/{bot:.3f} in (H {H:.3f} in)', flush=True)
    print(f'{sid}: crop bbox {bbox} cell {cw}x{ch} px, cell_w {cell_w:.4f} in, cell_h {cell_h:.4f} in, figsize {W:.3f} x {H:.3f} in'
          + (f' (uncropped-equivalent cell_h {cell_w * 324 / 256:.3f} in, {nrow * (args.top + cell_w * 324 / 256 + args.bot):.3f} in tall)' if bbox else ''), flush=True)
    fig = plt.figure(figsize=(W, H), dpi=args.dpi); li = 0
    for i, (im, name, m) in enumerate(zip(cells, names, metrics)):
        r, c = divmod(i, ncol)
        x0f = (c * (cell_w + gap)) / W; y0f = (bot + (nrow - 1 - r) * (top + cell_h + bot)) / H
        ax = fig.add_axes([x0f, y0f, cell_w / W, cell_h / H]); ax.imshow(im, interpolation='nearest'); ax.axis('off')
        if m is not None:
            fig.text(x0f + cell_w / W / 2, y0f + cell_h / H + 0.35 * top / H, f'{m[0]:.2f} dB / {m[1]:.3f}', ha='center', va='center', fontsize=args.fs_metric, family='monospace')
        lab = name if name == 'Reference' else f'({chr(97 + li)}) {name}'
        if name != 'Reference':
            li += 1
        fig.text(x0f + cell_w / W / 2, y0f - 0.45 * bot / H, lab, ha='center', va='center', fontsize=args.fs_label, fontweight='bold')
    base = os.path.join(out_dir, f'fig_recon_rdm_{ds}_{sid}{args.out_suffix}')
    fig.savefig(base + '.png', dpi=args.dpi, pad_inches=0.01, bbox_inches='tight'); fig.savefig(base + '.pdf', dpi=args.dpi, pad_inches=0.01, bbox_inches='tight'); plt.close(fig)
    info = {'sid': sid, 'roi_cyan': rc, 'roi_mag': rm, 'windows': {'main': wm, 'cyan': wc, 'mag': wg},
            'metrics': {n: m for n, m in zip(names, metrics) if m}, 'roi_psnr_mag_cyan': vis}
    if bbox is not None:
        info['crop_bbox_r0_r1_c0_c1'] = bbox; info['layout'] = {'cell_px': [cw, ch], 'cell_w_in': cell_w, 'cell_h_in': cell_h, 'top': top, 'bot': bot, 'figsize_in': [W, H]}
    json.dump(info, open(base + '.json', 'w'), indent=1, default=float)
    print(f'{sid}: cyan {rc} mag {rm} | ' + ' '.join(f'{n}={m[0]:.2f}' for n, m in zip(names, metrics) if m)
          + f" | best baseline {best_base}; ROI-mag PSNR {vis[best_base][0]:.2f} vs Ours {vis['Ours'][0]:.2f} (gap {vis['Ours'][0]-vis[best_base][0]:+.2f}); ROI-cyan gap {vis['Ours'][1]-vis[best_base][1]:+.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(CFG))
    ap.add_argument('--sids', required=True, help='comma list, or auto:N (evenly spaced test sids)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--win_main', default=''); ap.add_argument('--win_cyan', default=''); ap.add_argument('--win_mag', default='')
    ap.add_argument('--roi_cyan', default='', help='r,c top-left (single sid only)'); ap.add_argument('--roi_mag', default='')
    ap.add_argument('--cell_w', type=float, default=0.0, help='cell width in inches; 0 (default) = auto so that ncol columns fill 7.0 in, i.e. (7.0 - (ncol-1)*0.02)/ncol (before 2026-09-14 the default was 1.0)')
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--crop', action='store_true', help='crop the air around the body in every main image (one bbox per slice from x0); ROIs/crops/metrics unchanged')
    ap.add_argument('--top', type=float, default=0.13, help='metric band height above each row (in)'); ap.add_argument('--bot', type=float, default=0.15, help='label band height below each row (in)')
    ap.add_argument('--max_h', type=float, default=0.0, help='if > 0 and the figure would be taller, shrink --top/--bot (floors 0.10/0.11 in), never the images')
    ap.add_argument('--fs_metric', type=float, default=5.0); ap.add_argument('--fs_label', type=float, default=6.0)
    ap.add_argument('--out_suffix', default='')
    ap.add_argument('--ncol', type=int, default=7)
    ap.add_argument('--methods', default='', help='comma list of method names to keep (Input/Reference always kept)')
    ap.add_argument('--single_ref', action='store_true', help='keep only the FINAL Reference entry of the CFG list (drop the mid-row one), so a --methods subset can form one 8-cell row; default off = unchanged 2-row layout')
    ap.add_argument('--sheet', default='', help='also write a contact sheet (png) stacking all rows with sid labels')
    ap.add_argument('--ref_override', default='', help='npz:key of an alternative reference image (HU, 256x256) shown in the Reference column ONLY; metrics stay against the pipeline reference')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = CFG[args.dataset]
    if args.sids.startswith('auto:'):
        n = int(args.sids[5:]); _, path, key = cfg['methods'][1]
        all_s = Preds(path, key).sids(); sids = [all_s[int(i)] for i in np.linspace(0, len(all_s) - 1, n).round()]
    else:
        sids = args.sids.split(',')
    print('sids:', sids, flush=True)
    for sid in sids:
        make_row(args.dataset, sid, args, args.out_dir)
    if args.sheet:
        from PIL import ImageDraw
        ims = [Image.open(os.path.join(args.out_dir, f'fig_recon_rdm_{args.dataset}_{sid}{args.out_suffix}.png')).convert('RGB') for sid in sids]
        w = max(im.width for im in ims); pad = 40
        sheet = Image.new('RGB', (w, sum(im.height + pad for im in ims)), 'white'); y = 0; dr = ImageDraw.Draw(sheet)
        for sid, im in zip(sids, ims):
            dr.text((10, y + 8), f'{sid}', fill=(200, 0, 0)); sheet.paste(im, (0, y + pad)); y += im.height + pad
        sheet.save(args.sheet); print('sheet', args.sheet)


if __name__ == '__main__':
    main()
