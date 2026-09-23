"""Blank-scan intensity and Fano factor calibration from raw data."""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DETECT2_PATH = f"{DATA_ROOT}/2DeteCT"
LDCT_PATH = f"{DATA_ROOT}/LDCT_recon"


# 2DeteCT: Direct flat/dark measurement

def load_2detect_tif(path):
    """Load a 2DeteCT TIF file as float32 array."""
    return np.array(Image.open(path), dtype=np.float32)


def estimate_2detect_I0(sino_path, sample_slices, modes=(1, 2)):
    """Estimate I0 from flat-dark for each mode.

    Args:
        sino_path: base path to 2DeteCT data
        sample_slices: list of slice indices to average over
        modes: tuple of modes (1=low-dose, 2=full-dose)

    Returns:
        dict with per-mode results: I0_profile (956,), I0_mean, I0_std,
        and per-slice values for consistency check.
    """
    results = {}

    for mode in modes:
        I0_profiles = []

        for sl in sample_slices:
            base = os.path.join(sino_path, f'slice{sl:05d}', f'mode{mode}')
            dark = load_2detect_tif(os.path.join(base, 'dark.tif'))
            flat1 = load_2detect_tif(os.path.join(base, 'flat1.tif'))
            flat2 = load_2detect_tif(os.path.join(base, 'flat2.tif'))

            # Average flat fields
            flat = (flat1 + flat2) / 2.0

            dark_1d = dark[0] if dark.ndim == 2 else dark
            flat_1d = flat[0] if flat.ndim == 2 else flat

            I0_raw = flat_1d - dark_1d

            I0_binned = I0_raw[0::2] + I0_raw[1::2]

            I0_profiles.append(I0_binned)

        I0_profiles = np.array(I0_profiles)
        I0_mean_profile = np.mean(I0_profiles, axis=0)
        I0_std_profile = np.std(I0_profiles, axis=0)

        per_slice_means = np.mean(I0_profiles, axis=1)

        results[f'mode{mode}'] = {
            'I0_profile': I0_mean_profile,
            'I0_std_profile': I0_std_profile,
            'I0_mean': float(np.mean(I0_mean_profile)),
            'I0_median': float(np.median(I0_mean_profile)),
            'I0_min': float(np.min(I0_mean_profile)),
            'I0_max': float(np.max(I0_mean_profile)),
            'per_slice_means': per_slice_means.tolist(),
            'per_slice_std': float(np.std(per_slice_means)),
        }

    if 'mode1' in results and 'mode2' in results:
        results['dose_ratio_mode1_mode2'] = results['mode1']['I0_mean'] / results['mode2']['I0_mean']

    return results


def validate_2detect_fano(sino_path, slice_idx, mode, I0_profile):
    """Validate Poisson statistics via Fano factor in air region.

    Uses adjacent-angle differencing on sinogram air pixels (low attenuation)
    to estimate Var/Mean. For ideal photon counting: Fano ≈ 1.

    Args:
        sino_path: base path
        slice_idx: which slice to analyze
        mode: 1 or 2
        I0_profile: (956,) I0 profile for normalization

    Returns:
        dict with fano_mean, fano_profile, n_air_pixels
    """
    base = os.path.join(sino_path, f'slice{slice_idx:05d}', f'mode{mode}')
    sino = load_2detect_tif(os.path.join(base, 'sinogram.tif'))
    dark = load_2detect_tif(os.path.join(base, 'dark.tif'))
    flat1 = load_2detect_tif(os.path.join(base, 'flat1.tif'))
    flat2 = load_2detect_tif(os.path.join(base, 'flat2.tif'))
    flat = (flat1 + flat2) / 2.0

    sino = sino[:, 0::2] + sino[:, 1::2]
    dark_1d = dark[0, 0::2] + dark[0, 1::2]
    flat_1d = flat[0, 0::2] + flat[0, 1::2]

    norm = (sino - dark_1d) / (flat_1d - dark_1d + 1e-10)
    norm = norm[:-1, :]

    mean_transmission = np.mean(norm, axis=0)
    air_mask = mean_transmission > 0.95
    n_air = int(np.sum(air_mask))

    if n_air < 10:
        air_mask = mean_transmission > 0.90
        n_air = int(np.sum(air_mask))

    if n_air < 5:
        return {'fano_mean': float('nan'), 'n_air_pixels': 0,
                'warning': 'Too few air pixels found'}

    raw_counts = sino[:, air_mask] - dark_1d[air_mask]

    diff = raw_counts[1:, :] - raw_counts[:-1, :]
    var_diff = np.var(diff, axis=0) / 2.0
    mean_counts = np.mean(raw_counts, axis=0)

    fano_per_det = var_diff / (mean_counts + 1e-10)

    return {
        'fano_mean': float(np.mean(fano_per_det)),
        'fano_median': float(np.median(fano_per_det)),
        'fano_std': float(np.std(fano_per_det)),
        'fano_min': float(np.min(fano_per_det)),
        'fano_max': float(np.max(fano_per_det)),
        'n_air_pixels': n_air,
        'mean_air_counts': float(np.mean(mean_counts)),
    }


def load_ldct_sinogram(patient_id, dose, recon_dir):
    """Load LDCT 3D sinogram (log-transformed). Returns (n_angles, n_det, n_slices)."""
    d = os.path.join(recon_dir, f'{patient_id}_{dose}_complete')
    tif_path = os.path.join(d, 'scan_001_flat_fan_projections.tif')
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"No sinogram at {tif_path}")

    img = Image.open(tif_path)
    n_frames = img.n_frames
    h, w = img.size

    sino_3d = np.zeros((n_frames, w, h), dtype=np.float32)
    for i in range(n_frames):
        img.seek(i)
        sino_3d[i] = np.array(img, dtype=np.float32)
    img.close()

    return sino_3d


def estimate_ldct_I0(recon_dir, patient_id, n_sample_slices=50):
    """Estimate I0 for LDCT from noise variance in log-sinogram.

    Core formula: Var(y) ≈ exp(y) / I0  →  I0 ≈ exp(y_mean) / Var(y)

    Uses adjacent-slice differencing (skip 1 to avoid rebinning correlation)
    to isolate noise from signal.

    Returns:
        dict with I0 profiles, means, and cross-validation metrics.
    """
    results = {}

    for dose in ['full', 'low']:
        dir_path = os.path.join(recon_dir, f'{patient_id}_{dose}_complete')
        if not os.path.exists(dir_path):
            print(f"  Warning: {dir_path} not found, skipping")
            continue

        print(f"  Loading {patient_id}_{dose} sinogram...")
        sino_3d = load_ldct_sinogram(patient_id, dose, recon_dir)
        n_angles, n_det, n_slices = sino_3d.shape
        print(f"    Shape: {sino_3d.shape} (angles, det, slices)")

        margin = 5
        available = n_slices - 2 * margin
        if n_sample_slices > available // 2:
            n_sample_slices = available // 2
        sample_indices = np.linspace(margin, n_slices - margin - 3,
                                     n_sample_slices, dtype=int)

        var_noise_all = []
        y_mean_all = []

        for s in sample_indices:
            s2 = s + 2
            if s2 >= n_slices:
                continue
            sino_s = sino_3d[:, :, s]
            sino_s2 = sino_3d[:, :, s2]

            diff = sino_s - sino_s2
            var_per_det = np.var(diff, axis=0) / 2.0
            y_mean_per_det = np.mean(sino_s, axis=0)

            var_noise_all.append(var_per_det)
            y_mean_all.append(y_mean_per_det)

        var_noise_all = np.array(var_noise_all)
        y_mean_all = np.array(y_mean_all)

        var_noise_median = np.median(var_noise_all, axis=0)
        y_mean_median = np.median(y_mean_all, axis=0)

        safe_var = np.maximum(var_noise_median, 1e-12)
        I0_profile = np.exp(y_mean_median) / safe_var

        air_threshold = 0.05
        air_mask = y_mean_median < air_threshold
        n_air = int(np.sum(air_mask))

        if n_air > 5:
            I0_air = 1.0 / safe_var[air_mask]
            I0_air_mean = float(np.mean(I0_air))
            I0_air_median = float(np.median(I0_air))
        else:
            I0_air_mean = float('nan')
            I0_air_median = float('nan')

        center_slice = slice(n_det // 4, 3 * n_det // 4)
        I0_center = I0_profile[center_slice]

        results[dose] = {
            'I0_profile': I0_profile,
            'I0_mean': float(np.mean(I0_profile)),
            'I0_median': float(np.median(I0_profile)),
            'I0_center_mean': float(np.mean(I0_center)),
            'I0_center_median': float(np.median(I0_center)),
            'I0_min': float(np.min(I0_profile)),
            'I0_max': float(np.max(I0_profile)),
            'I0_air_mean': I0_air_mean,
            'I0_air_median': I0_air_median,
            'n_air_pixels': n_air,
            'y_mean_profile': y_mean_median,
            'var_noise_profile': var_noise_median,
            'n_angles': n_angles,
            'n_det': n_det,
            'n_slices': n_slices,
            'n_sample_slices': len(sample_indices),
        }

    if 'full' in results and 'low' in results:
        var_ratio = results['low']['var_noise_profile'] / (results['full']['var_noise_profile'] + 1e-12)
        alpha_from_var = 1.0 / np.median(var_ratio)

        results['dose_ratio'] = {
            'alpha_from_I0': results['low']['I0_median'] / (results['full']['I0_median'] + 1e-10),
            'alpha_from_var_ratio': float(alpha_from_var),
            'var_ratio_median': float(np.median(var_ratio)),
            'I0_ratio_center': results['low']['I0_center_median'] / (results['full']['I0_center_median'] + 1e-10),
        }

    return results


def plot_profiles(detect2_results, ldct_results, output_dir):
    """Create 4-panel summary plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if detect2_results:
        for mode_key in ['mode2', 'mode1']:
            if mode_key not in detect2_results:
                continue
            r = detect2_results[mode_key]
            profile = r['I0_profile']
            label = f'{mode_key} (mean={r["I0_mean"]:.0f})'
            ax.plot(profile, label=label, alpha=0.8)
        ax.set_xlabel('Detector channel (binned)')
        ax.set_ylabel('I0 (photon counts)')
        ax.set_title('2DeteCT: I0 profiles (flat - dark)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No 2DeteCT data', ha='center', va='center',
                transform=ax.transAxes)

    ax = axes[0, 1]
    if detect2_results and 'fano' in detect2_results:
        fano = detect2_results['fano']
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Poisson (Fano=1)')
        for mode_key, fano_data in fano.items():
            if 'fano_mean' in fano_data and not np.isnan(fano_data['fano_mean']):
                ax.bar(mode_key, fano_data['fano_mean'],
                       yerr=fano_data.get('fano_std', 0), capsize=5,
                       alpha=0.7, label=f'{mode_key}: {fano_data["fano_mean"]:.3f}')
        ax.set_ylabel('Fano factor (Var/Mean)')
        ax.set_title('2DeteCT: Fano factor (air region)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Fano data', ha='center', va='center',
                transform=ax.transAxes)

    ax = axes[1, 0]
    if ldct_results:
        for patient_id, patient_data in ldct_results.items():
            if patient_id == 'summary':
                continue
            for dose in ['full', 'low']:
                if dose not in patient_data:
                    continue
                r = patient_data[dose]
                profile = r['I0_profile']
                label = f'{patient_id}_{dose} (med={r["I0_median"]:.0f})'
                ax.plot(profile, label=label, alpha=0.7)
        ax.set_xlabel('Detector channel')
        ax.set_ylabel('I0 (estimated)')
        ax.set_title('LDCT: I0 profiles (exp(y)/Var)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No LDCT data', ha='center', va='center',
                transform=ax.transAxes)

    ax = axes[1, 1]
    labels = []
    alphas = []
    if detect2_results and 'dose_ratio_mode1_mode2' in detect2_results:
        labels.append('2DeteCT\n(mode1/mode2)')
        alphas.append(detect2_results['dose_ratio_mode1_mode2'])
    if ldct_results:
        for patient_id, patient_data in ldct_results.items():
            if patient_id == 'summary':
                continue
            if 'dose_ratio' in patient_data:
                dr = patient_data['dose_ratio']
                labels.append(f'LDCT {patient_id}\n(I0 ratio)')
                alphas.append(dr['alpha_from_I0'])
                labels.append(f'LDCT {patient_id}\n(var ratio)')
                alphas.append(dr['alpha_from_var_ratio'])
    if labels:
        bars = ax.bar(labels, alphas, alpha=0.7, color=['steelblue', 'coral', 'lightsalmon'] * 3)
        for bar, val in zip(bars, alphas):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Dose ratio (alpha)')
        ax.set_title('Dose ratios')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No ratio data', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'photon_count_profiles.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {out_path}")


def print_summary(detect2_results, ldct_results):
    """Print summary table and training parameter recommendations."""
    print("\n" + "=" * 70)
    print("PHOTON COUNT ESTIMATION SUMMARY")
    print("=" * 70)

    if detect2_results:
        print("\n--- 2DeteCT (Dectris EIGER2 photon-counting detector) ---")
        for mode_key in ['mode2', 'mode1']:
            if mode_key not in detect2_results:
                continue
            r = detect2_results[mode_key]
            dose_label = 'full-dose (1000μA)' if mode_key == 'mode2' else 'low-dose (33.3μA)'
            print(f"\n  {mode_key} ({dose_label}):")
            print(f"    I0 mean:   {r['I0_mean']:.1f}")
            print(f"    I0 median: {r['I0_median']:.1f}")
            print(f"    I0 range:  [{r['I0_min']:.1f}, {r['I0_max']:.1f}]")
            print(f"    Slice-to-slice std: {r['per_slice_std']:.1f}")

        if 'dose_ratio_mode1_mode2' in detect2_results:
            ratio = detect2_results['dose_ratio_mode1_mode2']
            expected = 33.3 / 1000.0
            print(f"\n  Dose ratio (mode1/mode2): {ratio:.6f}")
            print(f"  Expected (33.3/1000):     {expected:.6f}")
            print(f"  Relative error:           {abs(ratio - expected) / expected * 100:.2f}%")

        if 'fano' in detect2_results:
            print("\n  Fano factor (Var/Mean in air region):")
            for mode_key, fano_data in detect2_results['fano'].items():
                if 'fano_mean' in fano_data:
                    print(f"    {mode_key}: {fano_data['fano_mean']:.4f} "
                          f"(±{fano_data.get('fano_std', 0):.4f}, "
                          f"n_air={fano_data.get('n_air_pixels', 0)})")
                    if fano_data['fano_mean'] > 1.5:
                        print(f"      ⚠ Super-Poisson! Consider I0_eff = I0 / Fano")

    if ldct_results:
        print("\n--- LDCT (energy-integrating detector) ---")
        for patient_id, patient_data in ldct_results.items():
            if patient_id == 'summary':
                continue
            print(f"\n  Patient {patient_id}:")
            for dose in ['full', 'low']:
                if dose not in patient_data:
                    continue
                r = patient_data[dose]
                print(f"    {dose}-dose:")
                print(f"      I0 median (all):    {r['I0_median']:.0f}")
                print(f"      I0 median (center): {r['I0_center_median']:.0f}")
                print(f"      I0 range:           [{r['I0_min']:.0f}, {r['I0_max']:.0f}]")
                print(f"      I0 air mean:        {r['I0_air_mean']:.0f}" if not np.isnan(r['I0_air_mean']) else "      I0 air: N/A (no air pixels found)")
                print(f"      Sinogram shape:     ({r['n_angles']}, {r['n_det']}, {r['n_slices']})")

            if 'dose_ratio' in patient_data:
                dr = patient_data['dose_ratio']
                print(f"    Dose ratio:")
                print(f"      alpha (from I0 ratio):  {dr['alpha_from_I0']:.4f}")
                print(f"      alpha (from var ratio): {dr['alpha_from_var_ratio']:.4f}")
                print(f"      I0 ratio (center):      {dr['I0_ratio_center']:.4f}")

    print("\n" + "=" * 70)
    print("TRAINING PARAMETER RECOMMENDATIONS")
    print("=" * 70)

    if detect2_results and 'mode2' in detect2_results:
        I0_full = detect2_results['mode2']['I0_median']
        I0_low = detect2_results.get('mode1', {}).get('I0_median', I0_full * 0.0333)
        alpha = I0_low / I0_full if I0_full > 0 else 0.0333
        fano_val = 1.0
        if 'fano' in detect2_results and 'mode2' in detect2_results['fano']:
            fano_val = detect2_results['fano']['mode2'].get('fano_mean', 1.0)
        I0_eff = I0_full / fano_val if fano_val > 1.0 else I0_full

        print(f"\n  2DeteCT:")
        print(f"    --I0_high {I0_eff:.0f}  (measured={I0_full:.0f}" +
              (f", Fano-corrected by /{fano_val:.2f}" if fano_val > 1.0 else "") + ")")
        print(f"    --I0_low  {I0_low:.0f}")
        print(f"    alpha = {alpha:.6f}")
        print(f"    (current training uses: I0_high=1e5, I0_low=1e3, alpha=0.01)")

    if ldct_results:
        for patient_id, patient_data in ldct_results.items():
            if patient_id == 'summary':
                continue
            if 'full' in patient_data:
                I0_full = patient_data['full']['I0_center_median']
                I0_low = patient_data.get('low', {}).get('I0_center_median', I0_full * 0.25)
                alpha = I0_low / I0_full if I0_full > 0 else 0.25
                print(f"\n  LDCT ({patient_id}):")
                print(f"    --I0_high {I0_full:.0f}  (center-region median)")
                print(f"    --I0_low  {I0_low:.0f}")
                print(f"    alpha = {alpha:.6f}")
                print(f"    (current training uses: I0_high=1e5, I0_low=1e4, alpha=0.1)")

    print()


def save_results_json(detect2_results, ldct_results, output_dir):
    """Save results to JSON (convert numpy arrays to lists)."""

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    combined = {}
    if detect2_results:
        combined['2detect'] = to_serializable(detect2_results)
    if ldct_results:
        combined['ldct'] = to_serializable(ldct_results)

    out_path = os.path.join(output_dir, 'photon_count_results.json')
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Estimate real photon counts (I0)')
    parser.add_argument('--dataset', choices=['2detect', 'ldct', 'both'],
                        default='both', help='Which dataset to analyze')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for plots and JSON')
    parser.add_argument('--n_ldct_slices', type=int, default=50,
                        help='Number of LDCT slices to sample per patient')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'photon_count_results')
    os.makedirs(args.output_dir, exist_ok=True)

    detect2_results = None
    ldct_results = {}

    if args.dataset in ('2detect', 'both'):
        print("=" * 60)
        print("Estimating 2DeteCT photon counts...")
        print("=" * 60)

        sample_slices = list(range(1, 1001, 83))[:12]
        print(f"  Sampling slices: {sample_slices}")

        detect2_results = estimate_2detect_I0(DETECT2_PATH, sample_slices, modes=(1, 2))

        print("\n  Computing Fano factors...")
        detect2_results['fano'] = {}
        fano_slices = [sample_slices[0], sample_slices[len(sample_slices) // 2],
                       sample_slices[-1]]
        for mode in [1, 2]:
            fano_all = []
            for sl in fano_slices:
                I0_profile = detect2_results[f'mode{mode}']['I0_profile']
                fano = validate_2detect_fano(DETECT2_PATH, sl, mode, I0_profile)
                if not np.isnan(fano.get('fano_mean', float('nan'))):
                    fano_all.append(fano)

            if fano_all:
                detect2_results['fano'][f'mode{mode}'] = {
                    'fano_mean': float(np.mean([f['fano_mean'] for f in fano_all])),
                    'fano_median': float(np.median([f['fano_median'] for f in fano_all])),
                    'fano_std': float(np.std([f['fano_mean'] for f in fano_all])),
                    'n_air_pixels': int(np.mean([f['n_air_pixels'] for f in fano_all])),
                    'mean_air_counts': float(np.mean([f['mean_air_counts'] for f in fano_all])),
                    'n_slices_analyzed': len(fano_all),
                }
            else:
                detect2_results['fano'][f'mode{mode}'] = {
                    'fano_mean': float('nan'),
                    'warning': 'No valid Fano estimates',
                }

        print("  Done.")

    if args.dataset in ('ldct', 'both'):
        print("\n" + "=" * 60)
        print("Estimating LDCT photon counts...")
        print("=" * 60)

        patients = ['C004', 'C120']
        for patient_id in patients:
            full_dir = os.path.join(LDCT_PATH, f'{patient_id}_full_complete')
            low_dir = os.path.join(LDCT_PATH, f'{patient_id}_low_complete')
            if not os.path.exists(full_dir):
                print(f"  Skipping {patient_id}: no full-dose data")
                continue
            if not os.path.exists(low_dir):
                print(f"  Skipping {patient_id}: no low-dose data")
                continue

            print(f"\n  Patient: {patient_id}")
            ldct_results[patient_id] = estimate_ldct_I0(
                LDCT_PATH, patient_id, n_sample_slices=args.n_ldct_slices)

    print_summary(detect2_results, ldct_results if ldct_results else None)
    plot_profiles(detect2_results, ldct_results if ldct_results else None, args.output_dir)
    save_results_json(detect2_results, ldct_results if ldct_results else None, args.output_dir)


if __name__ == '__main__':
    main()
