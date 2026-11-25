"""Measurements and figures for the antialiased sawtooth oscillator comparison.

    python run_experiment.py            run everything
    python run_experiment.py --replot   redraw the AHR figure from the CSV
    python run_experiment.py --no-cpu   skip the (slow) timing experiment

Outputs go to results/ (CSV + JSON), figures/ (PDF + PNG) and audio/ (WAV).
"""

import argparse
import csv
import json
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from scipy.signal import ShortTimeFFT, windows  # noqa: E402

from oscillators import FS, METHODS, additive  # noqa: E402

# ==========================================================================
# Configuration
# ==========================================================================
HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
AUDIO = HERE / "audio"

# Spectral analysis
NFFT = 2 ** 17                      # analysis length (about 3 s)
WIN = windows.blackmanharris(NFFT, sym=False)   # sidelobes below -92 dB
BIN = FS / NFFT                     # bin spacing, about 0.34 Hz
GUARD = 8                           # bins on each side of a harmonic
MAIN_LOBE = 4                       # half-width of the BH main lobe in bins
BAND = (20.0, 20000.0)              # audible band for alias energy

# Experiments
NOTES = np.arange(21, 121)          # A0 (27.5 Hz) to C9 (8372 Hz)
KEY_NOTES = (33, 45, 57, 69, 81, 93, 105, 108)  # A1..A7 and C8 for the summary
THRESHOLDS = (-30.0, -40.0)         # dB, for the "highest clean note"
HARMONICS_F0 = 440.0
SWEEP = dict(seconds=3.0, f_start=100.0, f_end=8000.0)
ADDITIVE_FADE = 0.05                # fade band for additive in the sweep
CPU_F0 = (110.0, 440.0, 1760.0)
CPU_SECONDS = 10
CPU_REPEATS = 7

# Methods that alias; "Additive" is the alias-free reference
# (its AHR is the measurement floor of the analysis, not a property of it).
ALIASING = [m for m in METHODS if m != "Additive"]

STYLE = {
    "Naive": dict(color="#D85A30", ls="-", lw=1.2),
    "PolyBLEP": dict(color="#1D6FA5", ls="--", lw=1.2),
    "DPW-2": dict(color="#1D9E75", ls="-.", lw=1.2),
    "Additive": dict(color="black", ls="-", lw=0.7),
}

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


# ==========================================================================
# Small helpers
# ==========================================================================
def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def note_name(m):
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[int(m) % 12]}{int(m) // 12 - 1}"


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_fig(fig, name, **kwargs):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", **kwargs)
    plt.close(fig)


# ==========================================================================
# Spectral analysis
# ==========================================================================
def a_weight_power(f):
    """Squared magnitude of the IEC 61672 A-weighting curve (linear)."""
    f2 = np.asarray(f, float) ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    return (ra * 10.0 ** (2.0 / 20.0)) ** 2


FREQS = np.arange(NFFT // 2 + 1) * BIN
IN_BAND = (FREQS >= BAND[0]) & (FREQS <= BAND[1])
A_W = a_weight_power(FREQS)


def power_spectrum(x):
    X = np.fft.rfft(x[:NFFT] * WIN)
    return X.real ** 2 + X.imag ** 2


def harmonic_bins(f0):
    """Harmonic numbers and centre bins of every harmonic below Nyquist."""
    k = np.arange(1, int(np.floor((FS / 2.0) / f0 - 1e-9)) + 1)
    return k, np.round(k * f0 / BIN).astype(int)


def harmonic_mask(f0):
    """Boolean mask of all bins within +-GUARD bins of a harmonic."""
    _, centres = harmonic_bins(f0)
    idx = (centres[:, None] + np.arange(-GUARD, GUARD + 1)).ravel()
    mask = np.zeros(FREQS.size, bool)
    mask[idx[(idx >= 0) & (idx < FREQS.size)]] = True
    return mask


def ahr_db(x, f0):
    """Alias-to-harmonic ratio (dB): inharmonic power in 20 Hz - 20 kHz
    relative to the total harmonic power; unweighted and A-weighted."""
    P = power_spectrum(x)
    Pw = P * A_W
    H = harmonic_mask(f0)
    alias = IN_BAND & ~H
    plain = 10 * np.log10(P[alias].sum() / P[H].sum())
    weighted = 10 * np.log10(Pw[alias].sum() / Pw[H].sum())
    return plain, weighted


def harmonic_amplitudes(x, f0):
    """Amplitude of each harmonic from the energy in its main lobe."""
    P = power_spectrum(x)
    k, centres = harmonic_bins(f0)
    lo = np.maximum(centres - GUARD, 0)
    hi = np.minimum(centres + GUARD, P.size - 1)
    cum = np.concatenate(([0.0], np.cumsum(P)))    # sum over [lo, hi] in O(1)
    scale = NFFT * np.sum(WIN ** 2) / 4.0
    return k, np.sqrt((cum[hi + 1] - cum[lo]) / scale)


def alias_gap_hz(f0, folds=(1, 2, 3)):
    """Smallest distance (Hz) between a folded harmonic and a true harmonic.

    A harmonic k*f0 above Nyquist folds to m*fs - k*f0. All of these lie on
    a grid shifted from the harmonics by (m*fs mod f0), so the gap only
    depends on that remainder.
    """
    gaps = []
    for m in folds:
        r = (m * FS) % f0
        gaps.append(min(r, f0 - r))
    return min(gaps)


def analysis_f0(midi):
    """Frequency at which a note is analysed, and its detuning in cents.

    For some notes (e.g. G1, G2, G3) the folded harmonics land within a few
    tenths of a Hz of true harmonics, inside the guard band, where they would
    be counted as harmonic power and the AHR would come out far too low.
    Such notes are detuned upwards in 1-cent steps until the aliases (with
    their main lobe) are clear of the guard band. This doesn't change the
    musical meaning of the result.
    """
    base = float(midi_to_hz(midi))
    min_gap = (GUARD + MAIN_LOBE + 1) * BIN
    cents = 0
    while alias_gap_hz(base * 2.0 ** (cents / 1200.0)) < min_gap:
        cents += 1
    return base * 2.0 ** (cents / 1200.0), cents


def highest_clean_note(notes, values, threshold):
    """Highest note such that every note up to it stays below threshold."""
    best = None
    for m, v in zip(notes, values):
        if v > threshold:
            break
        best = int(m)
    return best


# ==========================================================================
# Experiment 1: alias-to-harmonic ratio vs pitch
# ==========================================================================
def experiment_ahr():
    """Returns (f0s, curves) with curves[name] = {"dB": [...], "dBA": [...]}."""
    rows, f0s = [], []
    curves = {name: {"dB": [], "dBA": []} for name in METHODS}
    for m in NOTES:
        f0, cents = analysis_f0(m)
        f0s.append(f0)
        row = {"midi": int(m), "note": note_name(m), "f0_hz": round(f0, 3),
               "detune_cents": cents}
        for name, osc in METHODS.items():
            plain, weighted = ahr_db(osc(f0, NFFT), f0)
            curves[name]["dB"].append(plain)
            curves[name]["dBA"].append(weighted)
            row[f"{name}_dB"] = round(plain, 2)
            row[f"{name}_dBA"] = round(weighted, 2)
        rows.append(row)
        print(f"  {row['note']:>4} {f0:8.1f} Hz  " + "  ".join(
            f"{n} {row[n + '_dB']:7.2f}" for n in METHODS))
    write_csv(RES / "ahr_vs_pitch.csv", rows)
    return np.array(f0s), curves


def load_ahr_csv():
    """Read results/ahr_vs_pitch.csv back into (f0s, curves)."""
    f0s, curves = [], {name: {"dB": [], "dBA": []} for name in METHODS}
    with open(RES / "ahr_vs_pitch.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            f0s.append(float(row["f0_hz"]))
            for name in METHODS:
                for unit in ("dB", "dBA"):
                    curves[name][unit].append(float(row[f"{name}_{unit}"]))
    return np.array(f0s), curves


def plot_ahr(f0s, curves):
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for name in METHODS:
        label = "Additive (floor)" if name == "Additive" else name
        ax.semilogx(f0s, curves[name]["dB"], label=label, **STYLE[name])
    for t in THRESHOLDS:
        ax.axhline(t, color="0.55", lw=0.6, ls=":")
    c8 = float(midi_to_hz(108))
    ax.axvline(c8, color="0.55", lw=0.6, ls=":")
    ax.text(c8 * 0.95, -4, "C8", fontsize=7, color="0.35",
            ha="right", va="center")
    ax.set_xlabel("Fundamental frequency (Hz)")
    ax.set_ylabel("Alias-to-harmonic ratio (dB)")
    ax.set_xlim(f0s[0], f0s[-1])
    ax.set_ylim(-122, 0)
    ax.grid(True, which="major", lw=0.3, color="0.85")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              ncol=4, handlelength=1.8, columnspacing=0.9,
              handletextpad=0.4, borderaxespad=0.2)
    save_fig(fig, "fig_ahr", bbox_inches="tight", pad_inches=0.02)


# ==========================================================================
# Experiment 2: exponential sweep (spectrograms + audio)
# ==========================================================================
def render_sweep(name, n, f0):
    if name == "Additive":
        return additive(f0, n, fade=ADDITIVE_FADE)
    return METHODS[name](f0, n)


def experiment_sweep():
    dur, f1, f2 = SWEEP["seconds"], SWEEP["f_start"], SWEEP["f_end"]
    n = int(dur * FS)
    t = np.arange(n) / FS
    f0 = f1 * (f2 / f1) ** (t / dur)
    sft = ShortTimeFFT(windows.hann(2048, sym=False), hop=256, fs=FS,
                       scale_to="magnitude")

    specs = {}
    for name in METHODS:
        y = render_sweep(name, n, f0)
        # same gain for every method so they can be compared by ear
        wavfile.write(AUDIO / f"sweep_{name}.wav", FS,
                      (0.5 * y).astype(np.float32))
        specs[name] = 20 * np.log10(np.abs(sft.stft(y)) + 1e-12)

    ref = max(s.max() for s in specs.values())
    t0, t1, fa, fb = sft.extent(n)
    fig, axes = plt.subplots(1, len(specs), figsize=(7.1, 1.9), sharey=True)
    for ax, (name, S) in zip(axes, specs.items()):
        im = ax.imshow(S - ref, origin="lower", aspect="auto",
                       extent=(t0, t1, fa / 1000, fb / 1000),
                       vmin=-100, vmax=0, cmap="magma",
                       interpolation="nearest")
        ax.set_xlim(0, dur)
        ax.set_ylim(0, FS / 2000)
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (kHz)")
    cb = fig.colorbar(im, ax=axes, pad=0.01, fraction=0.025)
    cb.set_label("Level (dB)")
    save_fig(fig, "fig_spectrograms", bbox_inches="tight", pad_inches=0.02)


# ==========================================================================
# Experiment 3: harmonic level error
# ==========================================================================
def experiment_harmonics(f0=HARMONICS_F0):
    k = harmonic_bins(f0)[0]
    theory = 2.0 / (np.pi * k)
    err = {name: 20 * np.log10(harmonic_amplitudes(osc(f0, NFFT), f0)[1] / theory)
           for name, osc in METHODS.items()}

    write_csv(RES / f"harmonics_{int(f0)}.csv", [
        {"k": int(kk), "freq_hz": round(kk * f0, 1),
         **{f"{name}_err_dB": round(float(e[i]), 3) for name, e in err.items()}}
        for i, kk in enumerate(k)])

    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    fk = k * f0 / 1000
    for name in ("PolyBLEP", "DPW-2", "Additive"):   # Naive overlaps Additive
        ax.plot(fk, err[name], label=name, **STYLE[name])
    ax.set_xlabel("Harmonic frequency (kHz)")
    ax.set_ylabel("Level error (dB)")
    ax.set_xlim(0, FS / 2000)
    ax.grid(True, lw=0.3, color="0.85")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(pad=0.3)
    save_fig(fig, "fig_harmonics")

    def err_near(name, target):
        i = int(np.argmin(np.abs(k * f0 - target)))
        return {"freq_hz": float(k[i] * f0), "err_dB": float(err[name][i])}

    targets = {"5k": 5000, "10k": 10000, "15k": 15000, "20k": 20000}
    return {name: {key: err_near(name, hz) for key, hz in targets.items()}
            for name in METHODS}


# ==========================================================================
# Experiment 4: computation time
# ==========================================================================
def time_oscillator(osc, f0, n):
    times = []
    for _ in range(CPU_REPEATS):
        t0 = time.perf_counter()
        osc(f0, n)
        times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times)) / CPU_SECONDS   # ms per s of audio


def experiment_cpu():
    n = CPU_SECONDS * FS
    rows, table = [], {}
    for f0 in CPU_F0:
        ms = {name: time_oscillator(osc, f0, n) for name, osc in METHODS.items()}
        table[str(int(f0))] = ms
        for name, v in ms.items():
            rel = v / ms["Naive"]
            rows.append({"f0_hz": f0, "method": name,
                         "ms_per_second_audio": round(v, 3),
                         "relative_to_naive": round(rel, 2)})
            print(f"  {f0:6.0f} Hz {name:9s} {v:8.3f} ms/s  x{rel:.1f}")
    write_csv(RES / "cpu.csv", rows)
    return table


# ==========================================================================
# Summary
# ==========================================================================
def ahr_summary(f0s, curves):
    note_index = {int(m): i for i, m in enumerate(NOTES)}
    summary = {
        "fs": FS, "nfft": NFFT, "guard_bins": GUARD, "band_hz": BAND,
        "detuned_notes": {note_name(m): analysis_f0(m)[1]
                          for m in NOTES if analysis_f0(m)[1]},
        "additive_floor_dB": [round(min(curves["Additive"]["dB"]), 2),
                              round(max(curves["Additive"]["dB"]), 2)],
        "key_notes": {}, "highest_clean_note": {},
        "improvement_over_naive_dB": {},
    }
    for m in KEY_NOTES:
        i = note_index[m]
        summary["key_notes"][note_name(m)] = {
            "f0_hz": round(float(f0s[i]), 2),
            **{name: {u: round(curves[name][u][i], 2) for u in ("dB", "dBA")}
               for name in ALIASING}}
    for name in ALIASING:
        clean = {}
        for unit in ("dB", "dBA"):
            for t in THRESHOLDS:
                m = highest_clean_note(NOTES, curves[name][unit], t)
                clean[f"{unit}<={t:g}"] = None if m is None else {
                    "note": note_name(m), "f0_hz": round(float(midi_to_hz(m)), 1)}
        summary["highest_clean_note"][name] = clean
        if name != "Naive":
            gains = np.array(curves["Naive"]["dB"]) - np.array(curves[name]["dB"])
            summary["improvement_over_naive_dB"][name] = {
                "min": round(float(gains.min()), 2),
                "median": round(float(np.median(gains)), 2),
                "max": round(float(gains.max()), 2)}
    return summary


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replot", action="store_true",
                        help="only redraw the AHR figure from the CSV")
    parser.add_argument("--no-cpu", action="store_true",
                        help="skip the timing experiment")
    args = parser.parse_args()

    for d in (RES, FIG, AUDIO):
        d.mkdir(exist_ok=True)

    if args.replot:
        plot_ahr(*load_ahr_csv())
        return

    print("Alias-to-harmonic ratio vs pitch")
    f0s, curves = experiment_ahr()
    plot_ahr(f0s, curves)
    summary = ahr_summary(f0s, curves)

    print(f"Spectrograms and audio of a {SWEEP['f_start']:g} Hz - "
          f"{SWEEP['f_end']:g} Hz sweep")
    experiment_sweep()

    print(f"Harmonic level error at {HARMONICS_F0:g} Hz")
    summary[f"harmonic_error_{int(HARMONICS_F0)}Hz"] = experiment_harmonics()

    if not args.no_cpu:
        print("CPU cost")
        summary["cpu_ms_per_second_audio"] = experiment_cpu()
    elif (RES / "summary.json").exists():          # keep the previous timings
        with open(RES / "summary.json") as fh:
            old = json.load(fh)
        if "cpu_ms_per_second_audio" in old:
            summary["cpu_ms_per_second_audio"] = old["cpu_ms_per_second_audio"]

    with open(RES / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
