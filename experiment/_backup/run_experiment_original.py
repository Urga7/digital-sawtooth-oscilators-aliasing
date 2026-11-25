import csv
import json
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import ShortTimeFFT, windows  # noqa: E402

from oscillators import FS, additive, dpw2, naive, polyblep  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

NFFT = 2 ** 17                      # analysis length (about 3 s)
WIN = windows.blackmanharris(NFFT, sym=False)   # sidelobes below -92 dB
BIN = FS / NFFT                     # bin spacing, about 0.34 Hz
GUARD = 8                           # bins on each side of a harmonic
BAND = (20.0, 20000.0)              # audible band for alias energy
NOTES = np.arange(21, 121)          # A0 (27.5 Hz) to C9 (8372 Hz)
ADD_NOTES = np.arange(21, 121, 6)   # additive is slow, so a subset
THRESHOLDS = (-30.0, -40.0)         # dB, for the "highest clean note"
CPU_F0 = (110.0, 440.0, 1760.0)
CPU_SECONDS = 10
CPU_REPEATS = 7

OSC = {"Naive": naive, "PolyBLEP": polyblep, "DPW-2": dpw2}
STYLE = {
    "Naive": dict(color="#D85A30", ls="-", lw=1.2),
    "PolyBLEP": dict(color="#1D6FA5", ls="--", lw=1.2),
    "DPW-2": dict(color="#1D9E75", ls="-.", lw=1.2),
    "Additive": dict(color="black", ls="none", marker="o", ms=3),
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


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def note_name(m):
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[int(m) % 12]}{int(m) // 12 - 1}"


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
    """Centre bin of every harmonic below Nyquist."""
    k = np.arange(1, int(np.floor((FS / 2.0) / f0 - 1e-9)) + 1)
    return k, np.round(k * f0 / BIN).astype(int)


def harmonic_mask(f0):
    mask = np.zeros(FREQS.size, bool)
    for c in harmonic_bins(f0)[1]:
        mask[max(0, c - GUARD): c + GUARD + 1] = True
    return mask


def ahr_db(x, f0):
    """Alias-to-harmonic ratio (dB): inharmonic power in 20 Hz - 20 kHz
    relative to the total harmonic power; unweighted and A-weighted."""
    P = power_spectrum(x)
    H = harmonic_mask(f0)
    alias = IN_BAND & ~H
    plain = 10 * np.log10(P[alias].sum() / P[H].sum())
    weighted = 10 * np.log10((P * A_W)[alias].sum() / (P * A_W)[H].sum())
    return plain, weighted


def harmonic_amplitudes(x, f0):
    """Amplitude of each harmonic from the energy in its main lobe."""
    P = power_spectrum(x)
    k, centres = harmonic_bins(f0)
    scale = NFFT * np.sum(WIN ** 2) / 4.0
    amps = np.array([
        np.sqrt(P[max(0, c - GUARD): c + GUARD + 1].sum() / scale)
        for c in centres
    ])
    return k, amps


def highest_clean_note(values, threshold):
    """Highest note such that every note up to it stays below threshold."""
    best = None
    for m, v in zip(NOTES, values):
        if v > threshold:
            break
        best = int(m)
    return best


def experiment_ahr():
    rows, curves = [], {name: ([], []) for name in OSC}
    add_vals = {}
    for m in NOTES:
        f0 = float(midi_to_hz(m))
        row = {"midi": int(m), "note": note_name(m), "f0_hz": round(f0, 3)}
        for name, osc in OSC.items():
            plain, weighted = ahr_db(osc(f0, NFFT), f0)
            row[f"{name}_dB"] = round(plain, 2)
            row[f"{name}_dBA"] = round(weighted, 2)
            curves[name][0].append(plain)
            curves[name][1].append(weighted)
        if m in ADD_NOTES:
            plain, weighted = ahr_db(additive(f0, NFFT), f0)
            row["Additive_dB"] = round(plain, 2)
            row["Additive_dBA"] = round(weighted, 2)
            add_vals[int(m)] = (plain, weighted)
        else:
            row["Additive_dB"] = row["Additive_dBA"] = ""
        rows.append(row)
        print(f"  {row['note']:>4} {f0:8.1f} Hz  " + "  ".join(
            f"{n} {row[n + '_dB']:7.2f}" for n in OSC))
    with open(RES / "ahr_vs_pitch.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return curves, add_vals


def plot_ahr(curves, add_vals):
    f = midi_to_hz(NOTES)
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for name, (plain, _) in curves.items():
        ax.semilogx(f, plain, label=name, **STYLE[name])
    am = sorted(add_vals)
    ax.semilogx(midi_to_hz(am), [add_vals[m][0] for m in am],
                label="Additive (floor)", **STYLE["Additive"])
    for t in THRESHOLDS:
        ax.axhline(t, color="0.55", lw=0.6, ls=":")
    ax.axvline(midi_to_hz(108), color="0.55", lw=0.6, ls=":")
    ax.text(midi_to_hz(108) * 0.95, -4, "C8", fontsize=7, color="0.35",
            ha="right", va="center")
    ax.set_xlabel("Fundamental frequency (Hz)")
    ax.set_ylabel("Alias-to-harmonic ratio (dB)")
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(-122, 0)
    ax.grid(True, which="major", lw=0.3, color="0.85")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              ncol=4, numpoints=1, handlelength=1.8, columnspacing=0.9,
              handletextpad=0.4, borderaxespad=0.2)
    fig.savefig(FIG / "fig_ahr.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / "fig_ahr.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def replot_ahr_from_csv():
    """Redraw Fig. 1 from results/ahr_vs_pitch.csv without recomputing."""
    curves = {name: ([], []) for name in OSC}
    add_vals = {}
    with open(RES / "ahr_vs_pitch.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            for name in OSC:
                curves[name][0].append(float(row[f"{name}_dB"]))
                curves[name][1].append(float(row[f"{name}_dBA"]))
            if row["Additive_dB"]:
                add_vals[int(row["midi"])] = (float(row["Additive_dB"]),
                                              float(row["Additive_dBA"]))
    plot_ahr(curves, add_vals)


def plot_spectrograms():
    dur, f1, f2 = 3.0, 100.0, 8000.0
    n = int(dur * FS)
    t = np.arange(n) / FS
    sweep = f1 * (f2 / f1) ** (t / dur)
    sft = ShortTimeFFT(windows.hann(2048, sym=False), hop=256, fs=FS,
                       scale_to="magnitude")
    specs = {}
    for name, osc in OSC.items():
        S = np.abs(sft.stft(osc(sweep, n)))
        specs[name] = 20 * np.log10(S + 1e-12)
    ref = max(s.max() for s in specs.values())
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0), sharey=True)
    extent = sft.extent(n)
    for ax, (name, S) in zip(axes, specs.items()):
        im = ax.imshow(S - ref, origin="lower", aspect="auto",
                       extent=(extent[0], extent[1], extent[2] / 1000,
                               extent[3] / 1000),
                       vmin=-100, vmax=0, cmap="magma",
                       interpolation="nearest")
        ax.set_xlim(0, dur)
        ax.set_ylim(0, FS / 2000)
        ax.set_title(name)
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (kHz)")
    cb = fig.colorbar(im, ax=axes, pad=0.01, fraction=0.025)
    cb.set_label("Level (dB)")
    fig.savefig(FIG / "fig_spectrograms.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / "fig_spectrograms.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def experiment_harmonics(f0=440.0):
    ideal_k, ideal = harmonic_amplitudes(additive(f0, NFFT), f0)
    theory = 2.0 / (np.pi * ideal_k)
    out = {"f0_hz": f0, "harmonic_freq_hz": (ideal_k * f0).tolist()}
    rows = [{"k": int(k), "freq_hz": round(k * f0, 1)} for k in ideal_k]
    for name, osc in OSC.items():
        _, amps = harmonic_amplitudes(osc(f0, NFFT), f0)
        err = 20 * np.log10(amps / theory)
        out[name] = err.tolist()
        for r, e in zip(rows, err):
            r[name + "_err_dB"] = round(float(e), 3)
    add_err = 20 * np.log10(ideal / theory)
    for r, e in zip(rows, add_err):
        r["Additive_err_dB"] = round(float(e), 3)
    with open(RES / "harmonics_440.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    fk = ideal_k * f0 / 1000
    for name in ("PolyBLEP", "DPW-2"):
        ax.plot(fk, out[name], label=name, **STYLE[name])
    ax.plot(fk, add_err, color="black", lw=0.8, label="Additive")
    ax.set_xlabel("Harmonic frequency (kHz)")
    ax.set_ylabel("Level error (dB)")
    ax.set_xlim(0, FS / 2000)
    ax.grid(True, lw=0.3, color="0.85")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_harmonics.pdf")
    fig.savefig(FIG / "fig_harmonics.png")
    plt.close(fig)

    def err_near(name, target):
        i = int(np.argmin(np.abs(ideal_k * f0 - target)))
        return {"freq_hz": float(ideal_k[i] * f0), "err_dB": float(out[name][i])}

    return {name: {"5k": err_near(name, 5000), "10k": err_near(name, 10000),
                   "15k": err_near(name, 15000), "20k": err_near(name, 20000)}
            for name in OSC}


def experiment_cpu():
    n = CPU_SECONDS * FS
    funcs = dict(OSC, Additive=additive)
    rows, table = [], {}
    for f0 in CPU_F0:
        for name, osc in funcs.items():
            times = []
            for _ in range(CPU_REPEATS):
                t0 = time.perf_counter()
                osc(f0, n)
                times.append(time.perf_counter() - t0)
            ms_per_s = 1000 * float(np.median(times)) / CPU_SECONDS
            table.setdefault(str(int(f0)), {})[name] = ms_per_s
        base = table[str(int(f0))]["Naive"]
        for name in funcs:
            ms = table[str(int(f0))][name]
            rows.append({"f0_hz": f0, "method": name,
                         "ms_per_second_audio": round(ms, 3),
                         "relative_to_naive": round(ms / base, 2)})
            print(f"  {f0:6.0f} Hz {name:9s} {ms:8.3f} ms/s  x{ms / base:.1f}")
    with open(RES / "cpu.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return table


def main():
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    print("Alias-to-harmonic ratio vs pitch")
    curves, add_vals = experiment_ahr()
    plot_ahr(curves, add_vals)

    summary = {"fs": FS, "nfft": NFFT, "guard_bins": GUARD, "band_hz": BAND,
               "key_notes": {}, "highest_clean_note": {},
               "additive_floor_dB": [min(v[0] for v in add_vals.values()),
                                     max(v[0] for v in add_vals.values())]}
    for m in (33, 45, 57, 69, 81, 93, 105, 108):
        i = int(np.where(NOTES == m)[0][0])
        summary["key_notes"][note_name(m)] = {
            "f0_hz": round(float(midi_to_hz(m)), 2),
            **{name: {"dB": round(curves[name][0][i], 2),
                      "dBA": round(curves[name][1][i], 2)} for name in OSC}}
    for name in OSC:
        summary["highest_clean_note"][name] = {}
        for label, idx in (("dB", 0), ("dBA", 1)):
            for t in THRESHOLDS:
                m = highest_clean_note(curves[name][idx], t)
                summary["highest_clean_note"][name][f"{label}<={t:g}"] = (
                    None if m is None else
                    {"note": note_name(m), "f0_hz": round(float(midi_to_hz(m)), 1)})
        gains = np.array(curves["Naive"][0]) - np.array(curves[name][0])
        summary.setdefault("improvement_over_naive_dB", {})[name] = {
            "min": round(float(gains.min()), 2),
            "median": round(float(np.median(gains)), 2),
            "max": round(float(gains.max()), 2)}

    print("Spectrograms of a 100 Hz - 8 kHz sweep")
    plot_spectrograms()
    print("Harmonic level error at 440 Hz")
    summary["harmonic_error_440Hz"] = experiment_harmonics(440.0)
    print("CPU cost")
    summary["cpu_ms_per_second_audio"] = experiment_cpu()

    with open(RES / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys

    if "--replot" in sys.argv:
        replot_ahr_from_csv()
    else:
        main()
