"""Explanatory figures for the paper (Figs. 1 and 2).

    python paper_figures.py

fig_waveforms: samples of the four oscillators around one jump of the
sawtooth, on top of the ideal analog waveform.
fig_line_spectra: spectrum of a single note (E6, 1319 Hz) of the naive,
PolyBLEP and DPW-2 oscillators, with the true harmonics marked, so that
every other peak is an alias.

Uses the style and the analysis of run_experiment.py.
"""

import matplotlib.pyplot as plt
import numpy as np

import run_experiment as rx
from oscillators import FS, METHODS

ALIASING = ["Naive", "PolyBLEP", "DPW-2"]
MARKERS = {"Naive": "o", "PolyBLEP": "s", "DPW-2": "^", "Additive": "D"}


def fig_waveforms(f0=3150.0, n=40, lo=17, hi=31):
    """Samples around one discontinuity; DPW-2 is drawn half a sample
    earlier, which removes its half-sample delay."""
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    t = np.linspace(lo, hi, 4000)
    ax.plot(t, 2 * ((t * f0 / FS) % 1.0) - 1, color="0.6", lw=0.8,
            label="Analog")
    k = np.arange(n)
    for name in ("Naive", "PolyBLEP", "DPW-2", "Additive"):
        y = METHODS[name](f0, n)
        x = k - 0.5 if name == "DPW-2" else k
        keep = (x >= lo) & (x <= hi)
        st = rx.STYLE[name]
        ax.plot(x[keep], y[keep], color=st["color"], lw=0.5,
                ls="-", marker=MARKERS[name], ms=2.6,
                label=name if name != "Additive" else "Additive (ideal)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, lw=0.3, color="0.85")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              ncol=5, handlelength=1.3, columnspacing=0.6, handletextpad=0.3,
              borderaxespad=0.2, fontsize=6.5)
    rx.save_fig(fig, "fig_waveforms", bbox_inches="tight", pad_inches=0.02)


def fig_line_spectra(midi=88):
    f0 = float(rx.midi_to_hz(midi))
    fig, axes = plt.subplots(len(ALIASING), 1, figsize=(3.45, 2.9),
                             sharex=True)
    k, centres = rx.harmonic_bins(f0)
    for ax, name in zip(axes, ALIASING):
        x = METHODS[name](f0, rx.NFFT)
        P = rx.power_spectrum(x)
        db = 10 * np.log10(P / P[centres[0]] + 1e-20)
        ax.plot(rx.FREQS / 1000, db, color=rx.STYLE[name]["color"], lw=0.5)
        ax.plot(k * f0 / 1000, db[centres], ls="none", marker="o", ms=2.6,
                mfc="none", mec="black", mew=0.6)
        ahr = rx.ahr_db(x, f0)[0]
        ax.text(0.99, 0.93, f"{name},  AHR \u2212{abs(ahr):.1f} dB",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(fc="white", ec="none", pad=1.0))
        ax.set_ylim(-80, 5)
        ax.set_yticks([-80, -40, 0])
        ax.grid(True, lw=0.3, color="0.85")
    axes[1].set_ylabel("Level (dB)")
    axes[-1].set_xlabel("Frequency (kHz)")
    axes[-1].set_xlim(0, FS / 2000)
    fig.tight_layout(pad=0.3, h_pad=0.2)
    rx.save_fig(fig, "fig_line_spectra")


if __name__ == "__main__":
    fig_waveforms()
    fig_line_spectra()
    print("figures/fig_waveforms.{pdf,png}, figures/fig_line_spectra.{pdf,png}")
