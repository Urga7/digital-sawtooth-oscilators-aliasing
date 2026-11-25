"""Short musical examples rendered with the four oscillators.

    python demo.py

The sweep in run_experiment.py shows aliasing under controlled conditions;
these clips show it in two typical synthesizer use cases:

- lead: a bright, unfiltered mono lead with portamento and delayed vibrato,
  played in octave 5 and then an octave higher (the worst case);
- bass: a sixteenth-note bass line through a resonant lowpass filter with a
  filter envelope (a typical subtractive patch, where aliasing is hardest
  to hear).

Each clip uses the same pitch, envelope and filter control signals for all
oscillators and is written with the same gain, so the files can be compared
by ear. <clip>_aliasing-only_<method>.wav contains only the aliasing of a
method: its output minus the additive reference with the method's harmonic
response applied (see harmonic_response), so the slight treble loss of
PolyBLEP and DPW-2 is not counted as aliasing.

Outputs: audio/demos/*.wav, results/demo_aliasing.csv and
figures/fig_demo_spectrograms.{pdf,png}.
"""

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from scipy.signal import ShortTimeFFT, lfilter, windows  # noqa: E402

from oscillators import FS, METHODS, additive  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
AUDIO = HERE / "audio" / "demos"
RES = HERE / "results"
FIG = HERE / "figures"

ADDITIVE_FADE = 0.05   # as in the sweep: no clicks when harmonics cross Nyquist
TAIL = 0.5             # seconds of silence/release after the last note
PEAK = 0.8             # common peak level of all versions of a clip
BAND = (20.0, 20000.0)  # band for the aliasing measure, as for the AHR
ALIASING = [m for m in METHODS if m != "Additive"]

# ==========================================================================
# Clips
# ==========================================================================
# Notes are (MIDI note, length in beats) and follow each other without gaps;
# "gate" is the fraction of each note that is held before the release.
_PHRASE = [(76, 0.5), (81, 0.5), (83, 0.5), (84, 1.0), (83, 0.5), (81, 0.5),
           (79, 1.0), (76, 1.0), (79, 0.5), (81, 2.0)]   # E5 .. C6, A minor
_RIFF = [33, 33, 45, 33, 33, 45, 33, 43,
         31, 31, 43, 31, 36, 38, 40, 45]                  # around A1 (55 Hz)

CLIPS = {
    "lead": dict(
        bpm=120,
        notes=_PHRASE + [(m + 12, b) for m, b in _PHRASE],
        gate=0.9,
        glide=0.025,                     # portamento time constant (s)
        vibrato=dict(rate=5.5, cents=25.0, delay=0.25, ramp=0.3),
        amp=dict(a=0.005, d=0.3, s=0.8, r=0.12),
        filt=None,
    ),
    "bass": dict(
        bpm=120,
        notes=[(m, 0.25) for m in _RIFF * 2],
        gate=0.75,
        glide=0.0,
        vibrato=None,
        amp=dict(a=0.002, d=0.25, s=0.7, r=0.04),
        # cutoff = base * 2^(octaves * filter envelope), resonance q
        filt=dict(base=200.0, octaves=4.5, a=0.002, d=0.12, q=2.5),
    ),
}


# ==========================================================================
# Control signals
# ==========================================================================
def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def schedule(clip):
    """Note events (MIDI note, start sample, length in samples), clip length."""
    beat = 60.0 / clip["bpm"] * FS
    events, t = [], 0.0
    for m, beats in clip["notes"]:
        start = int(round(t))
        t += beats * beat
        events.append((m, start, int(round(t)) - start))
    return events, int(round(t + TAIL * FS))


def pitch_curve(clip, events, n):
    """Per-sample fundamental frequency with portamento and vibrato."""
    pitch = np.empty(n)    # MIDI note number
    since = np.empty(n)    # seconds since the current note started
    ends = [s for _, s, _ in events[1:]] + [n]
    for (m, s, _), e in zip(events, ends):
        pitch[s:e] = m
        since[s:e] = np.arange(e - s) / FS
    if clip["glide"] > 0:
        a = np.exp(-1.0 / (clip["glide"] * FS))
        pitch, _ = lfilter([1.0 - a], [1.0, -a], pitch, zi=[a * pitch[0]])
    vib = clip["vibrato"]
    if vib:
        depth = vib["cents"] / 100.0 * np.clip(
            (since - vib["delay"]) / vib["ramp"], 0.0, 1.0)
        pitch = pitch + depth * np.sin(2 * np.pi * vib["rate"] * np.arange(n) / FS)
    return midi_to_hz(pitch)


def adsr(gate_len, a, d, s, r):
    """One note: linear attack, exponential decay to s, linear release."""
    t = np.arange(gate_len + int(r * FS)) / FS
    held = np.minimum(t / a, 1.0) * (s + (1 - s) * np.exp(-np.maximum(t - a, 0) / d))
    g = gate_len / FS
    off = held[gate_len - 1] if gate_len > 0 else 0.0
    return np.where(t < g, held, off * np.clip(1 - (t - g) / r, 0.0, 1.0))


def envelope(events, n, gate, a, d, s, r):
    """Envelope of the whole clip, retriggered by every note (mono synth)."""
    env = np.zeros(n)
    for _, start, length in events:
        e = adsr(int(length * gate), a, d, s, r)
        end = min(n, start + len(e))
        env[start:end] = np.maximum(env[start:end], e[:end - start])
    return env


def svf_lowpass(x, fc, q):
    """Resonant lowpass: trapezoidal state-variable filter, per-sample cutoff.

    The topology-preserving form stays stable when the cutoff changes from
    sample to sample, which a biquad with recomputed coefficients does not.
    """
    g = np.tan(np.pi * np.minimum(fc, 0.45 * FS) / FS)
    a1 = 1.0 / (1.0 + g * (g + 1.0 / q))
    a2 = g * a1
    a3 = g * a2
    y = []
    ic1 = ic2 = 0.0
    for xi, b1, b2, b3 in zip(x.tolist(), a1.tolist(), a2.tolist(), a3.tolist()):
        v3 = xi - ic2
        v1 = b1 * ic1 + b2 * v3
        v2 = ic2 + b2 * ic1 + b3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        y.append(v2)
    return np.array(y)


# ==========================================================================
# Rendering
# ==========================================================================
def oscillate(name, f0):
    if name == "Additive":
        return additive(f0, len(f0), fade=ADDITIVE_FADE)
    return METHODS[name](f0, len(f0))


def render(clip):
    """Render a clip with every oscillator, same control signals for all."""
    events, n = schedule(clip)
    f0 = pitch_curve(clip, events, n)
    amp = envelope(events, n, clip["gate"], **clip["amp"])
    filt = clip["filt"]
    if filt:
        fenv = envelope(events, n, clip["gate"], filt["a"], filt["d"], 0.0,
                        clip["amp"]["r"])
        fc = filt["base"] * 2.0 ** (filt["octaves"] * fenv)
    out = {}
    for name in METHODS:
        y = oscillate(name, f0)
        if filt:
            y = svf_lowpass(y, fc, filt["q"])
        out[name] = amp * y
    return out


# ==========================================================================
# Error measure and figure
# ==========================================================================
def harmonic_response(x, name):
    """Apply a method's effect on the wanted harmonics to the reference x.

    PolyBLEP replaces every step by one smoothed with a two-sample triangle,
    which scales a component at frequency f by sinc^2(f / fs). DPW-2's first
    difference scales it by sinc(f / fs) and delays it by half a sample.
    (At 19.8 kHz these give the 6.2 dB and 3.1 dB of Fig. 3.) The FFT
    filtering is circular, which is harmless because the clips start and end
    in silence.
    """
    if name not in ("PolyBLEP", "DPW-2"):
        return x
    X = np.fft.rfft(x)
    w = 2 * np.pi * np.arange(len(X)) / len(x)    # radians per sample
    s = np.sinc(w / (2 * np.pi))                  # sin(w/2) / (w/2)
    H = s ** 2 if name == "PolyBLEP" else s * np.exp(-0.5j * w)
    return np.fft.irfft(X * H, len(x))


def band_energy(x):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    keep = (f >= BAND[0]) & (f <= BAND[1])
    return np.sum(np.abs(X[keep]) ** 2)


def aliasing(out, name):
    """Aliasing signal of a method: output minus matched reference."""
    return out[name] - harmonic_response(out["Additive"], name)


def alias_db(out, name):
    """Aliasing energy relative to the reference, 20 Hz to 20 kHz. On steady
    tones this equals the AHR of run_experiment.py."""
    return 10 * np.log10(band_energy(aliasing(out, name))
                         / band_energy(out["Additive"]))


def plot(rendered):
    sft = ShortTimeFFT(windows.hann(2048, sym=False), hop=256, fs=FS,
                       scale_to="magnitude")
    fig, axes = plt.subplots(len(rendered), len(METHODS), figsize=(7.1, 4.2),
                             sharey=True, squeeze=False, layout="constrained")
    for row, (clip_name, out) in zip(axes, rendered.items()):
        specs = {k: 20 * np.log10(np.abs(sft.stft(out[k])) + 1e-12)
                 for k in METHODS}
        ref = max(S.max() for S in specs.values())
        n = len(out["Naive"])
        t0, t1, fa, fb = sft.extent(n)
        for ax, (name, S) in zip(row, specs.items()):
            im = ax.imshow(S - ref, origin="lower", aspect="auto",
                           extent=(t0, t1, fa / 1000, fb / 1000),
                           vmin=-100, vmax=0, cmap="magma",
                           interpolation="nearest")
            ax.set_xlim(0, n / FS)
            ax.set_ylim(0, FS / 2000)
            ax.set_title(f"{clip_name}: {name}", fontsize=8)
            ax.tick_params(labelsize=7)
        row[0].set_ylabel("Frequency (kHz)", fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Time (s)", fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.025)
    cb.set_label("Level (dB)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_demo_spectrograms.{ext}", dpi=200)
    plt.close(fig)


def main():
    AUDIO.mkdir(parents=True, exist_ok=True)
    rendered, rows = {}, []
    for clip_name, clip in CLIPS.items():
        out = render(clip)
        gain = PEAK / max(np.abs(y).max() for y in out.values())
        out = {k: gain * y for k, y in out.items()}
        rendered[clip_name] = out

        files = dict(out)
        for name in ALIASING:
            files[f"aliasing-only_{name}"] = aliasing(out, name)
            rows.append(dict(clip=clip_name, method=name,
                             aliasing_db=round(alias_db(out, name), 1)))
        for name, y in files.items():
            wavfile.write(AUDIO / f"{clip_name}_{name}.wav", FS,
                          y.astype(np.float32))
        print(f"{clip_name}: {len(out['Naive']) / FS:.1f} s, "
              f"{len(files)} files in {AUDIO.relative_to(HERE).as_posix()}")

    with open(RES / "demo_aliasing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "method", "aliasing_db"])
        w.writeheader()
        w.writerows(rows)
    print("\nAliasing relative to the signal (dB, 20 Hz - 20 kHz):")
    for r in rows:
        print(f"  {r['clip']:5s} {r['method']:9s} {r['aliasing_db']:7.1f}")

    plot(rendered)
    print("\nFigure: figures/fig_demo_spectrograms.{pdf,png}")


if __name__ == "__main__":
    main()
