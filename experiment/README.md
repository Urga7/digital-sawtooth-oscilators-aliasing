# Aliasing in digital sawtooth oscillators: experiment

## Running

```
pip install numpy scipy matplotlib
python run_experiment.py            # everything (under a minute)
python run_experiment.py --no-cpu   # skip timing, keep the previous timings
python run_experiment.py --replot   # only redraw Fig. 1 from the CSV
python demo.py                      # musical examples (a few seconds)
python paper_figures.py             # Figs. 1 and 2 of the paper
```

Results go into `results/`, figures (PDF + PNG) into `figures/`, audio into `audio/`.

## Oscillators (`oscillators.py`)

All share the signature `osc(f0, n)`; `f0` is a constant or a per-sample array (sweeps).

- **Naive**: trivially sampled sawtooth.
- **PolyBLEP**: naive sawtooth with a two-sample polynomial correction at each jump.
- **DPW-2**: squared naive sawtooth, differentiated and rescaled (half-sample delay).
- **Additive**: Fourier series truncated below Nyquist; the alias-free reference.
  The sines are generated with the Chebyshev recurrence
  `sin((k+1)t) = 2 cos(t) sin(kt) - sin((k-1)t)` (about 2x faster than calling
  `sin()` per harmonic; deviation from direct evaluation below −200 dB). In sweeps,
  harmonics are faded out over the top 5 % below Nyquist to avoid clicks.

## What is measured (`run_experiment.py`)

- **Alias-to-harmonic ratio (AHR)** for every MIDI note from A0 (27.5 Hz) to C9 (8372 Hz),
  for all four oscillators: power of all non-harmonic components between 20 Hz and 20 kHz
  relative to the total harmonic power, from a 2^17-point FFT with a Blackman-Harris
  window. Bins within ±8 bins of a harmonic count as harmonic. Reported unweighted (dB)
  and A-weighted (dBA). The Additive curve is the measurement floor of the analysis.
- **Detuning of some notes**: for some notes (e.g. G1, G2, G3) the folded harmonics land
  within a fraction of a Hz of true harmonics, inside the ±8-bin guard band, and would be
  counted as harmonic power (the AHR would come out tens of dB too low). Such notes are
  detuned upwards in 1-cent steps (at most a few cents) until the aliases are clear of the
  guard band. The detuning is listed in `ahr_vs_pitch.csv` and `summary.json`.
- **Highest clean note**: the highest note up to which the AHR stays below −30 dB or −40 dB.
- **Spectrograms and audio** of a 3 s exponential sweep from 100 Hz to 8 kHz, for all four
  oscillators (`figures/fig_spectrograms.*`, `audio/sweep_<method>.wav`, same gain for all).
- **Harmonic level error** at f0 = 440 Hz: how much each method attenuates harmonics
  compared with the ideal 2/(πk) amplitudes.
- **Computation time** per second of audio at 110, 440 and 1760 Hz (vectorised NumPy,
  median of 7 runs, 10 s of audio per run). Timings depend on the machine and on NumPy
  overhead, so they show orders of magnitude rather than real-time plugin costs.

## Musical examples (`demo.py`)

Two short clips rendered with all four oscillators, with identical pitch, envelope and
filter control signals and the same gain, for listening comparisons (e.g. in the talk):

- **lead** (8.5 s): unfiltered mono lead with portamento and delayed vibrato (25 cents,
  5.5 Hz), a phrase in octave 5 (E5 to C6) repeated an octave higher. The worst case:
  high notes, and the aliases move against the pitch during vibrato and glides.
- **bass** (4.5 s): sixteenth-note bass line around A1 through a resonant lowpass
  (trapezoidal state-variable filter) with a filter envelope. A typical subtractive patch.

`audio/demos/<clip>_<method>.wav` are the clips; `<clip>_aliasing-only_<method>.wav`
contain only the aliasing of a method: its output minus the additive reference with the
method's own harmonic response applied (sinc^2 for PolyBLEP, sinc and a half-sample delay
for DPW-2), so the slight treble loss of the corrected oscillators is not counted as
aliasing. `results/demo_aliasing.csv` gives the energy of that signal relative to the
reference, 20 Hz to 20 kHz; on steady tones it equals the AHR above within 0.3 dB.
Spectrograms of both clips are in `figures/fig_demo_spectrograms.*`.

## Files

- `oscillators.py`: the four oscillator implementations
- `run_experiment.py`: measurements, CSV/JSON results, figures and audio
- `demo.py`: musical example clips, their aliasing levels and spectrograms
- `paper_figures.py`: explanatory figures for the paper (samples around a jump,
  line spectra of one note with harmonics marked)
- `results/summary.json`: the numbers quoted in the paper
