"""Sawtooth oscillators with different strategies against aliasing.

Every oscillator has the signature ``osc(f0, n, fs=FS) -> np.ndarray`` and
returns ``n`` samples of a sawtooth rising from -1 to +1. ``f0`` is either a
constant frequency in Hz or an array of ``n`` per-sample frequencies (used for
pitch sweeps). All oscillators share the same phase, so their outputs can be
compared sample by sample (DPW-2 is delayed by half a sample, see below).
"""

import numpy as np

FS = 44100  # sampling rate in Hz


def phase(f0, n, fs=FS):
    """Normalised phase in [0, 1) and the per-sample phase increment.

    A constant f0 uses the exact closed form ``k * dt``; a frequency array is
    integrated with a float64 cumulative sum. Both start at phase 0.
    """
    if np.isscalar(f0):
        dt = float(f0) / fs
        return (np.arange(n) * dt) % 1.0, np.full(n, dt)
    f0 = np.asarray(f0, dtype=float)
    if f0.shape != (n,):
        raise ValueError("per-sample frequency array must have length n")
    inc = f0 / fs
    ph = np.concatenate(([0.0], np.cumsum(inc[:-1]))) % 1.0
    return ph, inc


# --------------------------------------------------------------------------
# Naive
# --------------------------------------------------------------------------
def naive(f0, n, fs=FS):
    """Trivially sampled sawtooth: samples of the ideal analog waveform."""
    ph, _ = phase(f0, n, fs)
    return 2.0 * ph - 1.0


# --------------------------------------------------------------------------
# PolyBLEP
# --------------------------------------------------------------------------
def polyblep_residual(ph, dt):
    """Two-sample polynomial approximation of the bandlimited step residual.

    The correction is non-zero only for the sample just after a
    discontinuity (ph < dt) and the sample just before it (ph > 1 - dt).
    """
    r = np.zeros_like(ph)
    after = ph < dt
    t = ph[after] / dt[after]
    r[after] = t + t - t * t - 1.0
    before = ph > 1.0 - dt
    t = (ph[before] - 1.0) / dt[before]
    r[before] = t * t + t + t + 1.0
    return r


def polyblep(f0, n, fs=FS):
    """Naive sawtooth with the PolyBLEP correction subtracted at each jump."""
    ph, dt = phase(f0, n, fs)
    return 2.0 * ph - 1.0 - polyblep_residual(ph, dt)


# --------------------------------------------------------------------------
# DPW-2
# --------------------------------------------------------------------------
def dpw2(f0, n, fs=FS):
    """Second-order differentiated parabolic waveform (DPW-2).

    The trivial sawtooth is squared into a parabolic waveform, which has no
    jump and therefore a spectrum that decays twice as fast (-12 dB/octave).
    A first-order difference turns it back into a sawtooth, and dividing by
    4 * f0 / fs restores unit amplitude. The previous sample of the very
    first output is computed from the phase one step before 0, so there is
    no start-up click. The output is delayed by half a sample relative to
    the other oscillators.
    """
    ph, dt = phase(f0, n, fs)
    p = (2.0 * ph - 1.0) ** 2
    p0_prev = (2.0 * ((-dt[0]) % 1.0) - 1.0) ** 2
    p_prev = np.concatenate(([p0_prev], p[:-1]))
    dt_prev = np.concatenate((dt[:1], dt[:-1]))
    return (p - p_prev) / (4.0 * dt_prev)


# --------------------------------------------------------------------------
# Additive (reference)
# --------------------------------------------------------------------------
def additive(f0, n, fs=FS, fade=0.0):
    """Ideal bandlimited sawtooth: Fourier series truncated below Nyquist.

    Uses 2*x - 1 = -(2/pi) * sum_k sin(2*pi*k*x) / k, so it shares its phase
    with naive(). Only harmonics with k * f0 < fs / 2 are kept.

    The sines are generated with the Chebyshev recurrence
        sin((k+1)t) = 2 cos(t) sin(kt) - sin((k-1)t),
    which needs one sin() and one cos() per sample in total instead of one
    sin() per harmonic per sample. Its rounding error stays far below the
    -90 dB measurement floor of the analysis.

    For a per-sample f0 (sweeps), harmonics disappear when they cross Nyquist.
    ``fade`` (a fraction of Nyquist, e.g. 0.05) fades them out over that
    band instead of switching them off abruptly, which would cause clicks.
    With fade=0 the series is truncated exactly, the ideal reference.
    """
    ph, inc = phase(f0, n, fs)
    nyq = 0.5
    f_norm = inc                                   # harmonic k is at k * f_norm
    n_harm = int(np.floor(nyq / f_norm.min() - 1e-9))
    constant = np.isscalar(f0)

    def gain(k):
        if constant:
            return 1.0                             # k <= n_harm already holds
        fk = k * f_norm
        if fade > 0.0:
            return np.clip((nyq - fk) / (fade * nyq), 0.0, 1.0)
        return (fk < nyq).astype(float)

    theta = 2.0 * np.pi * ph
    two_cos = 2.0 * np.cos(theta)
    s_prev = np.zeros(n)                           # sin(0 * theta)
    s = np.sin(theta)                              # sin(1 * theta)
    y = gain(1) * s
    for k in range(2, n_harm + 1):
        s_prev, s = s, two_cos * s - s_prev
        y += (gain(k) / k) * s
    return -(2.0 / np.pi) * y


METHODS = {
    "Naive": naive,
    "PolyBLEP": polyblep,
    "DPW-2": dpw2,
    "Additive": additive,
}
