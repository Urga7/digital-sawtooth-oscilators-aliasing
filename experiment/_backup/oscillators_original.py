import numpy as np

FS = 44100  # sampling rate in Hz


def phase(f0, n, fs=FS):
    """Return the normalised phase in [0, 1) and the per-sample increment."""
    if np.isscalar(f0):
        dt = float(f0) / fs
        ph = (np.arange(n) * dt) % 1.0
        return ph, np.full(n, dt)
    f0 = np.asarray(f0, dtype=float)
    if f0.shape != (n,):
        raise ValueError("per-sample frequency array must have length n")
    inc = f0 / fs
    ph = np.concatenate(([0.0], np.cumsum(inc[:-1]))) % 1.0
    return ph, inc


def naive(f0, n, fs=FS):
    """Trivially sampled sawtooth: samples of the ideal analog waveform."""
    ph, _ = phase(f0, n, fs)
    return 2.0 * ph - 1.0


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


def dpw2(f0, n, fs=FS):
    """Second-order differentiated parabolic waveform (DPW-2).

    The trivial sawtooth s is squared into a parabolic waveform, which has
    no discontinuity and therefore a spectrum that decays twice as fast.
    A first-order difference turns it back into a sawtooth; the factor
    1 / (4 * f0 / fs) restores unit amplitude. The output is delayed by
    half a sample relative to the other oscillators.
    """
    ph, dt = phase(f0, n, fs)
    p = (2.0 * ph - 1.0) ** 2
    p_prev = np.empty_like(p)
    p_prev[1:] = p[:-1]
    p_prev[0] = (2.0 * ((-dt[0]) % 1.0) - 1.0) ** 2
    dt_prev = np.empty_like(dt)
    dt_prev[1:] = dt[:-1]
    dt_prev[0] = dt[0]
    return (p - p_prev) / (4.0 * dt_prev)


def additive(f0, n, fs=FS):
    """Ideal bandlimited sawtooth: Fourier series truncated at Nyquist.

    Uses 2*x - 1 = -(2/pi) * sum_k sin(2*pi*k*x) / k, keeping only the
    harmonics k * f0 < fs / 2, so it shares its phase with naive().
    """
    if not np.isscalar(f0):
        raise ValueError("additive() supports a constant f0 only")
    n_harm = int(np.floor((fs / 2.0) / f0 - 1e-9))
    w = 2.0 * np.pi * float(f0) / fs * np.arange(n)
    y = np.zeros(n)
    for k in range(1, n_harm + 1):
        y += np.sin(k * w) / k
    return -(2.0 / np.pi) * y


METHODS = {
    "Naive": naive,
    "PolyBLEP": polyblep,
    "DPW-2": dpw2,
    "Additive": additive,
}
