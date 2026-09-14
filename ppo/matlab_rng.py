"""Bit-exact pure-Python reproduction of MATLAB's seeded uniform integer draw.

Reproduces:

    s = RandStream('mt19937ar', 'Seed', seed);
    user_positions = [randi(s, [x_min, x_max], num_users, 1), ...
                      randi(s, [y_min, y_max], num_users, 1)];

Findings that make it exact (each empirically verified, see module self-test):

1. MATLAB's 'mt19937ar' is the reference Mersenne Twister MT19937 with the classic
   scalar seeding routine ``init_genrand``.  MATLAB maps ``Seed`` 0 onto MT19937's
   own default seed 5489 rather than calling ``init_genrand(0)``; every non-zero
   seed is passed through unchanged.  (Checked: MATLAB ``rng(0); rand`` ->
   0.8147236863931789 == init_genrand(5489) res53[0]; MATLAB ``rng(1); rand`` ->
   0.4170220047025740 == init_genrand(1) res53[0].)  This is why
   ``numpy.random.RandomState(0)`` does NOT match: numpy's scalar seeding really is
   ``init_genrand(0)``, which yields 0.5488135039273248.

2. ``rand`` is the canonical 53-bit double: two 32-bit words per double,
   ``((w1 >> 5) * 2**26 + (w2 >> 6)) / 2**53``.

3. For these ranges ``randi(s, [lo, hi])`` is exactly
   ``floor(rand * (hi - lo + 1)) + lo`` -- one 53-bit double (two 32-bit words)
   consumed per value, no rejection sampling.  A masked-rejection-on-int32
   implementation and a raw ``int32 / 2**32`` implementation were both tested and
   both fail against the MATLAB ground truth.

4. Draw order: all ``num_users`` x values first (a full column), then all
   ``num_users`` y values, from the SAME continuing stream.

No MATLAB, no third-party RNG; numpy is used only for array plumbing/vectorisation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MatlabMT19937", "matlab_randi", "matlab_user_positions"]

_N = 624
_M = 397
_MATRIX_A = np.uint32(0x9908B0DF)
_UPPER = np.uint32(0x80000000)
_LOWER = np.uint32(0x7FFFFFFF)
_MT_DEFAULT_SEED = 5489  # MATLAB's Seed 0 maps here


class MatlabMT19937:
    """MT19937 seeded the way MATLAB's RandStream('mt19937ar','Seed',s) seeds it."""

    def __init__(self, seed: int = 0):
        seed = int(seed)
        if seed == 0:
            seed = _MT_DEFAULT_SEED
        mt = np.empty(_N, dtype=np.uint32)
        mt[0] = np.uint32(seed & 0xFFFFFFFF)
        for i in range(1, _N):
            prev = int(mt[i - 1])
            mt[i] = np.uint32((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)
        self._mt = mt
        self._mti = _N  # force a twist on first use

    def _twist(self) -> None:
        mt = self._mt
        old = mt.copy()
        # y[k], k in [0, N-1), is always built from two pre-twist words.
        y = (old[:-1] & _UPPER) | (old[1:] & _LOWER)
        tw = (y >> np.uint32(1)) ^ np.where(y & np.uint32(1), _MATRIX_A, np.uint32(0)).astype(np.uint32)
        K = _N - _M  # 227
        # k in [0, K): partner mt[k+M] is still pre-twist.
        mt[:K] = old[_M:] ^ tw[:K]
        # k in [K, N-1): partner is mt[k-K], already rewritten.  That is a shift-by-K
        # recurrence, so it must be walked in chunks of K -- doing it in one numpy
        # assignment would read stale words for k >= 2K.
        lo = K
        while lo < _N - 1:
            hi = min(lo + K, _N - 1)
            mt[lo:hi] = mt[lo - K : hi - K] ^ tw[lo:hi]
            lo = hi
        # k = N-1: pairs the last pre-twist word with the freshly written mt[0].
        y_last = (old[_N - 1] & _UPPER) | (mt[0] & _LOWER)
        tw_last = (y_last >> np.uint32(1)) ^ (_MATRIX_A if (y_last & np.uint32(1)) else np.uint32(0))
        mt[_N - 1] = mt[_M - 1] ^ np.uint32(tw_last)
        self._mti = 0

    def genrand_uint32(self, n: int) -> np.ndarray:
        """Return the next ``n`` tempered 32-bit outputs."""
        out = np.empty(int(n), dtype=np.uint32)
        filled = 0
        while filled < out.size:
            if self._mti >= _N:
                self._twist()
            take = min(_N - self._mti, out.size - filled)
            out[filled : filled + take] = self._mt[self._mti : self._mti + take]
            self._mti += take
            filled += take
        y = out
        y ^= y >> np.uint32(11)
        y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
        y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
        y ^= y >> np.uint32(18)
        return y

    def rand(self, n: int) -> np.ndarray:
        """Return the next ``n`` doubles in [0,1) with 53-bit resolution (MATLAB `rand`)."""
        n = int(n)
        w = self.genrand_uint32(2 * n).astype(np.uint64)
        a = w[0::2] >> np.uint64(5)   # 27 bits
        b = w[1::2] >> np.uint64(6)   # 26 bits
        return (a.astype(np.float64) * 67108864.0 + b.astype(np.float64)) * (1.0 / 9007199254740992.0)


def matlab_randi(stream: MatlabMT19937, lo: int, hi: int, n: int) -> np.ndarray:
    """``randi(stream, [lo, hi], n, 1)`` as an int64 array of length ``n``."""
    lo, hi, n = int(lo), int(hi), int(n)
    if hi < lo:
        raise ValueError(f"randi range [{lo}, {hi}] is empty")
    span = hi - lo + 1
    if span > (1 << 53):
        raise ValueError(
            "range wider than 2**53; MATLAB's randi is only verified bit-exact "
            "for spans representable in a 53-bit double"
        )
    return (np.floor(stream.rand(n) * span).astype(np.int64) + lo)


def matlab_user_positions(x_min, x_max, y_min, y_max, num_users, seed=0) -> np.ndarray:
    """Bit-exact port of SINREvaluation.m::generate_user_positions (isolated-stream branch).

    Returns an ``(num_users, 2)`` int64 array of ``(x, y)`` positions.

    Mirrors MATLAB's ``x_min = max(1, x_min); y_min = max(1, y_min)`` clamp, so a
    0..2000 x 0..1500 world draws ``randi([1,2000])`` and ``randi([1,1500])``.
    The clamp is idempotent, so pre-clamped bounds are safe to pass.
    """
    num_users = int(num_users)
    x_lo = max(1, int(x_min))
    y_lo = max(1, int(y_min))
    stream = MatlabMT19937(seed)
    xs = matlab_randi(stream, x_lo, int(x_max), num_users)   # whole x column first ...
    ys = matlab_randi(stream, y_lo, int(y_max), num_users)   # ... then the y column
    return np.column_stack((xs, ys))


if __name__ == "__main__":  # self-test against the MATLAB ground truth
    _EXPECT_X = [1630, 1812, 254, 1827, 1265, 196, 557, 1094]
    _EXPECT_Y = [947, 533, 1496, 337, 979, 908, 581, 214]
    _EXPECT_SHA = "5faabcecb3c4a0a48a5767b31e8f83cdaf8c5052f168a44dfbaf10bd7a52ffeb"
    import hashlib

    p = matlab_user_positions(0, 2000, 0, 1500, 1000, seed=0)
    assert p.shape == (1000, 2), p.shape
    assert p[:8, 0].tolist() == _EXPECT_X, p[:8, 0].tolist()
    assert p[:8, 1].tolist() == _EXPECT_Y, p[:8, 1].tolist()
    assert hashlib.sha256(p.tobytes()).hexdigest() == _EXPECT_SHA
    # MATLAB rand reference values (rng(0) and rng(1), first draw)
    assert abs(MatlabMT19937(0).rand(1)[0] - 0.8147236863931789) < 1e-16
    assert abs(MatlabMT19937(1).rand(1)[0] - 0.4170220047025740) < 1e-16
    print("ppo.matlab_rng self-test OK: 1000 MATLAB user positions reproduced bit-exactly")
