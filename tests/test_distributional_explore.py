"""tests/test_distributional_explore.py
=======================================

Unit tests for ``analysis.distributional_explore``.

All tests use only NumPy (no TF/SPECK required) — cipher-level integration is
tested via ``calibrate_dus.py``, not here.

Run with:
    cd src
    python -m pytest ../tests/test_distributional_explore.py -v

Or from repo root:
    python -m pytest tests/test_distributional_explore.py -v
"""

from __future__ import annotations

import sys
import os
import threading

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup so tests work from repo root or from src/
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for _p in [_SRC_DIR, os.path.join(_SRC_DIR, "analysis")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.distributional_explore import (
    linear_time_mmd2,
    random_fourier_features,
    hsic_rff,
    interaction_residual,
    dus_score,
    ParetoArchive,
    explore_dus,
    _generate_low_hw_polytope,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

RNG_SEED = 2024


def _rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


def _make_pm1(shape, rng) -> np.ndarray:
    """Uniform +/-1 float32 array -- mimics the generator's encoded bits."""
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=shape).astype(np.float32)


def _make_shifted(shape, rng, shift: float = 2.0) -> np.ndarray:
    """Clearly different distribution: shifted mean."""
    return (_make_pm1(shape, rng) + shift).astype(np.float32)


# ===========================================================================
# linear_time_mmd2
# ===========================================================================

class TestLinearTimeMmd2:

    def test_same_distribution_approx_zero(self):
        """MMD^2(X, X_fresh) should be approx 0 when both samples come from the same distribution."""
        rng = _rng()
        n, d = 1024, 32
        X = _make_pm1((n, d), rng)
        Y = _make_pm1((n, d), np.random.default_rng(RNG_SEED + 1))
        mmd2, se = linear_time_mmd2(X, Y, rng=_rng())
        assert abs(mmd2) < 5 * se + 0.05, (
            f"MMD2(same dist) should be approx 0, got {mmd2:.4f} +/- {se:.4f}"
        )

    def test_identical_arrays_zero(self):
        """MMD^2(X, X) with literally the same array object should be 0."""
        rng = _rng()
        n, d = 512, 16
        X = _make_pm1((n, d), rng)
        mmd2, _ = linear_time_mmd2(X, X, rng=_rng())
        assert abs(mmd2) < 1e-6, f"MMD2(X, X) should be 0, got {mmd2}"

    def test_different_distributions_large_positive(self):
        """MMD^2 should be clearly positive when Y has a shifted mean."""
        rng = _rng()
        n, d = 1024, 32
        X = _make_pm1((n, d), rng)
        Y = _make_shifted((n, d), np.random.default_rng(RNG_SEED + 2), shift=3.0)
        mmd2, se = linear_time_mmd2(X, Y, rng=_rng())
        assert mmd2 > 5 * se, (
            f"MMD2(different dist) should be >> 0, got {mmd2:.4f} +/- {se:.4f}"
        )
        assert mmd2 > 0.01, f"MMD2 too small: {mmd2}"

    def test_requires_even_n(self):
        """Should raise AssertionError for odd n."""
        rng = _rng()
        X = _make_pm1((11, 8), rng)
        Y = _make_pm1((11, 8), _rng())
        with pytest.raises(AssertionError):
            linear_time_mmd2(X, Y, rng=_rng())

    def test_requires_equal_n(self):
        """Should raise AssertionError when len(X) != len(Y)."""
        rng = _rng()
        X = _make_pm1((100, 8), rng)
        Y = _make_pm1((200, 8), rng)
        with pytest.raises(AssertionError):
            linear_time_mmd2(X, Y, rng=_rng())


# ===========================================================================
# random_fourier_features
# ===========================================================================

class TestRandomFourierFeatures:

    def test_output_shape(self):
        rng = _rng()
        n, d, D = 200, 32, 128
        X = _make_pm1((n, d), rng)
        Z = random_fourier_features(X, D, gamma=1.0 / d, rng=rng)
        assert Z.shape == (n, D), f"Expected ({n}, {D}), got {Z.shape}"

    def test_approximation_quality(self):
        """RFF kernel should approximate exact RBF on a pair of rows."""
        rng = _rng()
        n, d, D = 50, 8, 2048
        X = _make_pm1((n, d), rng)
        gamma = 1.0 / d
        Z = random_fourier_features(X, D, gamma=gamma, rng=rng)
        diff = X[0] - X[1]
        k_exact = float(np.exp(-gamma * float(np.dot(diff, diff))))
        k_approx = float(np.dot(Z[0], Z[1]))
        assert abs(k_approx - k_exact) < 0.15, (
            f"RFF approximation error too large: |{k_approx:.3f} - {k_exact:.3f}| = "
            f"{abs(k_approx - k_exact):.3f}"
        )


# ===========================================================================
# hsic_rff
# ===========================================================================

class TestHsicRff:

    def test_independent_noise_approx_zero(self):
        """HSIC(X, Y) approx 0 for independent X and Y."""
        rng = _rng()
        n, dx, dy = 2000, 32, 32
        X = rng.standard_normal((n, dx)).astype(np.float32)
        Y = np.random.default_rng(RNG_SEED + 3).standard_normal((n, dy)).astype(np.float32)
        hsic = hsic_rff(X, Y, D=256, rng=_rng())
        assert hsic < 5e-4, f"HSIC(independent) should be approx 0, got {hsic:.6f}"

    def test_dependent_clearly_positive(self):
        """HSIC(X, Y) should be clearly positive when Y is a linear function of X."""
        rng = _rng()
        n, d = 2000, 16
        X = rng.standard_normal((n, d)).astype(np.float32)
        W = rng.standard_normal((d, d)).astype(np.float32)
        Y = (X @ W).astype(np.float32)
        hsic = hsic_rff(X, Y, D=256, rng=_rng())
        assert hsic > 0.01, f"HSIC(Y=f(X)) should be clearly positive, got {hsic:.6f}"

    def test_non_negative(self):
        """HSIC estimate should be non-negative (it is a squared Frobenius norm)."""
        rng = _rng()
        n, d = 500, 8
        X = _make_pm1((n, d), rng)
        Y = _make_pm1((n, d), _rng())
        hsic = hsic_rff(X, Y, D=128, rng=_rng())
        assert hsic >= -1e-10, f"HSIC should be non-negative, got {hsic}"


# ===========================================================================
# interaction_residual
# ===========================================================================

class TestInteractionResidual:

    def test_runs_without_error(self):
        rng = _rng()
        n, bits = 256, 32
        C0 = _make_pm1((n, bits), rng)
        C1 = _make_pm1((n, bits), _rng())
        C2 = _make_pm1((n, bits), np.random.default_rng(1))
        C3 = _make_pm1((n, bits), np.random.default_rng(2))
        val = interaction_residual(C0, C1, C2, C3, rng=_rng())
        assert np.isfinite(val), f"interaction_residual returned non-finite: {val}"

    def test_independent_blocks_near_zero(self):
        """For four independent blocks, I(D) should be near zero."""
        rng = _rng()
        n, bits = 2000, 16
        C0 = rng.standard_normal((n, bits)).astype(np.float32)
        C1 = np.random.default_rng(10).standard_normal((n, bits)).astype(np.float32)
        C2 = np.random.default_rng(11).standard_normal((n, bits)).astype(np.float32)
        C3 = np.random.default_rng(12).standard_normal((n, bits)).astype(np.float32)
        val = interaction_residual(C0, C1, C2, C3, rng=_rng())
        # Should be small in magnitude for independent blocks
        assert abs(val) < 0.01, (
            f"interaction_residual(independent) should be near 0, got {val:.6f}"
        )


# ===========================================================================
# dus_score
# ===========================================================================

class TestDusScore:

    def _make_synthetic_quad(self, n: int, bits: int, rng, shift: float = 0.0):
        """Return (C0, C1, C2, C3) as +/-1 float32 arrays, optionally shifted."""
        return tuple(
            (_make_pm1((n, bits), rng) + shift).astype(np.float32)
            for _ in range(4)
        )

    def test_no_nans_end_to_end(self):
        """dus_score must return finite values on a small synthetic quadruple."""
        rng = _rng()
        n, bits = 256, 32
        pos_q = self._make_synthetic_quad(n, bits, rng, shift=0.5)
        neg_q = self._make_synthetic_quad(n, bits, np.random.default_rng(99), shift=-0.5)
        delta = (1, 2, 4)  # low HW
        score, comps = dus_score(pos_q, neg_q, delta, n_boot=4, rng=_rng())
        assert np.isfinite(score), f"DUS is not finite: {score}"
        assert all(np.isfinite(v) for v in [
            comps["mmd2_mean"], comps["stability"]
        ]), f"Non-finite in components: {comps}"
        assert len(comps["raw_scores"]) == 4, "Expected 4 bootstrap scores"

    def test_hw_penalty_lowers_score(self):
        """A high-HW delta_tuple should produce a lower DUS than low-HW, all else equal."""
        rng = _rng()
        n, bits = 256, 32
        pos_q = self._make_synthetic_quad(n, bits, rng, shift=0.3)
        neg_q = self._make_synthetic_quad(n, bits, np.random.default_rng(77), shift=-0.3)

        delta_low_hw = (1, 2, 4)                   # popcount = 1+1+1 = 3
        delta_high_hw = (0xFFFF, 0x7FFF, 0x3FFF)   # popcount = 16+15+14 = 45

        score_low, comps_low = dus_score(pos_q, neg_q, delta_low_hw, n_boot=4, rng=_rng())
        score_high, comps_high = dus_score(pos_q, neg_q, delta_high_hw, n_boot=4, rng=_rng())

        assert comps_low["hw_penalty"] < comps_high["hw_penalty"], (
            f"Expected low HW < high HW: "
            f"{comps_low['hw_penalty']} vs {comps_high['hw_penalty']}"
        )
        assert score_low > score_high, (
            f"Expected low-HW DUS ({score_low:.4f}) > high-HW DUS ({score_high:.4f})"
        )

    def test_components_keys_present(self):
        """dus_score must return all required component keys."""
        rng = _rng()
        n, bits = 128, 16
        pos_q = self._make_synthetic_quad(n, bits, rng)
        neg_q = self._make_synthetic_quad(n, bits, _rng())
        _, comps = dus_score(pos_q, neg_q, (1, 2, 4), n_boot=2, rng=_rng())
        for key in ("mmd2_mean", "stability", "hw_penalty", "raw_scores"):
            assert key in comps, f"Missing key '{key}' in components"


# ===========================================================================
# ParetoArchive
# ===========================================================================

class TestParetoArchive:

    def test_insert_non_dominated(self):
        arch = ParetoArchive()
        assert arch.insert((1, 2, 3), dus=0.5, hw_penalty=3)
        assert len(arch) == 1

    def test_dominated_point_rejected(self):
        arch = ParetoArchive()
        arch.insert((1, 2, 3), dus=0.8, hw_penalty=2)
        # (0.5, 3) is dominated by (0.8, 2): lower DUS AND higher HW
        inserted = arch.insert((4, 5, 6), dus=0.5, hw_penalty=3)
        assert not inserted, "Dominated point should be rejected"
        assert len(arch) == 1

    def test_dominates_existing_member(self):
        arch = ParetoArchive()
        arch.insert((1, 2, 3), dus=0.5, hw_penalty=3)
        # (0.8, 2) dominates (0.5, 3)
        inserted = arch.insert((4, 5, 6), dus=0.8, hw_penalty=2)
        assert inserted, "Dominating point should be inserted"
        assert len(arch) == 1

    def test_pareto_front_two_non_dominated(self):
        arch = ParetoArchive()
        arch.insert((1, 2, 3), dus=0.8, hw_penalty=3)
        arch.insert((4, 5, 6), dus=0.5, hw_penalty=1)
        assert len(arch) == 2

    def test_sorted_by_dus(self):
        arch = ParetoArchive()
        arch.insert((1, 2, 3), dus=0.3, hw_penalty=1)
        arch.insert((4, 5, 6), dus=0.8, hw_penalty=3)
        front = arch.sorted_by_dus()
        assert front[0]["dus"] >= front[1]["dus"], "Should be sorted descending by DUS"


# ===========================================================================
# explore_dus
# ===========================================================================

class TestExploreDus:
    """Verify explore_dus terminates within budget and returns a non-empty front."""

    @staticmethod
    def _synthetic_gen(pos_delta, neg_delta, rounds, n_samples):
        """Toy quadruple generator: returns +/-1 random arrays (no SPECK encryption)."""
        rng = np.random.default_rng(abs(hash(pos_delta)) & 0xFFFF_FFFF)
        n = min(n_samples, 256)
        bits = 32
        pos_q = tuple(rng.choice([-1.0, 1.0], size=(n, bits)).astype(np.float32) for _ in range(4))
        neg_q = tuple(rng.choice([-1.0, 1.0], size=(n, bits)).astype(np.float32) for _ in range(4))
        return pos_q, neg_q

    def test_returns_nonempty_pareto_front(self):
        """explore_dus must return at least one Pareto-front member."""
        rng = np.random.default_rng(RNG_SEED)
        init_pool = [
            _generate_low_hw_polytope(bit_size=32, max_hw=2, rng=rng)
            for _ in range(5)
        ]
        front = explore_dus(
            delta_pool_init=init_pool,
            generate_quadruples_fn=self._synthetic_gen,
            rounds=5,
            n_samples=256,
            budget=15,
            n_boot=2,
            verbose=False,
            rng=np.random.default_rng(RNG_SEED + 1),
        )
        assert len(front) > 0, "Pareto front should be non-empty"

    def test_terminates_within_budget(self):
        """explore_dus must not hang beyond a reasonable wall-clock time."""
        rng = np.random.default_rng(RNG_SEED)
        init_pool = [
            _generate_low_hw_polytope(bit_size=32, max_hw=2, rng=rng)
            for _ in range(3)
        ]
        result_holder = []

        def run():
            front = explore_dus(
                delta_pool_init=init_pool,
                generate_quadruples_fn=self._synthetic_gen,
                rounds=5,
                n_samples=128,
                budget=10,
                n_boot=2,
                verbose=False,
                rng=np.random.default_rng(RNG_SEED),
            )
            result_holder.append(front)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=60)
        assert not t.is_alive(), "explore_dus did not terminate within 60 seconds"
        assert result_holder, "explore_dus did not return a result"

    def test_front_sorted_by_dus(self):
        """Returned front should be sorted by DUS descending."""
        rng = np.random.default_rng(RNG_SEED)
        init_pool = [
            _generate_low_hw_polytope(bit_size=32, max_hw=2, rng=rng)
            for _ in range(5)
        ]
        front = explore_dus(
            delta_pool_init=init_pool,
            generate_quadruples_fn=self._synthetic_gen,
            rounds=5,
            n_samples=128,
            budget=12,
            n_boot=2,
            verbose=False,
            rng=np.random.default_rng(RNG_SEED + 2),
        )
        if len(front) > 1:
            for i in range(len(front) - 1):
                assert front[i]["dus"] >= front[i + 1]["dus"], (
                    "Front should be sorted by DUS descending"
                )
