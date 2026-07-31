"""distributional_explore.py
============================

Distributional-Utility Score (DUS) explorer for polytopic difference candidates
in PDND (Polytopic Differential Neural Distinguishers) for SPECK32/64.

This module is the **primary candidate-ranking path** and deliberately avoids
PCA, K-means, silhouette score, t-SNE, UMAP, and spectral clustering as core
scoring mechanisms.  Those legacy tools live in ``explore_legacy_pca_kmeans``
(see ``explore.py`` / ``poly_explore.py``) and may be used as optional
diagnostic utilities only.

Public API
----------
linear_time_mmd2        -- unbiased linear-time MMD² with RBF kernel
random_fourier_features -- RFF approximation of RBF feature map
hsic_rff                -- O(n·D) HSIC estimate via RFF
interaction_residual    -- heuristic ≥3-way dependence proxy
dus_score               -- composite Distributional-Utility Score
ParetoArchive           -- Pareto front helper (non-dominated archive)
explore_dus             -- Pareto hill-climbing search over polytope candidates
make_quadruples_adapter -- thin bridge to PolytopicQuadrupleGenerator

References
----------
* Gretton et al., "A Kernel Two-Sample Test," JMLR 2012.
* Mirzaali et al., "Improved Polytopic Differential Neural Distinguishers..."
* Seok & Lee, input-difference exploration (Algorithm 2).
"""

from __future__ import annotations

import sys
import os
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# A polytope difference tuple is 3 ints: (delta1, delta2, delta3).
DeltaTuple = Tuple[int, int, int]
# A quadruple of (n, bits) float32 arrays, +-1-encoded.
QuadrupleArrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# ===========================================================================
# §2a  Separability — linear-time MMD²
# ===========================================================================

def linear_time_mmd2(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Unbiased linear-time MMD² estimator with an RBF kernel.

    Uses the block estimator of Gretton et al. (2012), §6, which runs in
    O(n) time (vs O(n²) for the full U-statistic).  Requires n to be even.


    Parameters
    ----------
    X, Y : (n, d) float32 arrays
        Samples from the two distributions.  Must share the same n (even).
    gamma : float, optional
        RBF bandwidth γ (used as exp(-γ‖·‖²)).  Defaults to 1/d (median
        heuristic approximation for ±1-encoded bit vectors).
    rng : numpy Generator, optional
        Random state; created fresh if not provided.

    Returns
    -------
    mmd2_estimate : float
        Unbiased MMD² estimate.  Can be slightly negative for close distributions
        due to the unbiased construction — this is expected and correct.
    standard_error : float
        Standard error of the estimator (useful for diagnostics).
    """
    rng = rng or np.random.default_rng()
    n = X.shape[0]
    assert Y.shape[0] == n and n % 2 == 0, (
        f"X and Y must have equal, even sample count; got n={n}, Y.shape[0]={Y.shape[0]}"
    )
    gamma = gamma if gamma is not None else (1.0 / X.shape[1])

    def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Vectorised RBF kernel k(a_i, b_i) for paired rows."""
        diff = a - b
        return np.exp(-gamma * np.einsum("ij,ij->i", diff, diff))

    idx = np.arange(n).reshape(n // 2, 2)
    x1, x2 = X[idx[:, 0]], X[idx[:, 1]]
    y1, y2 = Y[idx[:, 0]], Y[idx[:, 1]]

    h = k(x1, x2) + k(y1, y2) - k(x1, y2) - k(x2, y1)
    return float(h.mean()), float(h.std(ddof=1) / np.sqrt(len(h)))


# ===========================================================================
# §2b  Higher-order interaction residual — HSIC via RFF
# ===========================================================================

def random_fourier_features(
    X: np.ndarray,
    D: int,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random Fourier Feature (RFF) approximation of an RBF kernel embedding.

    Bochner's theorem: k(x, y) ≈ φ(x)·φ(y) where
        φ(x) = sqrt(2/D) cos(Wx + b),  W ~ N(0, 2γ),  b ~ Uniform(0, 2π).

    Parameters
    ----------
    X : (n, d) float32 array
    D : int
        Number of random features (approximation dimension).
    gamma : float
        RBF bandwidth.
    rng : numpy Generator

    Returns
    -------
    Z : (n, D) float32 array — the RFF embedding.
    """
    n, d = X.shape
    W = rng.normal(scale=np.sqrt(2.0 * gamma), size=(d, D))
    b = rng.uniform(0.0, 2.0 * np.pi, size=D)
    return np.sqrt(2.0 / D) * np.cos(X @ W + b)


def hsic_rff(
    X: np.ndarray,
    Y: np.ndarray,
    D: int = 256,
    gamma_x: Optional[float] = None,
    gamma_y: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Linear-time HSIC estimate between feature blocks X and Y.

    Uses random Fourier features so complexity is O(n·D) instead of O(n²).
    The estimator is the empirical Hilbert-Schmidt norm of the cross-covariance
    operator after centering, approximated via RFF:

        HSIC_RFF ≈ ‖(Z_x - μ_x)ᵀ (Z_y - μ_y) / n‖_F²

    Parameters
    ----------
    X : (n, dx) float array
    Y : (n, dy) float array
    D : int
        RFF dimension per variable.
    gamma_x, gamma_y : float, optional
        RBF bandwidths; default to 1/dx and 1/dy respectively.
    rng : numpy Generator, optional

    Returns
    -------
    float — HSIC estimate (non-negative; ≈0 for independent variables).
    """
    rng = rng or np.random.default_rng()
    n = X.shape[0]
    gamma_x = gamma_x if gamma_x is not None else (1.0 / X.shape[1])
    gamma_y = gamma_y if gamma_y is not None else (1.0 / Y.shape[1])

    Zx = random_fourier_features(X, D, gamma_x, rng)
    Zy = random_fourier_features(Y, D, gamma_y, rng)

    # Centre the features (equivalent to applying the centering kernel H)
    Zx -= Zx.mean(axis=0, keepdims=True)
    Zy -= Zy.mean(axis=0, keepdims=True)

    # Frobenius norm squared of the empirical cross-covariance
    C = (Zx.T @ Zy) / n
    return float(np.sum(C * C))


def interaction_residual(
    C0: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    C3: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Heuristic higher-order interaction proxy for positive-class data.

    Computes I(Δ) = HSIC(C0, [C1‖C2‖C3]) − [HSIC(C0,C1) + HSIC(C0,C2) + HSIC(C0,C3)]

    NOTE: This is a **heuristic decomposition**, not an exact information-theoretic
    identity.  It is a cheap, directional proxy for "dependence not explained by
    pairwise structure," used only for ranking candidates, not as a rigorous
    interaction measure.  A positive value indicates genuine ≥3-way dependence
    beyond what the pairwise terms already capture — the signal that PDND's own
    ablation (Table 5) shows the network exploits.

    Parameters
    ----------
    C0, C1, C2, C3 : (n, bits) float arrays — positive-class ciphertext blocks.
    rng : numpy Generator, optional

    Returns
    -------
    float — I(Δ); positive => genuine ≥3-way dependence detected.
    """
    rng = rng or np.random.default_rng()
    rest = np.concatenate([C1, C2, C3], axis=1)
    hsic_total = hsic_rff(C0, rest, rng=rng)
    hsic_pairs = (
        hsic_rff(C0, C1, rng=rng)
        + hsic_rff(C0, C2, rng=rng)
        + hsic_rff(C0, C3, rng=rng)
    )
    return hsic_total - hsic_pairs


# ===========================================================================
# §3  Composite Distributional-Utility Score
# ===========================================================================

def dus_score(
    pos_quadruple: QuadrupleArrays,
    neg_quadruple: QuadrupleArrays,
    delta_tuple: DeltaTuple,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.05),
    n_boot: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, Dict]:
    """Composite Distributional-Utility Score for a polytope candidate.

    Score components:
    (a) MMD²   — linear-time separability between pos and neg ciphertext distributions.
    (b) I(Δ)   — heuristic ≥3-way interaction residual (positive-class only).
    (c) Stability — mean / (1 + std) over B bootstrap resamples of (a)+(b).
    (d) HW prior — subtract weighted Hamming-weight penalty Σ popcount(δᵢ).

    DUS = mean_raw + w3 · stability − w4 · hw_penalty

    Parameters
    ----------
    pos_quadruple : tuple (C0, C1, C2, C3)
        Each (n, bits) float32 array, ±1-encoded, label==1 samples.
    neg_quadruple : tuple (C0n, C1n, C2n, C3n)
        Each (n, bits) float32 array, ±1-encoded, label==0 samples.
        MUST be generated using a fixed neg polytope difference (not random
        noise) — consistent with Algorithm 1 of Mirzaali et al., so that DUS
        measures "candidate vs fixed-negative polytope" matching what training
        actually optimises.
    delta_tuple : (delta1, delta2, delta3) ints
        The candidate positive polytope as integers, for HW penalty.
    weights : (w1, w2, w3, w4)
        w1: MMD² weight, w2: interaction weight, w3: stability multiplier,
        w4: HW penalty weight.
    n_boot : int
        Number of bootstrap resamples for stability estimation.
    rng : numpy Generator, optional

    Returns
    -------
    dus_value : float
    components : dict with keys mmd2_mean, stability, hw_penalty, raw_scores.
    """
    rng = rng or np.random.default_rng()
    w1, w2, w3, w4 = weights
    C0, C1, C2, C3 = pos_quadruple
    C0n, C1n, C2n, C3n = neg_quadruple
    n = C0.shape[0]

    # Flattened 4-block vectors for MMD²
    X_pos = np.concatenate([C0, C1, C2, C3], axis=1)
    X_neg = np.concatenate([C0n, C1n, C2n, C3n], axis=1)

    raw: List[float] = []
    for _ in range(n_boot):
        # Bootstrap resample — ensure even count for linear_time_mmd2
        size = n - (n % 2)
        idx = rng.choice(n, size=size, replace=True)
        mmd2, _ = linear_time_mmd2(X_pos[idx], X_neg[idx], rng=rng)
        interaction = interaction_residual(
            C0[idx], C1[idx], C2[idx], C3[idx], rng=rng
        )
        raw.append(w1 * mmd2 + w2 * interaction)

    raw_arr = np.array(raw, dtype=np.float64)
    mean_raw = float(raw_arr.mean())
    std_raw = float(raw_arr.std(ddof=1)) if n_boot > 1 else 0.0
    stability = mean_raw / (1.0 + std_raw)

    hw_penalty = sum(bin(int(d)).count("1") for d in delta_tuple)

    dus = mean_raw + w3 * stability - w4 * hw_penalty

    return dus, {
        "mmd2_mean": mean_raw,
        "stability": stability,
        "hw_penalty": hw_penalty,
        "raw_scores": raw_arr.tolist(),
    }


# ===========================================================================
# §4 Pareto archive
# ===========================================================================

class ParetoArchive:
    """Non-dominated archive over two objectives: (DUS up, HW down).

    A point p dominates q if p.dus >= q.dus AND p.hw <= q.hw, with at least
    one strict inequality.  The archive keeps only non-dominated points.

    Attributes
    ----------
    archive : list of dicts with keys: delta_tuple, dus, hw_penalty, components.
    """

    def __init__(self) -> None:
        self.archive: List[Dict] = []

    def _dominates(self, a: Dict, b: Dict) -> bool:
        """Return True if `a` weakly dominates `b` on both objectives."""
        return (
            a["dus"] >= b["dus"]
            and a["hw_penalty"] <= b["hw_penalty"]
            and (a["dus"] > b["dus"] or a["hw_penalty"] < b["hw_penalty"])
        )

    def insert(
        self,
        delta_tuple: DeltaTuple,
        dus: float,
        hw_penalty: int,
        components: Optional[Dict] = None,
    ) -> bool:
        """Attempt to insert a new point into the archive.

        The point is inserted only if it is not dominated by any existing
        member.  Existing members dominated by the new point are pruned.

        Returns
        -------
        bool — True if the point was inserted.
        """
        new = {
            "delta_tuple": delta_tuple,
            "dus": dus,
            "hw_penalty": hw_penalty,
            "components": components or {},
        }
        # Check whether the new point is dominated by any archive member
        for existing in self.archive:
            if self._dominates(existing, new):
                return False  # new point is dominated; reject
        # Prune archive members dominated by the new point
        self.archive = [e for e in self.archive if not self._dominates(new, e)]
        self.archive.append(new)
        return True

    def sorted_by_dus(self) -> List[Dict]:
        """Return archive members sorted by DUS descending."""
        return sorted(self.archive, key=lambda x: x["dus"], reverse=True)

    def random_member(self, rng: np.random.Generator) -> Optional[Dict]:
        """Return a random archive member, or None if archive is empty."""
        if not self.archive:
            return None
        idx = rng.integers(0, len(self.archive))
        return self.archive[int(idx)]

    def __len__(self) -> int:
        return len(self.archive)

    def __repr__(self) -> str:
        return f"ParetoArchive(size={len(self.archive)})"


# ===========================================================================
# Internal helpers — low-HW polytope generation & bit-flip proposals
# ===========================================================================

def _generate_low_hw_delta(
    bit_size: int,
    max_hw: int,
    rng: np.random.Generator,
    existing_pool: Optional[set] = None,
) -> int:
    """Draw a random nonzero integer with HW in [1, max_hw], biased toward low HW."""
    weights = [1.0 / i for i in range(1, max_hw + 1)]
    w_total = sum(weights)
    probs = np.array([w / w_total for w in weights], dtype=np.float64)
    for _ in range(1000):  # finite retry cap
        hw_val = int(rng.choice(max_hw, p=probs)) + 1  # 1..max_hw
        positions = rng.choice(bit_size, size=hw_val, replace=False)
        val = int(sum(1 << int(p) for p in positions))
        if val == 0:
            continue
        if existing_pool is None or val not in existing_pool:
            return val
    # Fallback: single active bit
    return 1 << int(rng.integers(0, bit_size))


def _generate_low_hw_polytope(
    bit_size: int,
    max_hw: int,
    rng: np.random.Generator,
    pool: Optional[set] = None,
) -> DeltaTuple:
    """Generate a 3-tuple of distinct low-HW differences."""
    local: set = set()
    diffs = []
    for _ in range(3):
        d = _generate_low_hw_delta(bit_size, max_hw, rng, local)
        local.add(d)
        diffs.append(d)
    candidate: DeltaTuple = (diffs[0], diffs[1], diffs[2])
    if pool is not None:
        pool.add(candidate)
    return candidate


def _bit_flip_neighbor(
    delta_tuple: DeltaTuple,
    bit_size: int,
    rng: np.random.Generator,
    prefer_low_hw: bool = True,
) -> DeltaTuple:
    """Propose a neighbor by flipping one bit of one delta in the tuple.

    When prefer_low_hw=True, bits that are currently 1 (would decrease HW)
    are sampled with 2× the weight of bits that are 0 (would increase HW),
    provided the selected delta has HW > 1 (to avoid producing delta=0).
    """
    which = int(rng.integers(0, 3))
    d = delta_tuple[which]
    hw = bin(d).count("1")

    if prefer_low_hw and hw > 1:
        # Build weighted position list: set bits (1->0) weight 2, unset (0->1) weight 1
        positions = list(range(bit_size))
        weights_arr = np.array(
            [2.0 if (d >> pos) & 1 else 1.0 for pos in positions], dtype=np.float64
        )
        weights_arr /= weights_arr.sum()
        bit_pos = int(rng.choice(bit_size, p=weights_arr))
    else:
        bit_pos = int(rng.integers(0, bit_size))

    new_d = d ^ (1 << bit_pos)
    # Avoid zero difference
    if new_d == 0:
        new_d = 1 << int(rng.integers(0, bit_size))

    new_tuple = list(delta_tuple)
    new_tuple[which] = new_d
    return (new_tuple[0], new_tuple[1], new_tuple[2])


# ===========================================================================
# §5 Adapter — bridge to PolytopicQuadrupleGenerator
# ===========================================================================

def make_quadruples_adapter(
    pos_delta_tuple: DeltaTuple,
    neg_delta_tuple: DeltaTuple,
    rounds: int,
    n_samples: int,
    plain_bits: int = 32,
    key_bits: int = 64,
    use_gpu: bool = False,
) -> Tuple[QuadrupleArrays, QuadrupleArrays]:
    """Generate positive- and negative-class ciphertext quadruples.

    Reuses ``PolytopicQuadrupleGenerator`` with ``feature_mode='raw'`` so that
    the distribution exactly matches what training uses.  Splits the mixed batch
    by the generator's ``Y`` mask (Y==1 → positive class, Y==0 → negative class).

    The negative class uses a **fixed polytope** ``neg_delta_tuple`` (not random
    noise), consistent with Algorithm 1 of Mirzaali et al.: both label classes
    are generated from distinct, concrete polytope differences so that DUS
    measures the same "positive-polytope vs negative-polytope" contrast that
    the training distinguisher optimises.

    Parameters
    ----------
    pos_delta_tuple : (delta1, delta2, delta3) ints — candidate positive polytope.
    neg_delta_tuple : (delta1, delta2, delta3) ints — fixed negative polytope
        (e.g. the NEG_DELTAS from ``train.py``'s configuration, converted to ints;
        NOT random noise — see docstring above).
    rounds : int — number of cipher rounds.
    n_samples : int — total samples to generate (mixed pos + neg).
    plain_bits, key_bits : int — block / key size (SPECK32/64 defaults).
    use_gpu : bool — passed to the generator (CPU-only by default for portability).

    Returns
    -------
    pos_quad : (C0, C1, C2, C3) — positive-class arrays, +-1 float32, shape (n_pos, plain_bits).
    neg_quad : (C0n, C1n, C2n, C3n) — negative-class arrays, +-1 float32.

    Notes
    -----
    The split is by Y mask, so n_pos + n_neg == n_samples but n_pos ≈ n_neg
    (each sample is labelled 0 or 1 with equal probability by the generator).
    If the split is very unbalanced for a given small n_samples, increase
    n_samples or resample.
    """
    # Lazy import so this module has no hard TF dependency at module-load time.
    _src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

    import data_utils.PolytopicQadrupleGenerator as pqg
    import speck32.cipher as speck

    gen = pqg.PolytopicQuadrupleGenerator(
        encryption_function=speck.encrypt_wrapper,
        plain_bits=plain_bits,
        key_bits=key_bits,
        nr=rounds,
        pos_diffs=pos_delta_tuple,   # label-1 polytope (candidate under evaluation)
        neg_diffs=neg_delta_tuple,   # label-0 polytope (fixed reference; NOT random)
        related_key=False,
        feature_mode='raw',          # X = [C0, C1, C2, C3] — no diff features baked in
        n_samples=n_samples,
        batch_size=n_samples,        # generate all in one batch
        use_gpu=use_gpu,
        to_float32=True,             # converts {0,1} -> {-1,1} inside the generator
    )

    X, Y = gen[0]   # X: (n, 4*plain_bits), Y: (n,) uint8

    # Extract individual ciphertext blocks (each plain_bits wide)
    pb = plain_bits
    C0_all = X[:, 0 * pb: 1 * pb]
    C1_all = X[:, 1 * pb: 2 * pb]
    C2_all = X[:, 2 * pb: 3 * pb]
    C3_all = X[:, 3 * pb: 4 * pb]

    Y_flat = Y.ravel()
    pos_mask = Y_flat == 1
    neg_mask = Y_flat == 0

    pos_quad: QuadrupleArrays = (
        C0_all[pos_mask], C1_all[pos_mask],
        C2_all[pos_mask], C3_all[pos_mask],
    )
    neg_quad: QuadrupleArrays = (
        C0_all[neg_mask], C1_all[neg_mask],
        C2_all[neg_mask], C3_all[neg_mask],
    )
    return pos_quad, neg_quad


# ===========================================================================
# §4 Main search loop — Pareto hill-climbing
# ===========================================================================

def explore_dus(
    delta_pool_init: List[DeltaTuple],
    generate_quadruples_fn: Callable[
        [DeltaTuple, DeltaTuple, int, int], Tuple[QuadrupleArrays, QuadrupleArrays]
    ],
    rounds: int,
    n_samples: int,
    neg_delta_fixed: Optional[DeltaTuple] = None,
    budget: int = 200,
    bit_size: int = 32,
    max_hw: int = 3,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.05),
    n_boot: int = 8,
    explore_frac: float = 0.2,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Pareto-archive hill-climbing search over polytope candidates.

    Maintains a Pareto front on objectives (DUS up, HW down).  At each iteration,
    a new candidate is proposed either by:
      - (1 - explore_frac) fraction: bit-flipping a random archive member
        (exploitation, biased toward low-HW flips), or
      - (explore_frac) fraction: drawing a fresh low-HW polytope (exploration).
    The candidate is evaluated via ``generate_quadruples_fn`` + ``dus_score``,
    then inserted into the archive if non-dominated.

    Parameters
    ----------
    delta_pool_init : list of (delta1, delta2, delta3) int triples
        Initial seed candidates (low-HW biased recommended).
    generate_quadruples_fn : callable(pos_delta, neg_delta, rounds, n_samples)
        Must return (pos_quad, neg_quad) as ±1-encoded float32 numpy arrays.
        Pass ``make_quadruples_adapter`` for the standard SPECK32 pipeline, or
        a custom function for testing / other ciphers.
    rounds : int
        Number of cipher rounds for encryption.
    n_samples : int
        Samples per DUS evaluation.
    neg_delta_fixed : (delta1, delta2, delta3) ints, optional
        Fixed negative-class polytope.  If None, defaults to the first element
        of ``delta_pool_init`` as reference.  Must NOT be random noise — see
        ``make_quadruples_adapter`` docstring.
    budget : int
        Maximum number of DUS evaluations (including the initial pool).
    bit_size : int
        Bit width of each difference element (32 for SPECK32).
    max_hw : int
        Maximum Hamming weight for fresh random draws during exploration.
    weights : (w1, w2, w3, w4)
        Forwarded to ``dus_score``.
    n_boot : int
        Bootstrap resamples per DUS evaluation.
    explore_frac : float
        Fraction of iterations that draw fresh random candidates (vs. bit-flips).
    rng : numpy Generator, optional
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    list of dicts, each with keys: delta_tuple, dus, hw_penalty, components.
    Sorted by DUS descending (the Pareto front).
    """
    rng = rng or np.random.default_rng()
    archive = ParetoArchive()
    evaluated: set = set()   # deduplicate by delta_tuple

    # Resolve the fixed negative polytope reference
    if neg_delta_fixed is None:
        if not delta_pool_init:
            raise ValueError(
                "delta_pool_init must be non-empty when neg_delta_fixed is None."
            )
        neg_delta_fixed = delta_pool_init[0]

    def _evaluate(delta: DeltaTuple) -> Tuple[float, Dict]:
        pos_q, neg_q = generate_quadruples_fn(
            delta, neg_delta_fixed, rounds, n_samples
        )
        return dus_score(pos_q, neg_q, delta, weights=weights, n_boot=n_boot, rng=rng)

    n_eval = 0

    # ── Phase 1: Evaluate initial pool ──────────────────────────────────────
    if verbose:
        print("=" * 72)
        print("  DUS Explorer — Pareto Hill-Climbing")
        print(f"  budget={budget}  n_samples={n_samples}  rounds={rounds}")
        print("=" * 72)

    for delta in delta_pool_init:
        if n_eval >= budget:
            break
        if delta in evaluated:
            continue
        evaluated.add(delta)
        try:
            score, comps = _evaluate(delta)
        except Exception as exc:
            logger.warning("Evaluation failed for %s: %s", delta, exc)
            continue
        hw = comps["hw_penalty"]
        inserted = archive.insert(delta, score, hw, comps)
        n_eval += 1
        if verbose:
            tag = "archived" if inserted else "dominated"
            print(
                f"  [{n_eval:4d}/{budget}] {tag} | DUS={score:+.4f} "
                f"HW={hw} | {delta}"
            )

    # ── Phase 2: Hill-climbing until budget exhausted ────────────────────────
    pool_for_fresh: set = set(delta_pool_init)

    while n_eval < budget:
        use_explore = (rng.random() < explore_frac) or (len(archive) == 0)

        if use_explore:
            candidate = _generate_low_hw_polytope(bit_size, max_hw, rng, pool_for_fresh)
        else:
            member = archive.random_member(rng)
            if member is None:
                candidate = _generate_low_hw_polytope(bit_size, max_hw, rng, pool_for_fresh)
            else:
                candidate = _bit_flip_neighbor(
                    member["delta_tuple"], bit_size, rng, prefer_low_hw=True
                )

        if candidate in evaluated:
            continue  # Skip already-seen (saves one encryption call)

        evaluated.add(candidate)
        try:
            score, comps = _evaluate(candidate)
        except Exception as exc:
            logger.warning("Evaluation failed for %s: %s", candidate, exc)
            n_eval += 1
            continue

        hw = comps["hw_penalty"]
        inserted = archive.insert(candidate, score, hw, comps)
        n_eval += 1

        if verbose and (inserted or n_eval % 20 == 0):
            tag = "archived" if inserted else "rejected"
            print(
                f"  [{n_eval:4d}/{budget}] {tag} | DUS={score:+.4f} "
                f"HW={hw} | {candidate}"
            )

    if verbose:
        print(f"\n  Search complete. Pareto front size: {len(archive)}")
        print("=" * 72)

    return archive.sorted_by_dus()
