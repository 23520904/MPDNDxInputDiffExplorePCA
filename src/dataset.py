# dataset.py
#
# Polytopic dataset generation for the Explore Polytope algorithm.
#
# This is the MOST CRITICAL module in the pipeline. It defines
# the feature vectors that PCA/KMeans will evaluate.
#
# Key design decisions (see implementation_plan.md §5, Module B & C):
#   - Single-key construction: all k+1 ciphertexts share the same key
#     per sample, following PDND Algorithm 2.
#   - Unsupervised: no labels Y. Every sample is a "real" polytopic
#     encryption — no random/negative class.
#   - Supports two representations (R1 and R2) switchable at runtime
#     for ablation experiments.
#
# References:
#   Paper A — "A Novel Approach to Construct a Good Dataset for
#              Differential-Neural Cryptanalysis"
#   Paper B — "Polytopic Neural Differential Cryptanalysis (PDND 2026)"

from __future__ import annotations

import numpy as np
from enum import Enum
from os import urandom
from typing import Protocol, Sequence

from crypto import speck


# ──────────────────────────────────────────────
# Representation enum
# ──────────────────────────────────────────────

class Representation(Enum):
    """Feature vector representation fed to PCA.

    R1 — Raw Concatenated Ciphertexts
        Φ₁ = bin(C₀ˡ) ‖ bin(C₀ʳ) ‖ bin(C₁ˡ) ‖ bin(C₁ʳ) ‖ … ‖ bin(Cₖˡ) ‖ bin(Cₖʳ)
        Dimension: 2(k+1) × WORD_SIZE

    R2 — Ciphertext Differences Only
        Φ₂ = bin(C₁ˡ⊕C₀ˡ) ‖ bin(C₁ʳ⊕C₀ʳ) ‖ … ‖ bin(Cₖˡ⊕C₀ˡ) ‖ bin(Cₖʳ⊕C₀ʳ)
        Dimension: 2k × WORD_SIZE
    """
    R1_RAW_CONCAT = "R1"
    R2_DIFF_ONLY = "R2"


# ──────────────────────────────────────────────
# Binary conversion (generalised)
# ──────────────────────────────────────────────

def convert_to_binary(
    word_arrays: list[np.ndarray],
    word_size: int = speck.WORD_SIZE,
) -> np.ndarray:
    """Convert a list of uint16 word arrays into a binary feature matrix.

    This is a generalisation of Gohr's ``convert_to_binary`` that works
    for an arbitrary number of ciphertext words (not just 4).

    Parameters
    ----------
    word_arrays : list[np.ndarray]
        Each element has shape ``(N,)`` and dtype ``np.uint16``.
        For a polytope of size *k*, there are ``2(k+1)`` word arrays
        (left and right halves of ``k+1`` ciphertexts).
    word_size : int
        Number of bits per word (default 16 for Speck32).

    Returns
    -------
    np.ndarray
        Binary feature matrix of shape ``(N, num_words × word_size)``
        with dtype ``np.uint8``.
    """
    num_words = len(word_arrays)
    n_samples = len(word_arrays[0])
    total_bits = num_words * word_size

    X = np.zeros((total_bits, n_samples), dtype=np.uint8)
    for i in range(total_bits):
        word_idx = i // word_size
        bit_offset = word_size - (i % word_size) - 1
        X[i] = (word_arrays[word_idx] >> bit_offset) & 1

    return X.transpose()


# ──────────────────────────────────────────────
# Cipher abstraction
# ──────────────────────────────────────────────

class CipherEngine(Protocol):
    """Protocol for a cipher that can produce polytopic datasets.

    Any cipher (Speck, Simon, Simeck) can be plugged in by implementing
    this protocol. The ``expand_key`` and ``encrypt`` signatures must
    match the vectorised NumPy convention used in Gohr's code.
    """

    WORD_SIZE: int
    NUM_KEY_WORDS: int

    def expand_key(
        self, key: np.ndarray, num_rounds: int,
    ) -> list[np.ndarray]: ...

    def encrypt(
        self,
        plaintext: tuple[np.ndarray, np.ndarray],
        round_keys: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]: ...


class SpeckEngine:
    """Concrete cipher engine for Speck32/64."""

    WORD_SIZE: int = speck.WORD_SIZE
    NUM_KEY_WORDS: int = speck.NUM_KEY_WORDS

    def expand_key(
        self, key: np.ndarray, num_rounds: int,
    ) -> list[np.ndarray]:
        return speck.expand_key(key, num_rounds)

    def encrypt(
        self,
        plaintext: tuple[np.ndarray, np.ndarray],
        round_keys: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        return speck.encrypt(plaintext, round_keys)


# ──────────────────────────────────────────────
# Core: make_polytope_dataset
# ──────────────────────────────────────────────

def make_polytope_dataset(
    polytope: tuple[int, ...],
    num_rounds: int,
    num_samples: int,
    cipher: CipherEngine | None = None,
    representation: Representation = Representation.R1_RAW_CONCAT,
) -> np.ndarray:
    """Generate the unsupervised polytopic dataset 𝒟(P).

    For each of *N* independent samples:
      1. Sample a fresh random key K
      2. Sample a fresh random plaintext P₀
      3. Compute C₀ = E_K(P₀)
      4. For each Δᵢ in the polytope, compute Cᵢ = E_K(P₀ ⊕ Δᵢ)
      5. Build the feature vector according to ``representation``

    Parameters
    ----------
    polytope : tuple[int, ...]
        Canonical (sorted) tuple of *k* distinct nonzero input differences.
        Each difference is a ``block_size``-bit integer.
    num_rounds : int
        Number of cipher rounds.
    num_samples : int
        Number of independent (key, plaintext) pairs, i.e. *N*.
    cipher : CipherEngine or None
        Cipher to use.  Defaults to :class:`SpeckEngine`.
    representation : Representation
        Which feature representation to construct (R1 or R2).

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(N, d)`` with dtype ``np.uint8``,
        where ``d`` depends on the representation and polytope size *k*.
        - R1: ``d = 2(k+1) × WORD_SIZE``
        - R2: ``d = 2k × WORD_SIZE``
    """
    if cipher is None:
        cipher = SpeckEngine()

    k = len(polytope)
    ws = cipher.WORD_SIZE
    mask = 2**ws - 1
    n = num_samples

    # ── Step 1: Generate N random keys ──
    keys = np.frombuffer(
        urandom(2 * cipher.NUM_KEY_WORDS * n),
        dtype=np.uint16,
    ).reshape(cipher.NUM_KEY_WORDS, -1)
    round_keys = cipher.expand_key(keys, num_rounds)

    # ── Step 2: Generate N random base plaintexts ──
    p0_left = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    p0_right = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    # ── Step 3: Encrypt base plaintext ──
    c0_left, c0_right = cipher.encrypt((p0_left, p0_right), round_keys)

    # ── Step 4: Encrypt perturbed plaintexts ──
    ci_lefts: list[np.ndarray] = []
    ci_rights: list[np.ndarray] = []

    for diff in polytope:
        diff_left = (diff >> ws) & mask
        diff_right = diff & mask
        pi_left = p0_left ^ diff_left
        pi_right = p0_right ^ diff_right
        ci_left, ci_right = cipher.encrypt(
            (pi_left, pi_right), round_keys,
        )
        ci_lefts.append(ci_left)
        ci_rights.append(ci_right)

    # ── Step 5: Build feature vector ──
    if representation == Representation.R1_RAW_CONCAT:
        # Φ₁ = bin(C₀ˡ) ‖ bin(C₀ʳ) ‖ bin(C₁ˡ) ‖ bin(C₁ʳ) ‖ …
        word_arrays: list[np.ndarray] = [c0_left, c0_right]
        for i in range(k):
            word_arrays.append(ci_lefts[i])
            word_arrays.append(ci_rights[i])

    elif representation == Representation.R2_DIFF_ONLY:
        # Φ₂ = bin(C₁ˡ⊕C₀ˡ) ‖ bin(C₁ʳ⊕C₀ʳ) ‖ …
        word_arrays = []
        for i in range(k):
            word_arrays.append(ci_lefts[i] ^ c0_left)
            word_arrays.append(ci_rights[i] ^ c0_right)

    else:
        raise ValueError(f"Unknown representation: {representation}")

    return convert_to_binary(word_arrays, word_size=ws)


# ──────────────────────────────────────────────
# Convenience: compute feature dimension
# ──────────────────────────────────────────────

def feature_dimension(
    polytope_size: int,
    word_size: int = speck.WORD_SIZE,
    representation: Representation = Representation.R1_RAW_CONCAT,
) -> int:
    """Return the feature dimension *d* for a given polytope size and representation.

    Useful for setting the eigenvalue baseline ``λ_base = 1/d``.
    """
    if representation == Representation.R1_RAW_CONCAT:
        return 2 * (polytope_size + 1) * word_size
    elif representation == Representation.R2_DIFF_ONLY:
        return 2 * polytope_size * word_size
    else:
        raise ValueError(f"Unknown representation: {representation}")