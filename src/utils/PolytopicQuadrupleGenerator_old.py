"""
PolytopicQuadrupleGenerator
============================

Keras `Sequence` implementing the data-generation procedure described in:

    Mirzaali, Sadeghi, Bagheri - "Improved polytopic differential neural
    distinguishers for SIMON, SIMECK, and SPECK block ciphers" (2026),
    Algorithm 1 (single-key) and Algorithm 2 (related-key).

Unlike a "multi-pair" generator (which independently samples ONE difference
out of a list for each training row, producing a (P, P xor delta) PAIR),
this generator reproduces the paper's *polytope* construction:

  * pos_diffs = Delta = (Delta_1, Delta_2, Delta_3)   -> label 0
  * neg_diffs = delta = (delta_1, delta_2, delta_3)   -> label 1

For every sample we draw ONE random label Y_i and ONE random plaintext
anchor P0_i, then apply *all three* differences of the tuple selected by
Y_i simultaneously to build the plaintext quadruple

    (P0, P0^diff_1, P0^diff_2, P0^diff_3)

(single-key), or, in the related-key case, additionally derive three
related keys from a master key K0 using the key-difference tuple
(Algorithm 2):

    K0, K0^key_diff_1, K0^key_diff_2, K0^key_diff_3

Each of the 4 texts is encrypted (with its own key, in the related-key
case) to build the ciphertext quadruple (C0, C1, C2, C3), which -
concatenated and normalized to [-1, 1] exactly as described in the paper's
"Input and Normalization" section - is the model input X. Y is the label.
"""

from tensorflow.keras.utils import Sequence
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

DEFAULT_USE_GPU = cp is not None


def _int_to_bitarray(val, nbits, lib):
    """
    Convert val -> bit array length nbits in backend lib (np or cp).
    Returns uint8 array.

    `val` may be:
      - a single int (interpreted as an nbits-wide integer), or
      - a tuple/list of word-ints, e.g. (0x0040, 0x0) for a 32-bit SPECK
        difference given as (delta_x, delta_y) 16-bit words - matching the
        format used throughout speck.py (a=((0x0040,0x0), ...)). Words are
        concatenated MSB-first, each contributing nbits // len(val) bits.
      - a numpy/cupy array of bits (passed through / converted as-is).
    """
    if isinstance(val, (tuple, list)):
        nwords = len(val)
        word_bits = nbits // nwords
        combined = 0
        for w in val:
            combined = (combined << word_bits) | (int(w) & ((1 << word_bits) - 1))
        val = combined

    if isinstance(val, (int, np.integer)):
        bits = np.zeros(nbits, dtype=np.uint8)
        v = int(val)
        for i in range(nbits):
            bits[nbits - 1 - i] = (v >> i) & 1
        return lib.asarray(bits) if (cp is not None and lib is cp) else bits

    if cp is not None and isinstance(val, cp.ndarray) and lib is np:
        return cp.asnumpy(val).astype(np.uint8)

    if cp is not None and isinstance(val, np.ndarray) and lib is cp:
        return cp.asarray(val.astype(np.uint8))

    if isinstance(val, (np.ndarray,)) or (cp is not None and isinstance(val, cp.ndarray)):
        return val.astype(np.uint8)

    arr = np.asarray(val, dtype=np.uint8)
    return lib.asarray(arr) if (cp is not None and lib is cp) else arr


def _safe_encrypt(enc_fn, P, K, nr):
    """
    Call encryption function with automatic CPU <-> GPU fallback.
    Tries once, then falls back once. Cipher logic errors will still raise.
    """
    try:
        return enc_fn(P, K, nr)
    except Exception:
        if cp is None:
            raise
        if isinstance(P, cp.ndarray):
            return enc_fn(cp.asnumpy(P), cp.asnumpy(K), nr)
        else:
            return enc_fn(P, K, nr)


class PolytopicQuadrupleGenerator(Sequence):
    """
    Generates (quadruple-of-ciphertexts, label) batches following the
    (related-key) polytopic differential data-generation Algorithms 1 & 2
    of the polytopic-NDC paper.

    Parameters
    ----------
    encryption_function : callable(P, K, nr) -> C
        Bit-array encryption oracle: P, K are uint8 arrays of shape
        (N, plain_bits) / (N, key_bits); returns C of shape (N, plain_bits).
    plain_bits, key_bits : int
        Block size / key size in bits.
    nr : int
        Number of rounds to encrypt.
    pos_diffs : sequence of 3 plaintext differences (Delta_1, Delta_2, Delta_3)
        Polytope Delta, used for label 0. Each element is an int or bit array.
    neg_diffs : sequence of 3 plaintext differences (delta_1, delta_2, delta_3)
        Polytope delta, used for label 1.
    related_key : bool, default False
        If True, use the related-key scenario (Algorithm 2): a master key
        K0 is expanded into K0, K1, K2, K3 via pos_key_diffs / neg_key_diffs.
        If False (default), all 4 texts in a quadruple share the same key
        (Algorithm 1, single-key scenario).
    pos_key_diffs, neg_key_diffs : sequence of 3 key differences, required
        iff related_key=True. Correspond to Delta^key and delta^key.
    n_samples, batch_size : int
        Total number of samples / batch size.
    use_gpu : bool or None
        None -> auto-detect cupy availability.
    to_float32 : bool
        Cast final X to float32.
    encrypt_backend : {'numpy', 'cupy', 'auto'}
        Which backend to feed into encryption_function.
    """

    def __init__(self, encryption_function, plain_bits, key_bits, nr,
                 pos_diffs, neg_diffs,
                 related_key=False,
                 pos_key_diffs=None, neg_key_diffs=None,
                 n_samples=10 ** 7, batch_size=10 ** 5,
                 use_gpu=None, to_float32=True,
                 start_idx=0, encrypt_backend='numpy'):

        self.encryption_function = encryption_function
        self.plain_bits = plain_bits
        self.key_bits = key_bits
        self.nr = nr
        self.related_key = bool(related_key)

        if len(pos_diffs) != 3 or len(neg_diffs) != 3:
            raise ValueError(
                "pos_diffs and neg_diffs must each be a 3-tuple "
                "(Delta_1, Delta_2, Delta_3) / (delta_1, delta_2, delta_3), "
                "matching the polytope difference format of the paper."
            )

        if self.related_key:
            if pos_key_diffs is None or neg_key_diffs is None:
                raise ValueError(
                    "related_key=True requires pos_key_diffs and neg_key_diffs "
                    "(each a 3-tuple of key differences), per Algorithm 2."
                )
            if len(pos_key_diffs) != 3 or len(neg_key_diffs) != 3:
                raise ValueError("pos_key_diffs / neg_key_diffs must each have 3 entries.")

        # Precompute bit arrays (CPU) for the 3 plaintext differences per class
        self.pos_p_bits = np.array([_int_to_bitarray(d, plain_bits, np) for d in pos_diffs])
        self.neg_p_bits = np.array([_int_to_bitarray(d, plain_bits, np) for d in neg_diffs])

        if self.related_key:
            self.pos_k_bits = np.array([_int_to_bitarray(d, key_bits, np) for d in pos_key_diffs])
            self.neg_k_bits = np.array([_int_to_bitarray(d, key_bits, np) for d in neg_key_diffs])

        self.n = int(n_samples)
        self.batch_size = int(batch_size)
        self.start_idx = int(start_idx)

        if use_gpu is None:
            self.use_gpu = DEFAULT_USE_GPU
        else:
            self.use_gpu = bool(use_gpu)
        if self.use_gpu and cp is None:
            self.use_gpu = False

        self.to_float32 = bool(to_float32)

        if encrypt_backend == 'auto':
            self.encrypt_use_gpu = self.use_gpu
        elif encrypt_backend == 'cupy':
            self.encrypt_use_gpu = True
        else:
            self.encrypt_use_gpu = False
        if self.encrypt_use_gpu and cp is None:
            self.encrypt_use_gpu = False

        self.steps = (self.n + self.batch_size - 1) // self.batch_size
        # 4 ciphertexts (C0..C3) concatenated, per "Input and Normalization"
        self.input_dim = 4 * self.plain_bits

    def __len__(self):
        return self.steps

    def _to_backend(self, arr, lib):
        return lib.asarray(arr)

    def __getitem__(self, idx):
        curr_n = min(self.batch_size, self.n - idx * self.batch_size)
        if curr_n <= 0:
            raise IndexError

        lib = cp if (self.use_gpu and cp is not None) else np

        # --- Label ---
        Y = lib.random.randint(0, 2, size=curr_n).astype(lib.uint8)

        # --- Move precomputed polytope diffs to target backend ---
        pos_p = self._to_backend(self.pos_p_bits, lib)   # (3, plain_bits)
        neg_p = self._to_backend(self.neg_p_bits, lib)

        # --- Anchor plaintext P0 ---
        P0 = lib.random.randint(0, 2, (curr_n, self.plain_bits), dtype=lib.uint8)

        # Select, per-sample, the 3-tuple of diffs according to the label,
        # then build the plaintext quadruple (P0, P1, P2, P3) by applying
        # ALL THREE differences of the chosen tuple simultaneously.
        Y_col = Y.reshape(-1, 1)
        P_quad = [P0]
        for j in range(3):
            diff_j = lib.where(Y_col == 0, pos_p[j], neg_p[j])  # (curr_n, plain_bits)
            P_quad.append(P0 ^ diff_j)

        # --- Keys ---
        K0 = lib.random.randint(0, 2, (curr_n, self.key_bits), dtype=lib.uint8)
        if self.related_key:
            pos_k = self._to_backend(self.pos_k_bits, lib)
            neg_k = self._to_backend(self.neg_k_bits, lib)
            K_quad = [K0]
            for j in range(3):
                kdiff_j = lib.where(Y_col == 0, pos_k[j], neg_k[j])
                K_quad.append(K0 ^ kdiff_j)
        else:
            # single-key scenario: all 4 texts encrypted under the same key
            K_quad = [K0, K0, K0, K0]

        # --- Stack the 4 texts into one batch for a single encryption call ---
        P_all = lib.concatenate(P_quad, axis=0)   # (4*curr_n, plain_bits)
        K_all = lib.concatenate(K_quad, axis=0)   # (4*curr_n, key_bits)

        if (self.use_gpu and cp is not None) and not self.encrypt_use_gpu:
            P_in, K_in = cp.asnumpy(P_all), cp.asnumpy(K_all)
        elif (not self.use_gpu) and self.encrypt_use_gpu and cp is not None:
            P_in, K_in = cp.asarray(P_all), cp.asarray(K_all)
        else:
            P_in, K_in = P_all, K_all

        C_all = _safe_encrypt(self.encryption_function, P_in, K_in, self.nr)

        if self.use_gpu and cp is not None:
            C_all = cp.asarray(C_all) if not isinstance(C_all, cp.ndarray) else C_all
        else:
            C_all = cp.asnumpy(C_all) if (cp is not None and isinstance(C_all, cp.ndarray)) else C_all

        C0, C1, C2, C3 = lib.split(C_all, 4, axis=0)

        # --- Build model input: concatenated ciphertext quadruple, normalized to [-1, 1] ---
        X = lib.concatenate([C0, C1, C2, C3], axis=1)  # (curr_n, 4*plain_bits)

        if self.to_float32:
            X = X.astype(lib.float32)
            X = X * 2.0 - 1.0  # map bits {0,1} -> {-1,1}, per paper's normalization

        if self.use_gpu and cp is not None:
            return cp.asnumpy(X), cp.asnumpy(Y)

        return X.astype(np.float32) if self.to_float32 else X, Y.astype(np.uint8)