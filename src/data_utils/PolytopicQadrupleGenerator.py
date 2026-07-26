"""
PolytopicQuadrupleGenerator
===============================

Advanced variant of `PolytopicQuadrupleGenerator` that combines two ideas:

  1. The paper's polytope/quadruple data-generation procedure (Algorithm 1 /
     Algorithm 2 of the polytopic-NDC paper): one anchor P0, three
     simultaneous plaintext differences (Delta_1,Delta_2,Delta_3) for label 0
     / (delta_1,delta_2,delta_3) for label 1, encrypted into a ciphertext
     quadruple (C0,C1,C2,C3).

  2. The original "Polyhedral" generator's idea of also exposing the
     explicit XOR-difference features to the model (there: delta_C = C xor
     C_star for a *pair*). Here, generalized to the quadruple / d-difference
     definition used in the paper's "Preliminaries" section:

         Delta = (m0 xor m1, m0 xor m2, ..., m0 xor md)

     i.e. anchor-relative differences only:
         deltaC1 = C0 ^ C1
         deltaC2 = C0 ^ C2
         deltaC3 = C0 ^ C3

The paper itself runs this as an ablation (Section "Training with output
polytope differences", Table 5) using *only* (deltaC1, deltaC2, deltaC3) as
input and finds it *underperforms* feeding the raw ciphertext quadruple,
concluding that "ciphertexts provide more details to the distinguisher
model than their differences". This generator lets you choose, via
`feature_mode`, whether to reproduce that ablation, use the paper's main
setup, or combine both feature sets (the "advanced" combination requested):

    feature_mode='raw'   -> X = [C0, C1, C2, C3]                     (4 * plain_bits)  - paper's main PDND setup
    feature_mode='diff'  -> X = [deltaC1, deltaC2, deltaC3]          (3 * plain_bits)  - paper's Table 5 ablation
    feature_mode='full'  -> X = [deltaC1, deltaC2, deltaC3,
                                  C0, C1, C2, C3]                     (7 * plain_bits)  - combined / advanced (default)

`input_dim` is derived automatically from `feature_mode`, so no manual
adjustment is needed there. Everything else (label semantics, related-key
support, polytope-tuple validation, bit-array <-> tuple-of-words delta
input, GPU/CPU handling) is unchanged from `PolytopicQuadrupleGenerator`.
"""

from tensorflow.keras.utils import Sequence
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

DEFAULT_USE_GPU = cp is not None

FEATURE_MODES = ('raw', 'diff', 'full')


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
    Same polytope/quadruple data-generation as `PolytopicQuadrupleGenerator`
    (Algorithm 1 / 2 of the polytopic-NDC paper), with an additional
    `feature_mode` knob controlling whether the model input also includes
    the anchor-relative ciphertext differences (deltaC1, deltaC2, deltaC3),
    in the spirit of the original Polyhedral pair-generator's `delta_C`.

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
        Polytope Delta, used for label 0. Each element is an int, a tuple of
        words (e.g. (0x0040, 0x0)), or a bit array.
    neg_diffs : sequence of 3 plaintext differences (delta_1, delta_2, delta_3)
        Polytope delta, used for label 1.
    related_key : bool, default False
        If True, use the related-key scenario (Algorithm 2): a master key
        K0 is expanded into K0, K1, K2, K3 via pos_key_diffs / neg_key_diffs.
        If False (default), all 4 texts in a quadruple share the same key
        (Algorithm 1, single-key scenario).
    pos_key_diffs, neg_key_diffs : sequence of 3 key differences, required
        iff related_key=True. Correspond to Delta^key and delta^key.
    feature_mode : {'raw', 'diff', 'full'}, default 'full'
        'raw'  -> X = [C0, C1, C2, C3]                    (paper's main PDND input)
        'diff' -> X = [C0^C1, C0^C2, C0^C3]                (paper's Table 5 ablation input)
        'full' -> X = [C0^C1, C0^C2, C0^C3, C0, C1, C2, C3] (combined, advanced)
    n_samples, batch_size : int
        Total number of samples / batch size.
    use_gpu : bool or None
        None -> auto-detect cupy availability.
    to_float32 : bool
        Cast final X to float32 and normalize bits {0,1} -> {-1,1}.
    encrypt_backend : {'numpy', 'cupy', 'auto'}
        Which backend to feed into encryption_function.
    """

    def __init__(self, encryption_function, plain_bits, key_bits, nr,
                 pos_diffs, neg_diffs,
                 related_key=False,
                 pos_key_diffs=None, neg_key_diffs=None,
                 feature_mode='full',
                 n_samples=10 ** 7, batch_size=10 ** 5,
                 use_gpu=None, to_float32=True,
                 start_idx=0, encrypt_backend='numpy'):

        self.encryption_function = encryption_function
        self.plain_bits = plain_bits
        self.key_bits = key_bits
        self.nr = nr
        self.related_key = bool(related_key)

        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}")
        self.feature_mode = feature_mode

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
        self.start_idx = int(start_idx)  # reserved for deterministic slicing; unused, kept for API parity

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

        # input_dim depends on feature_mode - derived automatically, no
        # manual adjustment needed when switching modes.
        n_blocks = {'raw': 4, 'diff': 3, 'full': 7}[self.feature_mode]
        self.input_dim = n_blocks * self.plain_bits

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
        Y_col = Y.reshape(-1, 1)

        # --- Move precomputed polytope diffs to target backend ---
        pos_p = self._to_backend(self.pos_p_bits, lib)   # (3, plain_bits)
        neg_p = self._to_backend(self.neg_p_bits, lib)

        # --- Anchor plaintext P0, build quadruple (P0, P1, P2, P3) ---
        P0 = lib.random.randint(0, 2, (curr_n, self.plain_bits), dtype=lib.uint8)
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

        # --- Anchor-relative difference features (Polyhedral-style deltaC) ---
        deltaC1 = C0 ^ C1
        deltaC2 = C0 ^ C2
        deltaC3 = C0 ^ C3

        if self.feature_mode == 'raw':
            blocks = [C0, C1, C2, C3]
        elif self.feature_mode == 'diff':
            blocks = [deltaC1, deltaC2, deltaC3]
        else:  # 'full'
            blocks = [deltaC1, deltaC2, deltaC3, C0, C1, C2, C3]

        X = lib.concatenate(blocks, axis=1)

        if self.to_float32:
            X = X.astype(lib.float32)
            X = X * 2.0 - 1.0  # map bits {0,1} -> {-1,1}, per paper's normalization

        if self.use_gpu and cp is not None:
            return cp.asnumpy(X), cp.asnumpy(Y)

        return X.astype(np.float32) if self.to_float32 else X, Y.astype(np.uint8)