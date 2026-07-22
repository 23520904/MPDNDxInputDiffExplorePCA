"""Training/validation data generators for differential-neural distinguishers
on SPECK32/64.

Each generator returns (X, Y) where Y[i] in {0, 1} labels sample i as a
random pair (0) or a pair following the given input difference(s) (1), and X
is the corresponding bit-matrix ready to feed a Keras model.

`a` and `b` each hold one input-difference tuple per *extra* ciphertext pair
beyond the base pair (a[i] applied when Y==0, b[i] applied when Y==1). Any
number of tuples is supported -- this is where polytopic differences plug in,
just pass longer a/b tuples.
"""
import numpy as np
from os import urandom

import cipher

# Defaults from Gohr's original SPECK32/64 distinguisher (3 extra pairs).
DEFAULT_A = ((0x0040, 0x0), (0x0, 0x8000), (0x0060, 0x0))
DEFAULT_B = ((0x0020, 0x0), (0x0040, 0x8000), (0x0010, 0x2000))


def _random_words(n):
    """n independent random WORD_SIZE-bit words."""
    return np.frombuffer(urandom(2 * n), dtype=np.uint16)


def _random_labels(n):
    return np.frombuffer(urandom(n), dtype=np.uint8) & 1


def _random_keys(n, key_words=4):
    return np.frombuffer(urandom(2 * key_words * n), dtype=np.uint16).reshape(key_words, -1)


def _diff_pairs(plain_1, plain_2, Y, a, b):
    """One plaintext pair per (da, db) in zip(a, b): da applied where Y==0, db where Y==1."""
    pairs = []
    for (da0, da1), (db0, db1) in zip(a, b):
        p_1 = np.where(Y == 0, plain_1 ^ da0, plain_1 ^ db0)
        p_2 = np.where(Y == 0, plain_2 ^ da1, plain_2 ^ db1)
        pairs.append((p_1, p_2))
    return pairs


def convert_to_binary(word_arrays):
    """Stack word arrays into a (samples x bits) binary feature matrix.

    Each entry in `word_arrays` is a length-n array of WORD_SIZE-bit words;
    every word contributes WORD_SIZE consecutive bit-columns, MSB first.
    """
    n_words = len(word_arrays)
    n_samples = len(word_arrays[0])
    X = np.zeros((n_words * cipher.WORD_SIZE, n_samples), dtype=np.uint8)
    for i in range(n_words * cipher.WORD_SIZE):
        word_idx = i // cipher.WORD_SIZE
        bit_offset = cipher.WORD_SIZE - 1 - (i % cipher.WORD_SIZE)
        X[i] = (word_arrays[word_idx] >> bit_offset) & 1
    return X.transpose()


def make_train_data(n, nr, a=DEFAULT_A, b=DEFAULT_B, s_groups=1):
    """Fixed-key data: base ciphertext pair + one extra pair per difference in a/b."""
    Y = _random_labels(n)
    key = _random_keys(n)
    round_keys = cipher.expand_keys(key, nr)

    X = []
    for _ in range(s_groups):
        plain_1, plain_2 = _random_words(n), _random_words(n)
        X.extend(cipher.encrypt((plain_1, plain_2), round_keys))
        for p_1, p_2 in _diff_pairs(plain_1, plain_2, Y, a, b):
            X.extend(cipher.encrypt((p_1, p_2), round_keys))

    return convert_to_binary(X), Y


def make_train_rkdata(n, nr, a=DEFAULT_A, b=DEFAULT_B, s_groups=1):
    """Related-key data: the base pair uses the master key; each extra pair is
    encrypted under a key shifted by the same difference used on the plaintext
    (a[i] under Y==0, b[i] under Y==1), applied to the top two key words.
    """
    Y = _random_labels(n)
    key = _random_keys(n)
    round_keys = cipher.expand_keys(key, nr)

    round_keys_a, round_keys_b = [], []
    for (da0, da1), (db0, db1) in zip(a, b):
        key_a = key.copy()
        key_a[0] = key[0] ^ da1
        key_a[1] = key[1] ^ da0
        round_keys_a.append(cipher.expand_keys(key_a, nr))

        key_b = key.copy()
        key_b[0] = key[0] ^ db1
        key_b[1] = key[1] ^ db0
        round_keys_b.append(cipher.expand_keys(key_b, nr))

    X = []
    for _ in range(s_groups):
        plain_1, plain_2 = _random_words(n), _random_words(n)
        X.extend(cipher.encrypt((plain_1, plain_2), round_keys))

        for i, (p_1, p_2) in enumerate(_diff_pairs(plain_1, plain_2, Y, a, b)):
            c_1, c_2 = np.where(
                Y == 0,
                cipher.encrypt((p_1, p_2), round_keys_a[i]),
                cipher.encrypt((p_1, p_2), round_keys_b[i]),
            )
            X.append(c_1)
            X.append(c_2)

    return convert_to_binary(X), Y


def make_diff_data(n, nr, a=DEFAULT_A, b=DEFAULT_B, s_groups=1):
    """Fixed-key data with ciphertext blinding on the random (Y==0) class.

    The same blinding pair (R0, R1) is applied to every ciphertext pair for a
    given sample, so it cancels out in any pairwise XOR between the pairs:
    only the *relationship* between the pairs is exposed for random samples,
    never their raw values.
    """
    Y = _random_labels(n)
    num_rand = int(np.sum(Y == 0))
    key = _random_keys(n)
    round_keys = cipher.expand_keys(key, nr)
    R0 = _random_words(num_rand)
    R1 = _random_words(num_rand)

    X = []
    for _ in range(s_groups):
        plain_1, plain_2 = _random_words(n), _random_words(n)
        pairs = [(plain_1, plain_2)] + _diff_pairs(plain_1, plain_2, Y, a, b)

        for p_1, p_2 in pairs:
            c_1, c_2 = cipher.encrypt((p_1, p_2), round_keys)
            c_1[Y == 0] ^= R0
            c_2[Y == 0] ^= R1
            X.append(c_1)
            X.append(c_2)

    return convert_to_binary(X), Y


def make_train_data(pos_deltas,neg_deltas,)