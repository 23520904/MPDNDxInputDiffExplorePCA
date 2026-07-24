"""SPECK32/64 block cipher primitives.

Word size, rotation amounts, and the 4-word key schedule (expand_keys' `i % 3`)
are specific to the 32/64 parameter set.
"""
import numpy as np
from os import urandom

WORD_SIZE = 16
ALPHA = 7
BETA = 2
MASK_VAL = (1 << WORD_SIZE) - 1

def rotate_left(value, shift):
    return ((value << shift) ^ (value >> (WORD_SIZE - shift))) & MASK_VAL

def rotate_right(value, shift):
    return ((value >> shift) ^ (value << (WORD_SIZE - shift))) & MASK_VAL

def enc_one_round(p, k):
    c0, c1 = p
    c0 = rotate_right(c0, ALPHA)
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rotate_left(c1, BETA)
    c1 = c1 ^ c0
    return c0, c1

def expand_keys(key, num_rounds):
    round_keys = [0] * num_rounds
    round_keys[0] = key[-1]
    l = list(reversed(key[:-1]))
    for i in range(num_rounds - 1):
        l[i % 3], round_keys[i + 1] = enc_one_round((l[i % 3], round_keys[i]), i)
    return round_keys

def encryption(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x, y = enc_one_round((x, y), k)
    return (x, y)

# Ensure powers are 32-bit to prevent overflow during bit-packing
POWERS16 = np.array([1 << i for i in range(15, -1, -1)], dtype=np.uint32)

def encrypt_wrapper(P, K, nr):
    try:
        import cupy as cp
        lib = cp.get_array_module(P)
    except Exception:
        import numpy as lib

    P = P.astype(lib.uint32)
    K = K.astype(lib.uint32)
    p16 = lib.asarray(POWERS16, dtype=lib.uint32)

    # Reconstruct 16-bit words from bit arrays
    p_left = lib.dot(P[:, :16], p16).astype(lib.uint16)
    p_right = lib.dot(P[:, 16:], p16).astype(lib.uint16)

    k3 = lib.dot(K[:, :16], p16).astype(lib.uint16)
    k2 = lib.dot(K[:, 16:32], p16).astype(lib.uint16)
    k1 = lib.dot(K[:, 32:48], p16).astype(lib.uint16)
    k0 = lib.dot(K[:, 48:], p16).astype(lib.uint16)

    ks = expand_keys([k3, k2, k1, k0], nr)
    c_l, c_r = encryption((p_left, p_right), ks)

    # Convert back to bit arrays
    C_bits = lib.zeros(P.shape, dtype=lib.uint8)
    for i in range(16):
        C_bits[:, 15 - i] = (c_l >> i) & 1
        C_bits[:, 31 - i] = (c_r >> i) & 1
    return C_bits

if __name__ == '__main__':
    # Verify against SPECK32/64 Test Vector
    k_test = (0x1918, 0x1110, 0x0908, 0x0100)
    pt_test = (0x6574, 0x694c)
    ks_test = expand_keys(k_test, 22)
    ct_test = encryption(pt_test, ks_test)
    print(f'Test Vector Match (0xa868, 0x42f2): {ct_test == (0xa868, 0x42f2)}')