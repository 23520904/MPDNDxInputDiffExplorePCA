"""SPECK32/64 block cipher primitives.

Word size, rotation amounts, and the 4-word key schedule (expand_keys' `i % 3`)
are specific to the 32/64 parameter set. For SIMON32/SIMECK32 you'll want a
sibling module with the same shape (encrypt/decrypt/expand_keys) rather than
extending this one.
"""
WORD_SIZE = 16
ALPHA = 7
BETA = 2
MASK_VAL = (1 << WORD_SIZE) - 1


def rotate_left(value, shift):
    """Rotate a WORD_SIZE-bit value left by `shift` bits."""
    return ((value << shift) ^ (value >> (WORD_SIZE - shift))) & MASK_VAL


def rotate_right(value, shift):
    """Rotate a WORD_SIZE-bit value right by `shift` bits."""
    return ((value >> shift) ^ (value << (WORD_SIZE - shift))) & MASK_VAL


def enc_one_round(p, k):
    c0, c1 = p
    c0 = rotate_right(c0, ALPHA)
    c0 = (c0 + c1) & MASK_VAL
    c0 = c0 ^ k
    c1 = rotate_left(c1, BETA)
    c1 = c1 ^ c0
    return c0, c1


def dec_one_round(c, k):
    c0, c1 = c
    c1 = c1 ^ c0
    c1 = rotate_right(c1, BETA)
    c0 = c0 ^ k
    c0 = (c0 - c1) & MASK_VAL
    c0 = rotate_left(c0, ALPHA)
    return c0, c1


def expand_keys(key, num_rounds):
    """Expand a 4-word master key into `num_rounds` round keys.

    `key` is a length-4 sequence of word arrays ordered [k3, k2, k1, k0],
    matching the SPECK reference implementation's key-word ordering.
    """
    round_keys = [0] * num_rounds
    round_keys[0] = key[-1]
    l = list(reversed(key[:-1]))
    for i in range(num_rounds - 1):
        l[i % 3], round_keys[i + 1] = enc_one_round((l[i % 3], round_keys[i]), i)
    return round_keys


def encrypt(plaintext, round_keys):
    x, y = plaintext
    for k in round_keys:
        x, y = enc_one_round((x, y), k)
    return x, y


def decrypt(ciphertext, round_keys):
    x, y = ciphertext
    for k in reversed(round_keys):
        x, y = dec_one_round((x, y), k)
    return x, y


def check_testvector():
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6574, 0x694c)
    round_keys = expand_keys(key, 22)
    ct = encrypt(pt, round_keys)
    ok = ct == (0xa868, 0x42f2)
    print("Testvector verified." if ok else "Testvector not verified.")
    return ok


if __name__ == '__main__':
    check_testvector()