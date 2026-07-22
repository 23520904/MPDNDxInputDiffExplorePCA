"""SPECK32/64 block cipher primitives.

Word size, rotation amounts, and the 4-word key schedule (expand_keys' `i % 3`)
are specific to the 32/64 parameter set. For SIMON32/SIMECK32 you'll want a
sibling module with the same shape (encrypt/decrypt/expand_keys) rather than
extending this one.
"""
import numpy as np
from os import urandom

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

def encryption(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round((x,y), k)
    return(x, y)

POWERS16 = np.array(
    [1<<i for i in range(15,-1,-1)],
    dtype=np.uint16
)
def encrypt_wrapper(P, K, nr):
    """
    Hàm bọc xử lý mảng bit (bit-array) từ Data Generator, tương thích CPU/GPU.
    P: mảng bit shape (N, 32)
    K: mảng bit shape (N, 64)
    """
    # 1. Tự động nhận diện thư viện đang dùng là GPU (cupy) hay CPU (numpy)
    try:
        import cupy as cp
        lib = cp.get_array_module(P)
    except Exception:
        import numpy as lib
        
    # Ép kiểu dữ liệu về số nguyên 32-bit (tránh lỗi trôi dạt dấu phẩy động hoặc tràn bộ nhớ)
    P = P.astype(lib.uint32)
    K = K.astype(lib.uint32)
        
    # 2. KHÔI PHỤC BIT ARRAY VỀ KHỐI SỐ NGUYÊN (16-bit words)
    # Sử dụng nhân ma trận với lũy thừa của 2 để ghép 16 bit thành 1 số nguyên cực nhanh
    
    p_left  = lib.dot(P[:, :16], POWERS16).astype(lib.uint16)
    p_right = lib.dot(P[:, 16:], POWERS16).astype(lib.uint16)
    p_tuple = (p_left, p_right)
    
    k3 = lib.dot(K[:, :16], POWERS16)
    k2 = lib.dot(K[:, 16:32], POWERS16)
    k1 = lib.dot(K[:, 32:48], POWERS16)
    k0 = lib.dot(K[:, 48:], POWERS16)
    k_list = [k3, k2, k1, k0]
    
    # 3. GỌI HÀM MÃ HÓA GỐC CỦA BẠN
    ks = expand_keys(k_list, nr)
    c_left, c_right = encryption(p_tuple, ks)
    
    # 4. TÁCH SỐ NGUYÊN (CIPHERTEXT) TRỞ LẠI THÀNH BIT ARRAY ĐỂ TRẢ VỀ
    C_bits = lib.zeros(P.shape, dtype=lib.uint8)
    for i in range(16):
        C_bits[:, 15 - i] = (c_left >> i) & 1
        C_bits[:, 31 - i] = (c_right >> i) & 1
        
    return C_bits

if __name__ == '__main__':
    check_testvector()