from tensorflow.keras.utils import Sequence
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

DEFAULT_USE_GPU = cp is not None


def _int_to_bitarray(val, nbits, lib):
    """
    Convert val (int or numpy/cupy array) -> bit array length nbits
    in backend lib (np or cp). Returns uint8 array.
    """
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
            # Nhánh xử lý: GPU -> CPU fallback (Đã chuẩn hoá theo yêu cầu)
            return enc_fn(cp.asnumpy(P), cp.asnumpy(K), nr)
        else:
            # Nhánh xử lý: CPU -> GPU fallback
            print(enc_fn)
            print(type(P), type(K))
            return enc_fn(P, K, nr)

# old format
class PolyhedralMultiPairGenerator(Sequence):
    def __init__(self, encryption_function, plain_bits, key_bits, nr,
                 pos_deltas, neg_deltas,
                 n_samples=10**7, batch_size=10**5,
                 pairs=2, use_gpu=None, to_float32=True,
                 start_idx=0, encrypt_backend='numpy'):
        """
        pos_deltas: list of tuples [(delta_P1, delta_K1), (delta_P2, delta_K2), ...]
        neg_deltas: list of tuples [(delta_P3, delta_K3), (delta_P4, delta_K4), ...]
        """
        self.encryption_function = encryption_function
        self.plain_bits = plain_bits
        self.key_bits = key_bits
        self.nr = nr

        # 1. Tiền xử lý dữ liệu: Lọc bỏ các khác biệt 0x0 (HW=0 là vô nghĩa)
        valid_pos = [d for d in pos_deltas if d[0] != 0]
        valid_neg = [d for d in neg_deltas if d[0] != 0]

        if not valid_pos or not valid_neg:
            raise ValueError("Tập pos_deltas hoặc neg_deltas rỗng hoặc chỉ chứa input difference = 0x0.")

        self.num_pos = len(valid_pos)
        self.num_neg = len(valid_neg)

        # 2. Tiền tính toán mảng bit (Precompute Bitarrays) bằng CPU để tránh tràn số int lớn (128-bit)
        self.pos_p_bits = np.array([_int_to_bitarray(d[0], plain_bits, np) for d in valid_pos])
        self.pos_k_bits = np.array([_int_to_bitarray(d[1], key_bits, np) for d in valid_pos])
        
        self.neg_p_bits = np.array([_int_to_bitarray(d[0], plain_bits, np) for d in valid_neg])
        self.neg_k_bits = np.array([_int_to_bitarray(d[1], key_bits, np) for d in valid_neg])

        # Cấu hình batch và training
        self.n = int(n_samples)
        self.batch_size = int(batch_size)
        self.start_idx = int(start_idx)
        self.pairs = int(pairs)

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
        self.input_dim = self.pairs * 3 * self.plain_bits

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        curr_n = min(self.batch_size, self.n - idx * self.batch_size)
        if curr_n <= 0:
            raise IndexError

        lib = cp if (self.use_gpu and cp is not None) else np

        # Tạo Label Y: Nửa đầu là 1 (Positive), nửa sau là 0 (Negative)
        Y = lib.zeros(curr_n, dtype=lib.uint8)
        half_n = curr_n // 2
        Y[:half_n] = 1

        # Chuyển các mảng bit đã tiền tính toán lên thư viện đích (GPU/CPU)
        pos_p_lib = lib.asarray(self.pos_p_bits)
        pos_k_lib = lib.asarray(self.pos_k_bits)
        neg_p_lib = lib.asarray(self.neg_p_bits)
        neg_k_lib = lib.asarray(self.neg_k_bits)

        # Trích xuất ngẫu nhiên các index cho batch này
        idx_pos = lib.random.randint(0, self.num_pos, size=half_n)
        idx_neg = lib.random.randint(0, self.num_neg, size=curr_n - half_n)

        # Xây dựng ma trận delta_vecs cho toàn bộ batch
        batch_delta_P_vecs = lib.empty((curr_n, self.plain_bits), dtype=lib.uint8)
        batch_delta_K_vecs = lib.empty((curr_n, self.key_bits), dtype=lib.uint8)

        batch_delta_P_vecs[:half_n] = pos_p_lib[idx_pos]
        batch_delta_K_vecs[:half_n] = pos_k_lib[idx_pos]
        
        batch_delta_P_vecs[half_n:] = neg_p_lib[idx_neg]
        batch_delta_K_vecs[half_n:] = neg_k_lib[idx_neg]

        # Xáo trộn (Shuffle) đồng bộ cả Label và Delta vecs
        shuffle_idx = lib.random.permutation(curr_n)
        Y = Y[shuffle_idx]
        batch_delta_P_vecs = batch_delta_P_vecs[shuffle_idx]
        batch_delta_K_vecs = batch_delta_K_vecs[shuffle_idx]

        # Khởi tạo bản rõ (P) và Khóa (K0) ngẫu nhiên
        K0 = lib.random.randint(0, 2, (curr_n, self.key_bits), dtype=lib.uint8)
        P = lib.random.randint(0, 2, (curr_n * self.pairs, self.plain_bits), dtype=lib.uint8)

        # Tính K1 và lặp lại K cho tất cả pairs
        K1 = K0 ^ batch_delta_K_vecs
        K = lib.repeat(K0, self.pairs, axis=0)
        K_star = lib.repeat(K1, self.pairs, axis=0)

        # Tính P_star
        delta_P_vecs_repeated = lib.repeat(batch_delta_P_vecs, self.pairs, axis=0)
        P_star = P ^ delta_P_vecs_repeated

        # --- ENCRYPTION PREP ---
        if (self.use_gpu and cp is not None) and not self.encrypt_use_gpu:
            P_in, K_in = cp.asnumpy(P), cp.asnumpy(K)
            P_star_in, K_star_in = cp.asnumpy(P_star), cp.asnumpy(K_star)

        elif (not self.use_gpu) and self.encrypt_use_gpu and cp is not None:
            P_in, K_in = cp.asarray(P), cp.asarray(K)
            P_star_in, K_star_in = cp.asarray(P_star), cp.asarray(K_star)

        else:
            P_in, K_in = P, K
            P_star_in, K_star_in = P_star, K_star

        # --- SAFE ENCRYPT ---
        C = _safe_encrypt(self.encryption_function, P_in, K_in, self.nr)
        C_star = _safe_encrypt(self.encryption_function, P_star_in, K_star_in, self.nr)

        # --- NORMALIZE BACKEND ---
        if self.use_gpu and cp is not None:
            C = cp.asarray(C) if not isinstance(C, cp.ndarray) else C
            C_star = cp.asarray(C_star) if not isinstance(C_star, cp.ndarray) else C_star
        else:
            C = cp.asnumpy(C) if (cp is not None and isinstance(C, cp.ndarray)) else C
            C_star = cp.asnumpy(C_star) if (cp is not None and isinstance(C_star, cp.ndarray)) else C_star

        # Tính toán đặc trưng (features)
        delta_C = C ^ C_star
        triple = lib.concatenate([delta_C, C, C_star], axis=1)

        X = triple.reshape(curr_n, -1)
        if self.to_float32:
            X = X.astype(lib.float32)

        if self.use_gpu and cp is not None:
            return cp.asnumpy(X), cp.asnumpy(Y)

        return X.astype(np.float32), Y.astype(np.uint8)