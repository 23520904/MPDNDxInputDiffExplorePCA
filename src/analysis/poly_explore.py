import speck32.cipher as speck
import analysis.pca_helper as pca_helper
import csv
import os
import math
import time
from datetime import datetime
import random
import numpy as np
import data_utils.PolyhedralMultiPairGenerator as make_data
import data_utils.PolytopicQadrupleGenerator as make_quad_data
import data_utils.name_generator as name_generator
from scipy.stats import energy_distance
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# DUS explorer — re-exported from distributional_explore for a unified API.
# ---------------------------------------------------------------------------
try:
    from analysis.distributional_explore import (  # noqa: F401
        linear_time_mmd2,
        random_fourier_features,
        hsic_rff,
        interaction_residual,
        dus_score,
        ParetoArchive,
        explore_dus,
        make_quadruples_adapter,
    )
except ImportError:
    from distributional_explore import (  # noqa: F401
        linear_time_mmd2,
        random_fourier_features,
        hsic_rff,
        interaction_residual,
        dus_score,
        ParetoArchive,
        explore_dus,
        make_quadruples_adapter,
    )

# ---------------------------------------------------------------------------
# Generator helpers (Copied from explore.py)
# ---------------------------------------------------------------------------
def _weighted_sample_without_replacement(population, weights, k):
    population = list(population)
    weights = list(weights)
    selected = []
    for _ in range(k):
        idx = random.choices(range(len(population)), weights=weights, k=1)[0]
        selected.append(population[idx])
        population.pop(idx)
        weights.pop(idx)
    return selected

def hw(x):
    return int(x).bit_count()

def generate_numbers_with_hamming_weight(bit_size=32, hamming_weight=1, number_pool=None,
                                        bit_placement="original", wordsize=16,
                                        left_weight=3.0):
    number = 0
    while number == 0:
        if bit_placement == "original":
            bit_position = random.sample(range(bit_size), hamming_weight)
        else:
            w_left = left_weight if bit_placement == "left" else 1.0
            position_weights = [1.0] * wordsize + [w_left] * (bit_size - wordsize)
            bit_position = _weighted_sample_without_replacement(
                range(bit_size), position_weights, hamming_weight
            )

        for position in bit_position:
            number |= (1 << position)
        if number_pool is not None:
            if number in number_pool:
                number = 0
            else:
                number_pool.append(number)
                return number
        else:
            return number

def generate_polytope_diff_num(bit_size=32,
    max_hamming_weight=1,
    polytope_size=3,
    polytope_pool=None,
    bit_placement="original",
    wordsize=16,
    left_weight=3.0):
    
    if polytope_size <= 0: raise ValueError("polytope_size must be positive.")
    if max_hamming_weight <= 0 or max_hamming_weight > bit_size: raise ValueError("Invalid max_hamming_weight.")
    
    while True:
        local_pool = []
        polytope = []
        weights = [1 / i for i in range(1, max_hamming_weight + 1)]
        for _ in range(polytope_size):
            hw_val = random.choices(range(1, max_hamming_weight + 1), weights=weights, k=1)[0]
            diff = generate_numbers_with_hamming_weight(
                bit_size=bit_size, hamming_weight=hw_val, number_pool=local_pool,
                bit_placement=bit_placement, wordsize=wordsize, left_weight=left_weight
            )
            polytope.append(diff)
            
        polytope = tuple(polytope)
        if polytope_pool is None: return polytope
        if polytope not in polytope_pool:
            polytope_pool.add(polytope)
            return polytope

def number_to_difference(number, wordsize=16):
    left = (number >> wordsize) & 0xFFFF
    right = number & 0xFFFF
    return (left, right)

def pdiff_number_to_difference(pdiff, wordsize=16):
    return tuple(number_to_difference(number, wordsize=wordsize) for number in pdiff)

def diff_hex(d):
    return f"(0x{d[0]:04X}, 0x{d[1]:04X})"

# ---------------------------------------------------------------------------
# Multi-Layer Filtering Logic
# ---------------------------------------------------------------------------
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedShuffleSplit

def calculate_s_dist(X, Y, n_components=3, subsample=1000):
    """Tính Multivariate Energy Distance trên top k trục PCA."""
    k = min(n_components, X.shape[1])
    X_k = X[:, :k]
    
    X_pos = X_k[Y == 1]
    X_neg = X_k[Y == 0]
    
    # Subsample để tính khoảng cách đa chiều nhanh chóng
    if len(X_pos) > subsample:
        X_pos = X_pos[np.random.choice(len(X_pos), subsample, replace=False)]
    if len(X_neg) > subsample:
        X_neg = X_neg[np.random.choice(len(X_neg), subsample, replace=False)]
        
    # Energy distance đa chiều: 2 * E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
    dist_xy = np.mean(cdist(X_pos, X_neg, metric='euclidean'))
    dist_xx = np.mean(cdist(X_pos, X_pos, metric='euclidean'))
    dist_yy = np.mean(cdist(X_neg, X_neg, metric='euclidean'))
    
    energy_dist = 2 * dist_xy - dist_xx - dist_yy
    return energy_dist

def calculate_s_stable(X, Y, n_components=3, n_splits=5):
    """Tính Stability Score bằng Stratified Resampling."""
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=42)
    distances = []
    
    for _, test_index in sss.split(X, Y):
        # Lấy 1 nửa dataset có bảo toàn tỷ lệ Pos/Neg (Stratified)
        X_fold, Y_fold = X[test_index], Y[test_index]
        dist = calculate_s_dist(X_fold, Y_fold, n_components=n_components)
        distances.append(dist)
        
    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Công thức: thưởng cho mean lớn và std nhỏ
    s_stable = mean_dist / (1.0 + std_dist)
    return s_stable

def calculate_s_nonlinear(X, Y, sample_size=1000):
    """Tính Nonlinear Score bằng Kernel PCA trên một mẫu nhỏ (Stratified)."""
    N = min(sample_size, len(X))
    
    # Lấy mẫu cân bằng Pos/Neg cho K-PCA
    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]
    
    n_pos = min(N // 2, len(pos_idx))
    n_neg = min(N - n_pos, len(neg_idx))
    
    idx_pos = np.random.choice(pos_idx, n_pos, replace=False)
    idx_neg = np.random.choice(neg_idx, n_neg, replace=False)
    indices = np.concatenate([idx_pos, idx_neg])
    
    X_sub = X[indices]
    Y_sub = Y[indices]
    
    # Kernel PCA (RBF) chiếu xuống 3 chiều
    kpca = KernelPCA(n_components=3, kernel='rbf', fit_inverse_transform=False, n_jobs=-1)
    try:
        X_kpca = kpca.fit_transform(X_sub)
    except Exception as e:
        print(f"    [!] KernelPCA error: {e}")
        return 0.0
    
    # Tính energy distance trên không gian phi tuyến 3 chiều
    # Ta có thể gọi trực tiếp hàm calculate_s_dist với n_components=3
    return calculate_s_dist(X_kpca, Y_sub, n_components=3, subsample=sample_size)

# ---------------------------------------------------------------------------
# Main Exploration Function
# ---------------------------------------------------------------------------
def explore_legacy_pca_kmeans(
    blocksize=32,
    wordsize=16,
    nr=5,
    datasize=100000,
    max_hamming_weight=1,
    feature_mode='diff',
    t0=0.003,
    t1=3,
    tau_dist=0.001,
    tau_stable=0.001,
    n_components=3,
    max_iterations=5000,
    max_good_candidates=50,
    random_state=None,
    savepath=None,
    bit_placement="left",
    left_weight=3.0
):
    """Legacy multi-layer PCA + energy-distance + KernelPCA explorer.

    Kept for backward compatibility and side-by-side calibration against the
    new DUS-based ``explore_dus`` path.  Use ``explore_dus`` for new searches.
    """
    if not savepath:
        suffix = f"speck32_{nr}r_{feature_mode}_v2"
        output_dir = name_generator.generate_experiment_path(category="explore", suffix=suffix)
        os.makedirs(output_dir, exist_ok=True)
        savepath = os.path.join(output_dir, "poly_explore_log")

    print('=' * 80)
    print('  🔍 MULTI-LAYER POLYTOPE EXPLORATION v2')
    print('=' * 80)
    print(f'  📂 Log & CSV sẽ được lưu tại: {os.path.abspath(savepath)}.csv')
    print('=' * 80)

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    good_candidates = []

    for iteration in range(max_iterations):
        if len(good_candidates) >= max_good_candidates:
            print(f"Found {max_good_candidates} good candidate polytopes. Stopping.")
            break

        # 1. Generate Candidates
        pdiff_num1 = generate_polytope_diff_num(blocksize, max_hamming_weight=max_hamming_weight,
            bit_placement=bit_placement, wordsize=wordsize, left_weight=left_weight)
        pdiff_num2 = generate_polytope_diff_num(blocksize, max_hamming_weight=max_hamming_weight,
            bit_placement=bit_placement, wordsize=wordsize, left_weight=left_weight) 

        data_generator = make_quad_data.PolytopicQuadrupleGenerator(
            encryption_function=speck.encrypt_wrapper,
            plain_bits=blocksize, key_bits=64, nr=nr,
            pos_diffs=pdiff_num1, neg_diffs=pdiff_num2,
            related_key=False, feature_mode=feature_mode,
            n_samples=datasize, batch_size=datasize,
            use_gpu=True, to_float32=True
        )

        data_speck, Y = data_generator[0]
        Y = Y.flatten() # Make sure Y is 1D

        start_time = time.time()

        # =========================================================
        # LAYER 1: Linear Fast Gate
        # =========================================================
        lambda_base = 1 / data_speck.shape[1]
        eigen_value, _ = pca_helper.EigenValueDecomposition(dataset=data_speck)
        
        num_significant = np.sum(eigen_value - lambda_base > t0)
        s_linear = np.sum(eigen_value[eigen_value - lambda_base > 0] - lambda_base)

        print(
            f"Iter {iteration + 1:5d} | "
            f"max_eig={eigen_value.max():.6f} | "
            f"sig={num_significant} | "
            f"s_lin={s_linear:.6f}"
        )

        if num_significant < t1:
            continue

        # =========================================================
        # Chiếu PCA cho Layer 2 & 3
        # =========================================================
        try:
            pca_results = pca_helper.DimensionReduction(data_speck, n_components=n_components)
        except Exception as e:
            print(f"[Warning] PCA failed: {e}")
            continue

        # =========================================================
        # LAYER 2: Pairwise Distributional Contrast
        # =========================================================
        s_dist = calculate_s_dist(pca_results, Y, n_components)
        if s_dist < tau_dist:
            print(f"  -> Failed L2: S_dist {s_dist:.6f} < {tau_dist}")
            continue

        # =========================================================
        # LAYER 3: Stability Screening
        # =========================================================
        s_stable = calculate_s_stable(pca_results, Y, n_components, n_splits=5)
        if s_stable < tau_stable:
            print(f"  -> Failed L3: S_stable {s_stable:.6f} < {tau_stable}")
            continue

        # =========================================================
        # LAYER 4: Nonlinear Confirmation (Dùng làm điểm xếp hạng)
        # =========================================================
        s_nonlinear = calculate_s_nonlinear(data_speck, Y, sample_size=1000)

        # =========================================================
        # KẾT THÚC CÁC MÀNG LỌC: Ghi nhận Candidate
        # =========================================================
        elapsed_time = time.time() - start_time
        
        good_candidates.append({
            'iteration': iteration + 1,
            'pdiff1': pdiff_num1,
            'pdiff2': pdiff_num2,
            's_linear': s_linear,
            's_dist': s_dist,
            's_stable': s_stable,
            's_nonlinear': s_nonlinear,
            'elapsed_time': elapsed_time
        })
        
        # Sắp xếp lại danh sách theo s_nonlinear
        good_candidates.sort(key=lambda x: x['s_nonlinear'], reverse=True)
        
        # Log ra màn hình candidate vừa tìm thấy
        pdiff1_words = pdiff_number_to_difference(pdiff_num1, wordsize)
        pdiff2_words = pdiff_number_to_difference(pdiff_num2, wordsize)
        polyA_hex = "[" + ", ".join(diff_hex(x) for x in pdiff1_words) + "]"
        polyB_hex = "[" + ", ".join(diff_hex(x) for x in pdiff2_words) + "]"
        
        current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        
        message = f"""
================================================================================
[+] CANDIDATE #{len(good_candidates)} FOUND | ITERATION: {iteration + 1:,}/{max_iterations:,}
================================================================================
> Search Speed      : {elapsed_time:.3f} sec | Time: {current_time}

[SCORES]
- S_nonlinear (L4) : {s_nonlinear:.6f} <-- Ranking Score
- S_linear (L1)    : {s_linear:.6f} (sig={num_significant})
- S_dist (L2)      : {s_dist:.6f}
- S_stable (L3)    : {s_stable:.6f}

[POLYTOPE DETAILS]
- Polytope 1    : {polyA_hex}
- Polytope 2    : {polyB_hex} 
--------------------------------------------------------------------------------
"""
        print(message)

        # Ghi log ra file
        if savepath is not None:
            # ---------------- TXT ----------------
            with open(savepath + ".txt", "a", encoding="utf8") as f:
                f.write(message)

            # ---------------- CSV ----------------
            csv_path = savepath + ".csv"
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, "a", newline="", encoding="utf8") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow([
                        "candidate_rank", "iteration", "time", "round", "datasize",
                        "s_nonlinear", "s_linear", "s_dist", "s_stable",
                        "num_significant", "polyA_hex", "polyB_hex", "elapsed_time"
                    ])

                writer.writerow([
                    len(good_candidates),
                    iteration + 1,
                    current_time,
                    nr, datasize,
                    s_nonlinear, s_linear, s_dist, s_stable,
                    num_significant, polyA_hex, polyB_hex, elapsed_time
                ])
                
    return good_candidates


# ---------------------------------------------------------------------------
# Backward-compatible alias — existing call sites continue to work.
# New code should call explore_legacy_pca_kmeans explicitly.
# ---------------------------------------------------------------------------
explore_polytopic_quadruple_differences = explore_legacy_pca_kmeans
