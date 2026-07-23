import crypto.speck as speck
import pca_helper
import csv
import os
import clustering_helper
import math
import time
from datetime import datetime
import random
import numpy as np
import utils.PolyhedralMultiPairGenerator as make_data
import importlib
import utils.PolytopicQadrupleGenerator as make_quad_data

def hw(x):
    return int(x).bit_count()
def calculate_combinations(m, n):
    return math.factorial(m) // (math.factorial(n) * math.factorial(m - n))

def generate_numbers_with_hamming_weight(bit_size = 32, hamming_weight = 1, number_pool=None):
    number = 0
    while number == 0:
        bit_position = random.sample(range(bit_size), hamming_weight)
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
    polytope_pool=None):
    if polytope_size <= 0:
        raise ValueError("polytope_size must be positive.")
    if max_hamming_weight <= 0 or max_hamming_weight > bit_size:
        raise ValueError("Invalid max_hamming_weight.")
    while True:

        local_pool = []

        polytope = []
        weights = [1 / i for i in range(1, max_hamming_weight + 1)]
        for _ in range(polytope_size):
            hw = random.choices(
                range(1, max_hamming_weight + 1),
                weights=weights,
                k=1,
            )[0]
            diff =  generate_numbers_with_hamming_weight(
                bit_size=bit_size,
                hamming_weight=hw,
                number_pool=local_pool,
            )

            polytope.append(diff)
        # canonical representation
        polytope = tuple(polytope)

        if polytope_pool is None:
            return polytope

        if polytope not in polytope_pool:
            polytope_pool.add(polytope)
            return polytope

    
def number_to_difference(number, wordsize=16):
    left = (number >> wordsize) & 0xFFFF
    right = number & 0xFFFF
    return (left, right)

def pdiff_number_to_difference(pdiff, wordsize=16):
    return tuple(
        number_to_difference(number, wordsize=wordsize)
        for number in pdiff
    )

def diff_hex(d):
    return f"(0x{d[0]:04X}, 0x{d[1]:04X})"

def explore_polytope_differences(blocksize=32, wordsize=16, nr=5, datasize=100000, max_hamming_weight=1, t0=0.003, t1=3, n_components=3, max_iterations=5000, max_good_candidates=50,random_state=None, savepath=None):

    # Thiết lập random_state để đảm bảo khả năng tái lập (reproducibility)
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # pdiffs_num = set()

    

    good_candidates_found = 0

    for iteration in range(max_iterations):
        # Dừng sớm nếu đã thu thập đủ số lượng candidate tốt
        if good_candidates_found >= max_good_candidates:
            print(f"Đã tìm đủ {max_good_candidates} candidates tốt. Dừng thuật toán.")
            break



        pdiff_num1 = generate_polytope_diff_num(blocksize, max_hamming_weight=max_hamming_weight) 
        pdiff_num2 = generate_polytope_diff_num(blocksize, max_hamming_weight=max_hamming_weight) 
        
        
        pos_deltas = [(d, 0) for d in pdiff_num1]
        neg_deltas = [(d, 0) for d in pdiff_num2]
        
        data_generator = make_data.PolyhedralMultiPairGenerator(
            encryption_function=speck.encrypt_wrapper,
            pos_deltas=pos_deltas,
            neg_deltas=neg_deltas,
            plain_bits=blocksize,
            key_bits=64,
            nr=nr,
            n_samples=datasize,
            batch_size=datasize,  # Generate all data in one batch
            start_idx=0,
            use_gpu=True,
            to_float32=True,  # Convert data to float32 for PCA
        )

        data_speck, Y = data_generator[0]  # Get the entire dataset in one batch
        
        lambda_base = 1/data_speck.shape[1]

        pdiff1 = pdiff_number_to_difference(pdiff_num1, wordsize)
        pdiff2 = pdiff_number_to_difference(pdiff_num2, wordsize)


        eigen_value, eigen_vector = pca_helper.EigenValueDecomposition(dataset=data_speck)
        num_significant = np.sum(eigen_value - lambda_base > t0)


        print(
            f"iteration={iteration + 1:5d} | "
            f"max={eigen_value.max():.6f} | "
            f"sig={num_significant}\n"
            f"lambda_base={lambda_base:.6f} |"
            f"data_speck.shape={data_speck.shape} |"
            f"sum(eigen_value)={np.sum(eigen_value):.6f}"
        )
        if num_significant >= t1:

            try:
                pca_results = pca_helper.DimensionReduction(
                    data_speck,
                    n_components=n_components
                )

                start_time = time.time()

                labels = clustering_helper.kmeans_clustering(
                    pca_results,
                    3 ** n_components,
                    3
                )

                score = float(
                    clustering_helper.calculate_silhouette(
                        pca_results,
                        labels
                    )
                )

                elapsed_time = time.time() - start_time


                # Tăng bộ đếm khi tìm được một candidate đạt chuẩn
                good_candidates_found += 1

            except Exception as e:
                print(f"[Warning] Skip candidate : {e}")
                continue

            current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

            mask = (eigen_value - lambda_base) > t0
            selected_indices = np.where(mask)[0]
            selected_eigenvalues = eigen_value[mask]

            # ---------------------------------------------------------
            # format polytope
            # ---------------------------------------------------------

            

            polyA_hex = "[" + ", ".join(diff_hex(x) for x in pdiff1) + "]"
            hw_polyA = [hw(x) for x in pdiff_num1]

            polyB_hex = "[" + ", ".join(diff_hex(x) for x in pdiff2) + "]"
            hw_polyB = [hw(x) for x in pdiff_num2]

            message = f"""
================================================================================
[+] CANDIDATE #{good_candidates_found}/{max_good_candidates} FOUND | ITERATION: {iteration + 1:,}/{max_iterations:,}
================================================================================
> Search Speed      : {elapsed_time:.3f} sec | Time: {current_time}

[1. CLUSTERING QUALITY EVALUATION]
- Silhouette Score  : {score:.6f}  <-- Critical metric (Closer to 1 is better)

[2. PCA FEATURES EVALUATION (Tune t0 & t1)]
- Configuration     : Requires t1={t1} components exceeding t0={t0}
- Actual Result     : Found {int(num_significant)} significant components
- Max Eigenvalue    : {np.max(selected_eigenvalues):.6f} (Max deviation from lambda_base)
- Selected Axes     : {np.round(selected_eigenvalues, 6).tolist()}

[3. POLYTOPE DIFFERENCES DETAILS (Tune Max HW={max_hamming_weight})]
- Polytope 1 (HEX)  : {polyA_hex}
  -> HW Distribution: {hw_polyA} (Total HW: {sum(hw_polyA)})
- Polytope 2 (HEX)  : {polyB_hex} 
  -> HW Distribution: {hw_polyB} (Total HW: {sum(hw_polyB)})
--------------------------------------------------------------------------------
"""

            print(message)

            if savepath is not None:

                save_dir = os.path.dirname(savepath)

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)

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
                        "candidate",
                        "time",
                        "iteration",
                        "round",
                        "datasize",
                        "blocksize",
                        "wordsize",
                        "max_hw",
                        "random_seed",
                        "polyA_hw",
                        "polyB_hw",
                        "polyA_total_hw",
                        "polyB_total_hw",
                        "lambda_base",
                        "t0",
                        "t1",
                        "pca_components",
                        "clusters",
                        "polyA_hex",
                        "polyB_hex",
                        "polyA_decimal",
                        "polyB_decimal",
                        "num_significant",
                        "selected_index",
                        "selected_eigenvalues",
                        "silhouette",
                        "max_eigen_ratio",
                        "elapsed_time"
                                ])

                    writer.writerow([
                        good_candidates_found,
                        current_time,
                        iteration + 1,
                        nr,
                        datasize,
                        blocksize,
                        wordsize,
                        max_hamming_weight,
                        random_state,
                        ";".join(map(str, hw_polyA)),
                        ";".join(map(str, hw_polyB)),
                        sum(hw_polyA),
                        sum(hw_polyB),
                        lambda_base,
                        t0,
                        t1,
                        n_components,
                        3 ** n_components,
                        polyA_hex,
                        polyB_hex,
                        str(pdiff1),
                        str(pdiff2),
                        int(num_significant),
                        ";".join(map(str, selected_indices)),
                        ";".join(f"{x:.8f}" for x in selected_eigenvalues),
                        score,
                        eigen_value.max(),
                        elapsed_time
                    ])
                    
                    

def explore_polytope_differences_using_pndc_mtd(blocksize=32, wordsize=16, nr=5, datasize=100000, max_hamming_weight=1, t0=0.003, t1=3, n_components=3, max_iterations=5000, max_good_candidates=50,random_state=None, savepath=None):

    # Thiết lập random_state để đảm bảo khả năng tái lập (reproducibility)
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    

    

    good_candidates_found = 0

    for iteration in range(max_iterations):
        # Dừng sớm nếu đã thu thập đủ số lượng candidate tốt
        if good_candidates_found >= max_good_candidates:
            print(f"Đã tìm đủ {max_good_candidates} candidates tốt. Dừng thuật toán.")
            break

        pdiff_num1 = generate_polytope_diff_num(blocksize, max_hamming_weight=max_hamming_weight) 
        pdiff_num2 = generate_polytope_diff_num(blocksize, max_hamming_weight=max_hamming_weight) 
        
        
        pdiff1 = pdiff_number_to_difference(pdiff_num1, wordsize)
        pdiff2 = pdiff_number_to_difference(pdiff_num2, wordsize)
        
        data_speck,Y = speck.make_train_data(datasize, nr, pdiff1,pdiff2)
        
        lambda_base = 1/data_speck.shape[1]

        eigen_value, eigen_vector = pca_helper.EigenValueDecomposition(dataset=data_speck)
        num_significant = np.sum(eigen_value - lambda_base > t0)

        print(
            f"iteration={iteration + 1:5d} | "
            f"max={eigen_value.max():.6f} | "
            f"sig={num_significant}\n"
            f"lambda_base={lambda_base:.6f} |"
            f"data_speck.shape={data_speck.shape} |"
            f"sum(eigen_value)={np.sum(eigen_value):.6f}"
        )
        if num_significant >= t1:

            try:
                pca_results = pca_helper.DimensionReduction(
                    data_speck,
                    n_components=n_components
                )

                start_time = time.time()

                labels = clustering_helper.kmeans_clustering(
                    pca_results,
                    3 ** n_components,
                    3
                )

                score = float(
                    clustering_helper.calculate_silhouette(
                        pca_results,
                        labels
                    )
                )

                elapsed_time = time.time() - start_time


                # Tăng bộ đếm khi tìm được một candidate đạt chuẩn
                good_candidates_found += 1

            except Exception as e:
                print(f"[Warning] Skip candidate : {e}")
                continue

            current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

            mask = (eigen_value - lambda_base) > t0
            selected_indices = np.where(mask)[0]
            selected_eigenvalues = eigen_value[mask]

            # ---------------------------------------------------------
            # format polytope
            # ---------------------------------------------------------

            

            polyA_hex = "[" + ", ".join(diff_hex(x) for x in pdiff1) + "]"
            hw_polyA = [hw(x) for x in pdiff_num1]

            polyB_hex = "[" + ", ".join(diff_hex(x) for x in pdiff2) + "]"
            hw_polyB = [hw(x) for x in pdiff_num2]

            message = f"""
================================================================================
[+] CANDIDATE #{good_candidates_found}/{max_good_candidates} FOUND | ITERATION: {iteration + 1:,}/{max_iterations:,}
================================================================================
> Search Speed      : {elapsed_time:.3f} sec | Time: {current_time}

[1. CLUSTERING QUALITY EVALUATION]
- Silhouette Score  : {score:.6f}  <-- Critical metric (Closer to 1 is better)

[2. PCA FEATURES EVALUATION (Tune t0 & t1)]
- Configuration     : Requires t1={t1} components exceeding t0={t0}
- Actual Result     : Found {int(num_significant)} significant components
- Max Eigenvalue    : {np.max(selected_eigenvalues):.6f} (Max deviation from lambda_base)
- Selected Axes     : {np.round(selected_eigenvalues, 6).tolist()}

[3. POLYTOPE DIFFERENCES DETAILS (Tune Max HW={max_hamming_weight})]
- Polytope 1 (HEX)  : {polyA_hex}
  -> HW Distribution: {hw_polyA} (Total HW: {sum(hw_polyA)})
- Polytope 2 (HEX)  : {polyB_hex} 
  -> HW Distribution: {hw_polyB} (Total HW: {sum(hw_polyB)})
--------------------------------------------------------------------------------
"""

            print(message)

            if savepath is not None:

                save_dir = os.path.dirname(savepath)

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)

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
                        "candidate",
                        "time",
                        "iteration",
                        "round",
                        "datasize",
                        "blocksize",
                        "wordsize",
                        "max_hw",
                        "random_seed",
                        "polyA_hw",
                        "polyB_hw",
                        "polyA_total_hw",
                        "polyB_total_hw",
                        "lambda_base",
                        "t0",
                        "t1",
                        "pca_components",
                        "clusters",
                        "polyA_hex",
                        "polyB_hex",
                        "polyA_decimal",
                        "polyB_decimal",
                        "num_significant",
                        "selected_index",
                        "selected_eigenvalues",
                        "silhouette",
                        "max_eigen_ratio",
                        "elapsed_time"
                                ])

                    writer.writerow([
                        good_candidates_found,
                        current_time,
                        iteration + 1,
                        nr,
                        datasize,
                        blocksize,
                        wordsize,
                        max_hamming_weight,
                        random_state,
                        ";".join(map(str, hw_polyA)),
                        ";".join(map(str, hw_polyB)),
                        sum(hw_polyA),
                        sum(hw_polyB),
                        lambda_base,
                        t0,
                        t1,
                        n_components,
                        3 ** n_components,
                        polyA_hex,
                        polyB_hex,
                        str(pdiff1),
                        str(pdiff2),
                        int(num_significant),
                        ";".join(map(str, selected_indices)),
                        ";".join(f"{x:.8f}" for x in selected_eigenvalues),
                        score,
                        eigen_value.max(),
                        elapsed_time
                    ])



def explore_polytopic_quadruple_differences(
    blocksize=32,
    wordsize=16,
    nr=5,
    datasize=100000,
    max_hamming_weight=1,
    feature_mode='diff',  # Options: 'diff', 'raw', or 'full'
    t0=0.003,
    t1=3,
    n_components=3,
    max_iterations=5000,
    max_good_candidates=50,
    random_state=None,
    savepath=None
):
    """
    Explores polytopic quadruple differences for block ciphers using PCA.
    Uses identical logging format and CSV/TXT output saving logic as explore_polytope_differences.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    good_candidates_found = 0

    for iteration in range(max_iterations):
        if good_candidates_found >= max_good_candidates:
            print(f"Found {max_good_candidates} good candidate polytopes. Stopping.")
            break

        pdiff_num1 = generate_polytope_diff_num(blocksize, max_hamming_weight=max_hamming_weight) 
        pdiff_num2 = generate_polytope_diff_num(blocksize, max_hamming_weight=max_hamming_weight) 

        data_generator = make_quad_data.PolytopicQuadrupleGenerator(
            encryption_function=speck.encrypt_wrapper,
            plain_bits=blocksize,
            key_bits=64,
            nr=nr,
            pos_diffs=pdiff_num1,     # 3-tuple (Delta_1, Delta_2, Delta_3)
            neg_diffs=pdiff_num2,     # 3-tuple (delta_1, delta_2, delta_3)
            related_key=False,        # Single-key scenario
            feature_mode=feature_mode, # 'diff', 'raw', or 'full'
            n_samples=datasize,
            batch_size=datasize,      # Generate all in 1 batch for PCA
            use_gpu=True,
            to_float32=True
        )

        data_speck, Y = data_generator[0]

        lambda_base = 1 / data_speck.shape[1]

        pdiff1 = pdiff_number_to_difference(pdiff_num1, wordsize)
        pdiff2 = pdiff_number_to_difference(pdiff_num2, wordsize)

        eigen_value, eigen_vector = pca_helper.EigenValueDecomposition(dataset=data_speck)
        num_significant = np.sum(eigen_value - lambda_base > t0)

        print(
            f"iteration={iteration + 1:5d} | "
            f"max={eigen_value.max():.6f} | "
            f"sig={num_significant}\n"
            f"lambda_base={lambda_base:.6f} |"
            f"data_speck.shape={data_speck.shape} |"
            f"sum(eigen_value)={np.sum(eigen_value):.6f}"
        )

        if num_significant >= t1:

            try:
                pca_results = pca_helper.DimensionReduction(
                    data_speck,
                    n_components=n_components
                )

                start_time = time.time()

                labels = clustering_helper.kmeans_clustering(
                    pca_results,
                    3 ** n_components,
                    3
                )

                score = float(
                    clustering_helper.calculate_silhouette(
                        pca_results,
                        labels
                    )
                )

                elapsed_time = time.time() - start_time

                good_candidates_found += 1

            except Exception as e:
                print(f"[Warning] Skip candidate : {e}")
                continue

            current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

            mask = (eigen_value - lambda_base) > t0
            selected_indices = np.where(mask)[0]
            selected_eigenvalues = eigen_value[mask]

            polyA_hex = "[" + ", ".join(diff_hex(x) for x in pdiff1) + "]"
            hw_polyA = [hw(x) for x in pdiff_num1]

            polyB_hex = "[" + ", ".join(diff_hex(x) for x in pdiff2) + "]"
            hw_polyB = [hw(x) for x in pdiff_num2]

            message = f"""
================================================================================
[+] CANDIDATE #{good_candidates_found}/{max_good_candidates} FOUND | ITERATION: {iteration + 1:,}/{max_iterations:,}
================================================================================
> Search Speed      : {elapsed_time:.3f} sec | Time: {current_time}

[1. CLUSTERING QUALITY EVALUATION]
- Silhouette Score  : {score:.6f}  <-- Critical metric (Closer to 1 is better)

[2. PCA FEATURES EVALUATION (Tune t0 & t1)]
- Configuration     : Requires t1={t1} components exceeding t0={t0}
- Actual Result     : Found {int(num_significant)} significant components
- Max Eigenvalue    : {np.max(selected_eigenvalues):.6f} (Max deviation from lambda_base)
- Selected Axes     : {np.round(selected_eigenvalues, 6).tolist()}

[3. POLYTOPE DIFFERENCES DETAILS (Tune Max HW={max_hamming_weight})]
- Polytope 1 (HEX)  : {polyA_hex}
  -> HW Distribution: {hw_polyA} (Total HW: {sum(hw_polyA)})
- Polytope 2 (HEX)  : {polyB_hex} 
  -> HW Distribution: {hw_polyB} (Total HW: {sum(hw_polyB)})
--------------------------------------------------------------------------------
"""

            print(message)

            if savepath is not None:

                save_dir = os.path.dirname(savepath)

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)

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
                        "candidate",
                        "time",
                        "iteration",
                        "round",
                        "datasize",
                        "blocksize",
                        "wordsize",
                        "max_hw",
                        "random_seed",
                        "polyA_hw",
                        "polyB_hw",
                        "polyA_total_hw",
                        "polyB_total_hw",
                        "lambda_base",
                        "t0",
                        "t1",
                        "pca_components",
                        "clusters",
                        "polyA_hex",
                        "polyB_hex",
                        "polyA_decimal",
                        "polyB_decimal",
                        "num_significant",
                        "selected_index",
                        "selected_eigenvalues",
                        "silhouette",
                        "max_eigen_ratio",
                        "elapsed_time"
                                ])

                    writer.writerow([
                        good_candidates_found,
                        current_time,
                        iteration + 1,
                        nr,
                        datasize,
                        blocksize,
                        wordsize,
                        max_hamming_weight,
                        random_state,
                        ";".join(map(str, hw_polyA)),
                        ";".join(map(str, hw_polyB)),
                        sum(hw_polyA),
                        sum(hw_polyB),
                        lambda_base,
                        t0,
                        t1,
                        n_components,
                        3 ** n_components,
                        polyA_hex,
                        polyB_hex,
                        str(pdiff1),
                        str(pdiff2),
                        int(num_significant),
                        ";".join(map(str, selected_indices)),
                        ";".join(f"{x:.8f}" for x in selected_eigenvalues),
                        score,
                        eigen_value.max(),
                        elapsed_time
                    ])