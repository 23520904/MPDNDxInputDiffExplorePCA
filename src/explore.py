import crypto.speck as speck
import pca_helper
import csv
import os
import clustering_helper
import math
import time
from datetime import datetime
import random

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
    hamming_weight=1,
    polytope_size=3,
    polytope_pool=None,):
    if polytope_size <= 0:
        raise ValueError("polytope_size must be positive.")
    
    while True:

        local_pool = []

        polytope = []

        for _ in range(polytope_size):
            diff =  generate_numbers_with_hamming_weight(
                bit_size=bit_size,
                hamming_weight=hamming_weight,
                number_pool=local_pool,
            )

            polytope.append(diff)
        # canonical representation
        polytope = tuple(sorted(polytope))

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

def explore_polytope_differences(blocksize=32, wordsize=16, nr=5, datasize=100000, hamming_weight=1, t0=0.003, t1=3, n_components=3, savepath=None):
    pdiffs_num = set()

    lambda_base = 1/(4*blocksize)

    num_idiff_cases =  calculate_combinations(blocksize,hamming_weight)
    num_pdiff_cases = calculate_combinations(num_idiff_cases,3)
    
    
    while len(pdiffs_num) < num_pdiff_cases:
        pdiff_num1 = generate_polytope_diff_num(blocksize, hamming_weight, polytope_pool=pdiffs_num)
        pdiff_num2 = generate_polytope_diff_num(blocksize, hamming_weight, polytope_pool=pdiffs_num)
        
        pdiff1 = pdiff_number_to_difference(pdiff_num1,wordsize)
        pdiff2 = pdiff_number_to_difference(pdiff_num2,wordsize)
        
        data_speck = speck.make_train_data(datasize, nr, pdiff1,pdiff2)

        eigen_value, eigen_vector = pca_helper.EigenValueDecomposition(dataset=data_speck)

        if sum(eigen_value - lambda_base > t0) >= t1:

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

            score = clustering_helper.calculate_silhouette(
                pca_results,
                labels
            )

            end_time = time.time()

            elapsed_time = end_time - start_time

            current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

            num_significant = np.sum(eigen_value - lambda_base > t0)

            polytope_str = (
                f"[({polytope_difference[0][0]:04x},{polytope_difference[0][1]:04x}), "
                f"({polytope_difference[1][0]:04x},{polytope_difference[1][1]:04x}), "
                f"({polytope_difference[2][0]:04x},{polytope_difference[2][1]:04x})]"
            )

            message = (
                "\n"
                "============================================================\n"
                f"[{current_time}]\n"
                f"Polytope Difference : {polytope_str}\n"
                f"Significant Eigen   : {num_significant}\n"
                f"Silhouette Score    : {score:.6f}\n"
                f"Elapsed Time        : {elapsed_time:.3f} sec\n"
                "============================================================"
            )

            print(message)

            if savepath is not None:

                # ---------- TXT ----------
                with open(savepath + ".txt", "a") as f:
                    f.write(message + "\n")

                # ---------- CSV ----------
                csv_path = savepath + ".csv"

                file_exists = os.path.isfile(csv_path)

                with open(csv_path, "a", newline="") as csvfile:

                    writer = csv.writer(csvfile)

                    if not file_exists:
                        writer.writerow([
                            "time",
                            "polytope_difference",
                            "significant_eigen",
                            "silhouette_score",
                            "elapsed_time"
                        ])

                    writer.writerow([
                        current_time,
                        polytope_str,
                        num_significant,
                        round(score, 6),
                        round(elapsed_time, 3)
                    ])







        
        
        


        


    