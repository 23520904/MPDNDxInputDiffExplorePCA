# pca_helper.py
#
# PCA utilities for the Explore Polytope pipeline.
#
# These functions are representation-agnostic: they operate on a generic
# matrix X ∈ ℝ^{N×d} regardless of how the features were constructed.
# Ported from Paper A (Seok, SeoulTech) with no cipher-specific logic.
#
# Mathematical justification for reuse (see implementation_plan.md §5, Module D):
#   Under a random polytope, all d bit-features are ≈ i.i.d. Bernoulli(½),
#   so explained variance ratios are ≈ 1/d.  Under a good polytope, output
#   difference biases create off-diagonal covariance terms, which PCA captures
#   as leading eigenvalues exceeding 1/d.  This argument is dimension-agnostic.

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

def EigenValueDecomposition(dataset, alg=None, title=None, visualize_ratio='no'):
    scaler = StandardScaler()
    pca = PCA()
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(dataset)
    pca.fit_transform(dataset)
    
    if visualize_ratio == 'yes':
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_)
        plt.xlabel('features')
        plt.ylabel('variance')
        if alg is not None and title is not None:
            plt.title(alg.upper() + ' Variance - ' + title)
        plt.xticks(features)
        plt.show()
        plt.close()
    return pca.explained_variance_ratio_, pca.components_

def DimensionReduction(dataset, n_components = 3, alg=None, title=None):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(dataset)
    pca_results = pca.fit_transform(dataset)
    
    return pca_results