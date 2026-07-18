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


def eigenvalue_decomposition(
    dataset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA eigenvalues (explained variance ratios) and eigenvectors.

    Parameters
    ----------
    dataset : np.ndarray
        Feature matrix of shape ``(N, d)``.

    Returns
    -------
    eigen_values : np.ndarray
        Explained variance ratios, shape ``(d,)``, summing to 1.
    eigen_vectors : np.ndarray
        Principal components, shape ``(d, d)``.
    """
    scaler = StandardScaler()
    pca = PCA()
    pipeline = make_pipeline(scaler, pca)
    pipeline.fit(dataset)

    return pca.explained_variance_ratio_, pca.components_


def dimension_reduction(
    dataset: np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """Project the dataset onto *n_components* principal components.

    Parameters
    ----------
    dataset : np.ndarray
        Feature matrix of shape ``(N, d)``.
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    np.ndarray
        Projected data of shape ``(N, n_components)``.
    """
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    pipeline = make_pipeline(scaler, pca)
    return pipeline.fit_transform(dataset)


def passes_eigenvalue_filter(
    eigen_values: np.ndarray,
    feature_dim: int,
    t0: float = 0.003,
    t1: float = 3.0,
) -> bool:
    """Pre-filter: check if a polytope's eigenvalue spectrum is non-random.

    A polytope passes the filter if at least ``t1`` eigenvalues exceed
    the random baseline ``λ_base = 1/d`` by more than ``t0``.

    The random baseline is ``1/d`` because for i.i.d. Bernoulli(½) features
    after StandardScaler, each explained variance ratio ≈ 1/d.

    Parameters
    ----------
    eigen_values : np.ndarray
        Explained variance ratios from :func:`eigenvalue_decomposition`.
    feature_dim : int
        Total feature dimension ``d`` (= number of bits in the feature vector).
        For R1 with polytope size k: ``d = 2(k+1) × WORD_SIZE``.
    t0 : float
        Eigenvalue excess threshold.
    t1 : float
        Minimum number of eigenvalues that must exceed the baseline.

    Returns
    -------
    bool
        ``True`` if the polytope passes the pre-filter.
    """
    lambda_base = 1.0 / feature_dim
    num_exceeding = np.sum(eigen_values - lambda_base > t0)
    return num_exceeding >= t1


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