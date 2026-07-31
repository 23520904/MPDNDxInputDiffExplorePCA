"""calibrate_dus.py
===================

Calibration / validation script for the Distributional-Utility Score (DUS).

For a held-out set of ~15-20 candidate polytopes (known-good ones from the
PDND paper's Table-1 differences plus random draws):

1. Computes ``dus_score`` for each (cheap, n_samples ~ 10^4).
2. Optionally trains an actual PDND distinguisher (--train flag) and records
   validation accuracy.  Skipped by default (--no-train) because full training
   is hardware-intensive.
3. Computes Pearson and Spearman correlation between DUS and accuracy
   (only meaningful when --train is used).
4. Runs ``explore_legacy_pca_kmeans``-derived silhouette scores on the same
   candidates and reports silhouette-vs-accuracy correlation for comparison.
5. Saves a scatter plot ``dus_vs_accuracy.png`` (or ``dus_scores.png`` in
   --no-train mode).

Usage
-----
  # Fast: compute DUS only (no training)
  python src/calibrate_dus.py --rounds 5 --n-samples 10000

  # Full calibration (requires GPU + time):
  python src/calibrate_dus.py --rounds 5 --n-samples 100000 --train --epochs 5

Run from the repository root:
  cd c:\\Users\\dinhl\\Desktop\\Researching\\MPDNDxInputDiffExplorePCA
  python src/calibrate_dus.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from src/
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR if _THIS_DIR.name == "src" else _THIS_DIR / "src"
for _p in [str(_SRC_DIR), str(_SRC_DIR / "analysis")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.distributional_explore import (
    dus_score,
    make_quadruples_adapter,
    ParetoArchive,
)

# ---------------------------------------------------------------------------
# Candidate polytopes
# ---------------------------------------------------------------------------
# Format: list of dicts with 'name', 'pos_diffs' (3-tuple of ints), 'neg_diffs' (3-tuple).
# Known-good candidates derived from PDND paper Table-1 / train.py POS_DELTAS / NEG_DELTAS.
# Each diff element is a 32-bit integer encoding the (left_word, right_word) pair as:
#   val = (left_word << 16) | right_word
# Matching the format used in explore.py's generate_polytope_diff_num output.
#
# POS_DELTAS from train.py: [(16384, 0), (0, 128), (32, 0)]
#   => 16384<<16|0 = 1073741824, 0<<16|128 = 128, 32<<16|0 = 2097152
# NEG_DELTAS from train.py: [(32, 0), (0, 1056), (0, 1026)]
#   => 2097152, 1056, 1026
def _words_to_int(left: int, right: int, wordsize: int = 16) -> int:
    return ((left & 0xFFFF) << wordsize) | (right & 0xFFFF)


KNOWN_POS_DELTA = (
    _words_to_int(16384, 0),   # (0x4000, 0x0000) — from train.py POS_DELTAS[0]
    _words_to_int(0, 128),     # (0x0000, 0x0080)
    _words_to_int(32, 0),      # (0x0020, 0x0000)
)
KNOWN_NEG_DELTA = (
    _words_to_int(32, 0),      # (0x0020, 0x0000) — from train.py NEG_DELTAS[0]
    _words_to_int(0, 1056),    # (0x0000, 0x0420)
    _words_to_int(0, 1026),    # (0x0000, 0x0402)
)

# Fixed reference negative polytope for DUS evaluation
# (used as neg_diffs for ALL candidates to keep comparison fair)
REFERENCE_NEG_DELTA = KNOWN_NEG_DELTA


def _make_random_polytope(rng: np.random.Generator, bit_size: int = 32) -> Tuple[int, int, int]:
    """Generate a random low-HW polytope (HW 1 or 2 per element)."""
    diffs = []
    local: set = set()
    for _ in range(3):
        for _ in range(100):
            hw = int(rng.choice(2)) + 1  # 1 or 2
            positions = rng.choice(bit_size, size=hw, replace=False)
            val = int(sum(1 << int(p) for p in positions))
            if val not in local:
                local.add(val)
                diffs.append(val)
                break
        else:
            diffs.append(1 << int(rng.integers(0, bit_size)))
    return (diffs[0], diffs[1], diffs[2])


def build_candidate_set(n_random: int = 12, rng_seed: int = 42) -> List[Dict]:
    """Build the held-out calibration set."""
    rng = np.random.default_rng(rng_seed)
    candidates = []

    # 1. Paper known-good candidate (POS_DELTAS from train.py)
    candidates.append({
        "name": "paper_pos_delta",
        "pos_diffs": KNOWN_POS_DELTA,
        "source": "paper_table1",
    })

    # 2. Swap pos/neg to get a candidate expected to perform poorly
    candidates.append({
        "name": "paper_neg_as_pos",
        "pos_diffs": KNOWN_NEG_DELTA,
        "source": "paper_neg_as_pos",
    })

    # 3. Several variants near the known-good candidate (bit flips)
    base = list(KNOWN_POS_DELTA)
    for i, bit_pos in enumerate([0, 1, 14, 15, 16, 17]):
        variant = list(base)
        variant[i % 3] ^= (1 << bit_pos)
        if variant[i % 3] == 0:
            variant[i % 3] = 1
        candidates.append({
            "name": f"paper_variant_{i}",
            "pos_diffs": (variant[0], variant[1], variant[2]),
            "source": "bit_flip_variant",
        })

    # 4. Random low-HW draws
    for i in range(n_random):
        poly = _make_random_polytope(rng)
        candidates.append({
            "name": f"random_{i:02d}",
            "pos_diffs": poly,
            "source": "random_low_hw",
        })

    return candidates


# ---------------------------------------------------------------------------
# DUS scoring
# ---------------------------------------------------------------------------

def compute_dus_for_candidates(
    candidates: List[Dict],
    rounds: int,
    n_samples: int,
    n_boot: int = 8,
    weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.05),
    verbose: bool = True,
) -> List[Dict]:
    """Run dus_score on every candidate and attach results."""
    results = []
    for i, cand in enumerate(candidates):
        if verbose:
            print(f"\n[{i+1}/{len(candidates)}] {cand['name']} ({cand['source']})")
            print(f"  pos_diffs = {cand['pos_diffs']}")
        t0 = time.perf_counter()
        try:
            pos_q, neg_q = make_quadruples_adapter(
                pos_delta_tuple=cand["pos_diffs"],
                neg_delta_tuple=REFERENCE_NEG_DELTA,
                rounds=rounds,
                n_samples=n_samples,
            )
            # Ensure equal even size for MMD2
            n_pos = pos_q[0].shape[0]
            n_neg = neg_q[0].shape[0]
            n_use = min(n_pos, n_neg)
            n_use -= n_use % 2
            if n_use < 2:
                raise ValueError(f"Too few samples after split: n_pos={n_pos}, n_neg={n_neg}")
            pos_q = tuple(a[:n_use] for a in pos_q)
            neg_q = tuple(a[:n_use] for a in neg_q)

            score, comps = dus_score(
                pos_q, neg_q,
                delta_tuple=cand["pos_diffs"],
                weights=weights,
                n_boot=n_boot,
            )
            elapsed = time.perf_counter() - t0
            hw = comps["hw_penalty"]
            if verbose:
                print(f"  DUS={score:+.4f}  MMD2={comps['mmd2_mean']:+.4f}  "
                      f"stability={comps['stability']:+.4f}  HW={hw}  ({elapsed:.1f}s)")
            results.append({
                **cand,
                "dus": score,
                "hw_penalty": hw,
                "mmd2_mean": comps["mmd2_mean"],
                "stability": comps["stability"],
                "raw_scores": comps["raw_scores"],
                "elapsed": elapsed,
                "error": None,
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"  ERROR: {exc}")
            results.append({
                **cand,
                "dus": float("nan"),
                "hw_penalty": -1,
                "mmd2_mean": float("nan"),
                "stability": float("nan"),
                "raw_scores": [],
                "elapsed": elapsed,
                "error": str(exc),
            })
    return results


# ---------------------------------------------------------------------------
# Legacy PCA/silhouette scoring (for comparison)
# ---------------------------------------------------------------------------

def compute_legacy_scores_for_candidates(
    candidates: List[Dict],
    rounds: int,
    n_samples: int,
    verbose: bool = True,
) -> List[Optional[float]]:
    """Compute silhouette scores via the legacy PCA+KMeans path for comparison."""
    try:
        import speck32.cipher as speck
        import data_utils.PolytopicQadrupleGenerator as pqg
        import analysis.pca_helper as pca_helper
        import analysis.clustering_helper as clustering_helper
    except ImportError as e:
        if verbose:
            print(f"[WARNING] Legacy scoring unavailable: {e}")
        return [None] * len(candidates)

    scores = []
    for i, cand in enumerate(candidates):
        if verbose:
            print(f"\n  Legacy [{i+1}/{len(candidates)}] {cand['name']}")
        try:
            gen = pqg.PolytopicQuadrupleGenerator(
                encryption_function=speck.encrypt_wrapper,
                plain_bits=32, key_bits=64, nr=rounds,
                pos_diffs=cand["pos_diffs"],
                neg_diffs=REFERENCE_NEG_DELTA,
                feature_mode='diff',
                n_samples=n_samples, batch_size=n_samples,
                use_gpu=False, to_float32=True,
            )
            X, Y = gen[0]
            Y = Y.ravel()
            eigen_value, _ = pca_helper.EigenValueDecomposition(X)
            lambda_base = 1.0 / X.shape[1]
            t0_thresh = 0.003
            t1_thresh = 3
            num_sig = int(np.sum(eigen_value - lambda_base > t0_thresh))
            if num_sig < t1_thresh:
                scores.append(None)
                if verbose:
                    print(f"    Filtered (sig={num_sig} < {t1_thresh})")
                continue
            pca_results = pca_helper.DimensionReduction(X, n_components=3)
            labels = clustering_helper.kmeans_clustering(pca_results, 27, 3)
            sil = float(clustering_helper.calculate_silhouette(pca_results, labels))
            scores.append(sil)
            if verbose:
                print(f"    silhouette={sil:.4f}  sig={num_sig}")
        except Exception as exc:
            scores.append(None)
            if verbose:
                print(f"    ERROR: {exc}")
    return scores


# ---------------------------------------------------------------------------
# Optional training (--train)
# ---------------------------------------------------------------------------

def train_one_candidate(
    cand: Dict,
    rounds: int,
    n_samples: int,
    epochs: int,
    input_size: int = 96,  # 3 * 32 for feature_mode='diff'
) -> Optional[float]:
    """Train a minimal PDND distinguisher and return validation accuracy.

    Uses the make_dataset / create_and_compile_model path from train.py.
    Only imports TF when called; safe to skip in --no-train mode.
    """
    try:
        import tensorflow as tf
        import speck32.train as train_module
        import speck32.model as model_module
    except ImportError as e:
        print(f"[WARNING] TF not available for training: {e}")
        return None

    config = {
        "pos_deltas": cand["pos_diffs"],
        "neg_deltas": REFERENCE_NEG_DELTA,
        "batch_size": 5000,
        "feature_mode": "diff",
        "input_size": input_size,
    }
    try:
        strategy = train_module.get_strategy()
        model = train_module.create_and_compile_model(strategy, input_size)
        dataset, steps = train_module.make_dataset(config, rounds, n_samples)
        history = model.fit(dataset, steps_per_epoch=steps, epochs=epochs, verbose=0)
        val_acc = float(history.history.get("acc", [float("nan")])[-1])
        return val_acc
    except Exception as exc:
        print(f"  [Training ERROR] {exc}")
        return None


# ---------------------------------------------------------------------------
# Correlation & scatter plot
# ---------------------------------------------------------------------------

def compute_correlations(
    results: List[Dict],
    accuracies: List[Optional[float]],
    legacy_silhouettes: List[Optional[float]],
    verbose: bool = True,
) -> Dict:
    """Compute Pearson/Spearman correlations between DUS/silhouette and accuracy."""
    from scipy.stats import pearsonr, spearmanr

    valid_dus = [
        (r["dus"], acc) for r, acc in zip(results, accuracies)
        if acc is not None and np.isfinite(r["dus"])
    ]
    valid_sil = [
        (sil, acc) for sil, acc in zip(legacy_silhouettes, accuracies)
        if acc is not None and sil is not None
    ]

    stats: Dict = {}
    if len(valid_dus) >= 3:
        xs, ys = zip(*valid_dus)
        p_r, p_p = pearsonr(xs, ys)
        s_r, s_p = spearmanr(xs, ys)
        stats["dus_pearson"] = (float(p_r), float(p_p))
        stats["dus_spearman"] = (float(s_r), float(s_p))
        if verbose:
            print(f"\nDUS  vs accuracy: Pearson r={p_r:.3f} (p={p_p:.3f}), "
                  f"Spearman r={s_r:.3f} (p={s_p:.3f})")
    else:
        if verbose:
            print(f"\n[WARNING] Too few paired (DUS, accuracy) points for correlation: "
                  f"{len(valid_dus)} (need ≥ 3).")

    if len(valid_sil) >= 3:
        xs, ys = zip(*valid_sil)
        p_r, p_p = pearsonr(xs, ys)
        s_r, s_p = spearmanr(xs, ys)
        stats["sil_pearson"] = (float(p_r), float(p_p))
        stats["sil_spearman"] = (float(s_r), float(s_p))
        if verbose:
            print(f"Sil  vs accuracy: Pearson r={p_r:.3f} (p={p_p:.3f}), "
                  f"Spearman r={s_r:.3f} (p={s_p:.3f})")
    return stats


def save_scatter_plot(
    results: List[Dict],
    accuracies: List[Optional[float]],
    output_path: str,
    train_mode: bool = False,
) -> None:
    """Save DUS scatter plot (vs accuracy if train_mode, vs HW otherwise)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not available; skipping scatter plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"DUS Calibration — rounds={results[0].get('rounds', '?')} "
        f"({'with accuracy' if train_mode else 'no-train mode'})",
        fontsize=13,
    )

    # --- Left: DUS vs HW penalty ---
    ax = axes[0]
    hws = [r["hw_penalty"] for r in results if np.isfinite(r["dus"])]
    dus_vals = [r["dus"] for r in results if np.isfinite(r["dus"])]
    names = [r["name"] for r in results if np.isfinite(r["dus"])]
    colors = ["steelblue" if r["source"] != "paper_table1" else "crimson"
              for r in results if np.isfinite(r["dus"])]
    ax.scatter(hws, dus_vals, c=colors, alpha=0.8, s=80, edgecolors="k", linewidths=0.5)
    for n, x, y in zip(names, hws, dus_vals):
        ax.annotate(n, (x, y), fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("HW Penalty (sum popcount(δᵢ))")
    ax.set_ylabel("DUS")
    ax.set_title("DUS vs Hamming Weight")
    ax.grid(True, alpha=0.3)

    # --- Right: DUS vs accuracy (if available) or stability ---
    ax = axes[1]
    if train_mode and any(a is not None for a in accuracies):
        xs = [r["dus"] for r, a in zip(results, accuracies)
              if a is not None and np.isfinite(r["dus"])]
        ys = [a for r, a in zip(results, accuracies)
              if a is not None and np.isfinite(r["dus"])]
        ax.scatter(xs, ys, c="steelblue", alpha=0.8, s=80, edgecolors="k", linewidths=0.5)
        ax.set_xlabel("DUS")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title("DUS vs Validation Accuracy")
    else:
        stabs = [r["stability"] for r in results if np.isfinite(r["dus"])]
        ax.scatter(dus_vals, stabs, c="darkorange", alpha=0.8, s=80, edgecolors="k", linewidths=0.5)
        ax.set_xlabel("DUS (composite)")
        ax.set_ylabel("Stability score")
        ax.set_title("DUS vs Stability (no-train mode)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nScatter plot saved to: {output_path}")


def save_csv(
    results: List[Dict],
    accuracies: List[Optional[float]],
    legacy_silhouettes: List[Optional[float]],
    output_path: str,
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "source", "pos_diffs",
            "dus", "mmd2_mean", "stability", "hw_penalty",
            "accuracy", "legacy_silhouette", "elapsed_s", "error",
        ])
        for r, acc, sil in zip(results, accuracies, legacy_silhouettes):
            writer.writerow([
                r["name"], r["source"], str(r["pos_diffs"]),
                r["dus"], r.get("mmd2_mean", ""), r.get("stability", ""),
                r.get("hw_penalty", ""),
                acc if acc is not None else "",
                sil if sil is not None else "",
                r.get("elapsed", ""), r.get("error", ""),
            ])
    print(f"CSV results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate DUS scorer against held-out polytope candidates."
    )
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of cipher rounds (default: 5)")
    parser.add_argument("--n-samples", type=int, default=10_000,
                        help="Samples per DUS evaluation (default: 10000)")
    parser.add_argument("--n-boot", type=int, default=8,
                        help="Bootstrap resamples for DUS (default: 8)")
    parser.add_argument("--n-random", type=int, default=12,
                        help="Number of random candidates to include (default: 12)")
    parser.add_argument("--train", action="store_true", default=False,
                        help="Actually train PDND distinguishers (slow; hardware-intensive)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs per candidate when --train is used (default: 3)")
    parser.add_argument("--legacy", action="store_true", default=True,
                        help="Also compute legacy PCA/silhouette scores (default: on)")
    parser.add_argument("--no-legacy", action="store_false", dest="legacy",
                        help="Skip legacy PCA/silhouette scoring")
    parser.add_argument("--output-dir", type=str, default="results/calibrate_dus",
                        help="Output directory for CSV and plots")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 72)
    print("  DUS Calibration Script")
    print(f"  rounds={args.rounds}  n_samples={args.n_samples}  "
          f"train={args.train}  legacy={args.legacy}")
    print("=" * 72)

    # 1. Build candidate set
    candidates = build_candidate_set(n_random=args.n_random, rng_seed=args.seed)
    print(f"\nBuilt {len(candidates)} candidates.")

    # 2. DUS scoring
    print("\n--- DUS Scoring ---")
    results = compute_dus_for_candidates(
        candidates, rounds=args.rounds, n_samples=args.n_samples,
        n_boot=args.n_boot, verbose=True,
    )
    # Attach rounds for plot title
    for r in results:
        r["rounds"] = args.rounds

    # 3. Legacy PCA/silhouette (optional)
    legacy_silhouettes: List[Optional[float]] = [None] * len(candidates)
    if args.legacy:
        print("\n--- Legacy PCA / Silhouette Scoring ---")
        legacy_silhouettes = compute_legacy_scores_for_candidates(
            candidates, rounds=args.rounds, n_samples=args.n_samples, verbose=True,
        )

    # 4. Training (optional, --train)
    accuracies: List[Optional[float]] = [None] * len(candidates)
    if args.train:
        print("\n--- Training PDND Distinguishers ---")
        print("  [NOTE] This is slow. Use --no-train for a quick DUS-only run.")
        for i, cand in enumerate(candidates):
            print(f"\n  Training [{i+1}/{len(candidates)}] {cand['name']} ...")
            acc = train_one_candidate(
                cand, rounds=args.rounds, n_samples=args.n_samples, epochs=args.epochs,
            )
            accuracies[i] = acc
            print(f"  val_acc = {acc}")
    else:
        print("\n[--no-train mode] Skipping PDND training.")
        print("  Re-run with --train --epochs N to get accuracy numbers for correlation.")

    # 5. Correlations
    if args.train and any(a is not None for a in accuracies):
        _ = compute_correlations(results, accuracies, legacy_silhouettes, verbose=True)
    else:
        print("\n[Correlation] Skipped (no accuracy data; use --train to enable).")

    # 6. Print DUS ranking table
    print("\n" + "=" * 72)
    print("  DUS Ranking (all candidates)")
    print("=" * 72)
    sorted_res = sorted(results, key=lambda x: x["dus"] if np.isfinite(x["dus"]) else -1e9,
                        reverse=True)
    print(f"  {'Rank':>4}  {'Name':<25}  {'DUS':>8}  {'HW':>3}  {'MMD2':>8}  {'Stability':>9}")
    print("  " + "-" * 64)
    for rank, r in enumerate(sorted_res, 1):
        print(f"  {rank:>4}  {r['name']:<25}  {r['dus']:>8.4f}  "
              f"{r.get('hw_penalty', '?'):>3}  "
              f"{r.get('mmd2_mean', float('nan')):>8.4f}  "
              f"{r.get('stability', float('nan')):>9.4f}")

    # 7. Save outputs
    csv_path = os.path.join(args.output_dir, f"calibration_{timestamp}.csv")
    plot_path = os.path.join(args.output_dir, f"dus_vs_accuracy_{timestamp}.png")
    save_csv(results, accuracies, legacy_silhouettes, csv_path)
    save_scatter_plot(results, accuracies, plot_path, train_mode=args.train)

    print("\nDone.")


if __name__ == "__main__":
    main()
