from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from week3_clustering import (
    FEATURE_NAMES,
    cluster_profile,
    cluster_sizes,
    compare_clusterings,
    evaluate_k_values,
    fit_hierarchical,
    fit_kmeans,
    load_wine_csv,
    pca_projection,
    prepare_features,
    scale_features,
    select_k,
)


SAMPLE = PROJECT_ROOT / "data" / "sample" / "wine_sample.csv"


def dataset():
    return load_wine_csv(SAMPLE)


def test_feature_preparation_has_expected_columns():
    X, y = prepare_features(dataset())
    assert list(X.columns) == FEATURE_NAMES
    assert len(X) == 18
    assert len(y) == 18


def test_standardization_produces_unit_scale():
    X, _ = prepare_features(dataset())
    X_scaled, _ = scale_features(X)
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-7)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-7)


def test_k_selection_and_labels():
    X, _ = prepare_features(dataset())
    X_scaled, _ = scale_features(X)
    metrics = evaluate_k_values(X_scaled, [2, 3, 4, 5])
    k = select_k(metrics)
    assert 2 <= k <= 5

    model, labels = fit_kmeans(X_scaled, k)
    assert len(labels) == len(X)
    assert len(np.unique(labels)) == k
    assert model.n_clusters == k


def test_cluster_profile_and_sizes():
    X, _ = prepare_features(dataset())
    X_scaled, _ = scale_features(X)
    _, labels = fit_kmeans(X_scaled, 3)

    profile = cluster_profile(X_scaled, labels)
    sizes = cluster_sizes(labels)

    assert profile.shape == (3, 13)
    assert sizes["count"].sum() == len(X)
    assert set(sizes["cluster"]) == {0, 1, 2}


def test_pca_and_hierarchical_comparison():
    X, _ = prepare_features(dataset())
    X_scaled, _ = scale_features(X)
    _, km_labels = fit_kmeans(X_scaled, 3)
    _, hc_labels = fit_hierarchical(X_scaled, 3)

    projection, pca = pca_projection(X_scaled)
    comparison = compare_clusterings(km_labels, hc_labels)

    assert projection.shape == (len(X), 2)
    assert len(pca.components_) == 2
    assert comparison.loc[0, "metric"] == "adjusted_rand_index"
    assert -1.0 <= comparison.loc[0, "value"] <= 1.0
