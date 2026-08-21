from __future__ import annotations

import io
from pathlib import Path
from urllib.error import URLError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


UCI_RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

FEATURE_NAMES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]

COLUMN_NAMES = FEATURE_NAMES + ["class"]


def download_wine_data(destination: Path, timeout: int = 30) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        UCI_RAW_URL,
        timeout=timeout,
        headers={"User-Agent": "week3-wine-clustering-project/1.0"},
    )
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def load_wine_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    X = work[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(work["class"], errors="coerce").astype("Int64")

    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing/non-numeric values.")
    if not X.columns.equals(pd.Index(FEATURE_NAMES)):
        raise ValueError("Unexpected feature columns.")
    if X.shape[0] < 3:
        raise ValueError("At least 3 observations are required.")

    return X, y


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def evaluate_k_values(
    X_scaled: np.ndarray,
    k_values: list[int],
    random_state: int = 42,
) -> pd.DataFrame:
    records = []
    for k in k_values:
        if k < 2 or k > len(X_scaled) - 1:
            continue
        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=random_state,
        )
        labels = model.fit_predict(X_scaled)
        records.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(X_scaled, labels)),
            }
        )
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("No valid k values were supplied.")
    return result


def select_k(metrics: pd.DataFrame) -> int:
    # Choose the highest silhouette score; in ties choose smaller k.
    best = (
        metrics.sort_values(
            by=["silhouette_score", "k"],
            ascending=[False, True],
        )
        .iloc[0]
    )
    return int(best["k"])


def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    model = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=random_state,
    )
    labels = model.fit_predict(X_scaled)
    return model, labels


def fit_hierarchical(
    X_scaled: np.ndarray,
    n_clusters: int,
) -> tuple[AgglomerativeClustering, np.ndarray]:
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    )
    labels = model.fit_predict(X_scaled)
    return model, labels


def pca_projection(X_scaled: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(X_scaled)
    return projection, pca


def cluster_profile(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    cols = feature_names or FEATURE_NAMES
    standardized = pd.DataFrame(X_scaled, columns=cols)
    standardized["cluster"] = labels
    profile = standardized.groupby("cluster", as_index=True)[cols].mean()
    return profile


def cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    s = pd.Series(labels, name="cluster")
    result = s.value_counts().sort_index().rename("count").reset_index()
    result["percentage"] = result["count"] / result["count"].sum() * 100
    return result


def compare_clusterings(kmeans_labels: np.ndarray, hierarchical_labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["adjusted_rand_index"],
            "value": [float(adjusted_rand_score(kmeans_labels, hierarchical_labels))],
        }
    )


def create_figures(
    X_scaled: np.ndarray,
    metrics: pd.DataFrame,
    kmeans_labels: np.ndarray,
    profile: pd.DataFrame,
    hierarchical_labels: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    created = []

    # Elbow
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(metrics["k"], metrics["inertia"], marker="o")
    ax.set_title("K-Means Elbow Curve")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    fig.tight_layout()
    p = output_dir / "01_elbow_curve.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    created.append(p)

    # Silhouette
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(metrics["k"], metrics["silhouette_score"], marker="o")
    ax.set_title("Silhouette Score by Number of Clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    fig.tight_layout()
    p = output_dir / "02_silhouette_scores.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    created.append(p)

    # PCA
    projection, pca = pca_projection(X_scaled)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=kmeans_labels,
        s=40,
        alpha=0.85,
    )
    ax.set_title(
        "K-Means Clusters in PCA Space "
        f"({pca.explained_variance_ratio_[0]*100:.1f}% + "
        f"{pca.explained_variance_ratio_[1]*100:.1f}% variance)"
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    p = output_dir / "03_pca_kmeans_clusters.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    created.append(p)

    # Profile heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(profile, cmap="vlag", center=0, ax=ax)
    ax.set_title("Cluster Profile: Mean Standardized Feature Values")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    p = output_dir / "04_cluster_profile_heatmap.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    created.append(p)

    # Representative boxplot of cluster distances/profile scores
    profile_long = (
        profile.reset_index()
        .melt(id_vars="cluster", var_name="feature", value_name="mean_z")
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=profile_long, x="cluster", y="mean_z", ax=ax)
    ax.set_title("Distribution of Standardized Cluster-Profile Feature Means")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean standardized feature value")
    fig.tight_layout()
    p = output_dir / "05_feature_boxplots.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    created.append(p)

    # Dendrogram
    linkage_matrix = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, no_labels=True, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    p = output_dir / "06_hierarchical_dendrogram.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    created.append(p)

    # KMeans vs hierarchical in PCA
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(projection[:, 0], projection[:, 1], c=kmeans_labels, s=35)
    axes[0].set_title("K-Means")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].scatter(projection[:, 0], projection[:, 1], c=hierarchical_labels, s=35)
    axes[1].set_title("Hierarchical")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.suptitle("Comparison of Unsupervised Cluster Assignments")
    fig.tight_layout()
    p = output_dir / "07_kmeans_vs_hierarchical.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    created.append(p)

    return created
