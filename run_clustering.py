from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from week3_clustering import (
    FEATURE_NAMES,
    UCI_RAW_URL,
    cluster_profile,
    cluster_sizes,
    compare_clusterings,
    create_figures,
    download_wine_data,
    evaluate_k_values,
    fit_hierarchical,
    fit_kmeans,
    load_wine_csv,
    prepare_features,
    scale_features,
    select_k,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 3 wine clustering analysis.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use the bundled sample instead of downloading live UCI data.",
    )
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"

    raw_path = (
        PROJECT_ROOT / "data" / "sample" / "wine_sample.csv"
        if args.sample else
        raw_dir / "wine.data"
    )

    try:
        if not args.sample:
            if not raw_path.exists():
                download_wine_data(raw_path)
        df = load_wine_csv(raw_path)
        acquisition_mode = "sample" if args.sample else "official UCI"
    except Exception as exc:
        print(f"Live UCI download unavailable ({exc}); using bundled sample.")
        df = load_wine_csv(PROJECT_ROOT / "data" / "sample" / "wine_sample.csv")
        acquisition_mode = "sample-fallback"

    X, y = prepare_features(df)
    X_scaled, scaler = scale_features(X)

    metrics = evaluate_k_values(
        X_scaled,
        k_values=list(range(2, 8)),
        random_state=42,
    )
    selected_k = select_k(metrics)

    kmeans, kmeans_labels = fit_kmeans(X_scaled, selected_k, random_state=42)
    hierarchical, hierarchical_labels = fit_hierarchical(X_scaled, selected_k)

    profile = cluster_profile(X_scaled, kmeans_labels, FEATURE_NAMES)
    sizes = cluster_sizes(kmeans_labels)
    comparison = compare_clusterings(kmeans_labels, hierarchical_labels)

    # Save processed data.
    processed_dir.mkdir(parents=True, exist_ok=True)
    clustered = X.copy()
    clustered["cluster_kmeans"] = kmeans_labels
    clustered["cluster_hierarchical"] = hierarchical_labels
    clustered["class_reference_only"] = y.astype("Int64")
    clustered.to_csv(processed_dir / "wine_clustered.csv", index=False)

    X_out = X.copy()
    X_out.to_csv(processed_dir / "wine_clustering_ready.csv", index=False)

    # Save tables.
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables_dir / "cluster_metrics.csv", index=False)
    profile.to_csv(tables_dir / "cluster_profile.csv")
    sizes.to_csv(tables_dir / "cluster_sizes.csv", index=False)
    comparison.to_csv(tables_dir / "hierarchical_comparison.csv", index=False)

    # Save figures.
    create_figures(
        X_scaled,
        metrics,
        kmeans_labels,
        profile,
        hierarchical_labels,
        figures_dir,
    )

    print("\\nClustering complete.")
    print("Acquisition mode:", acquisition_mode)
    print("Observations:", len(X))
    print("Features:", len(FEATURE_NAMES))
    print("Selected k:", selected_k)
    print("Best silhouette:", f"{metrics.loc[metrics['k'].eq(selected_k), 'silhouette_score'].iloc[0]:.4f}")
    print("Cluster sizes:")
    print(sizes.to_string(index=False))
    print("Outputs written under data/processed and outputs/.")


if __name__ == "__main__":
    main()
