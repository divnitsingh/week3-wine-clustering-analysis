# Week 3 — Unsupervised Learning and Clustering Analysis

A GitHub-ready, hands-on Python/Jupyter project implementing the Week 3 internship task with **K-Means and Hierarchical Clustering**.

## Project objective

Segment wine samples into meaningful groups using physicochemical measurements, without using the target class during clustering.

The project demonstrates:

- Public data acquisition from the UCI Machine Learning Repository.
- Data inspection and validation.
- Feature-only clustering preparation.
- Standardization to make distance-based algorithms comparable across variables.
- K-Means clustering.
- Elbow analysis for candidate values of `k`.
- Silhouette analysis for model selection.
- Hierarchical clustering as a complementary method.
- PCA visualization in two dimensions.
- Cluster-size and centroid analysis.
- Feature-level cluster profiling.
- Interpretation and practical implications.
- Automated tests and reproducible outputs.

## Dataset

**Dataset:** Wine  
**Provider:** UCI Machine Learning Repository  
**Official dataset page:**  
https://archive.ics.uci.edu/dataset/109/wine

The dataset contains 178 wine samples and 13 continuous chemical measurements. The original dataset also contains a cultivar/class label. **That label is not used to train the clustering models.** It is kept separately only for optional post-hoc comparison.

## Project structure

```text
week3_wine_clustering_analysis/
├── README.md
├── DATA_ATTRIBUTION.md
├── requirements.txt
├── run_clustering.py
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
│   └── week3_clustering_analysis.ipynb
├── src/
│   └── week3_clustering.py
├── tests/
│   └── test_clustering.py
└── outputs/
    ├── figures/
    └── tables/
```

## Setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

### Full live-data workflow
```bash
python run_clustering.py
```

### Offline workflow
```bash
python run_clustering.py --sample
```

The live workflow downloads the public UCI Wine dataset. If internet access is unavailable, the project automatically falls back to the bundled sample fixture.

## Outputs

The pipeline creates:

### Processed data
- `data/processed/wine_clustering_ready.csv`
- `data/processed/wine_clustered.csv`

### Tables
- `outputs/tables/cluster_metrics.csv`
- `outputs/tables/cluster_profile.csv`
- `outputs/tables/cluster_sizes.csv`
- `outputs/tables/hierarchical_comparison.csv`

### Figures
- `outputs/figures/01_elbow_curve.png`
- `outputs/figures/02_silhouette_scores.png`
- `outputs/figures/03_pca_kmeans_clusters.png`
- `outputs/figures/04_cluster_profile_heatmap.png`
- `outputs/figures/05_feature_boxplots.png`
- `outputs/figures/06_hierarchical_dendrogram.png`
- `outputs/figures/07_kmeans_vs_hierarchical.png`

## Methodology

### 1. Feature selection
The 13 numerical wine chemistry measurements are used for clustering. The cultivar/class label is excluded from the unsupervised learning input.

### 2. Standardization
Features are transformed with `StandardScaler`. This matters because K-Means and hierarchical clustering are distance-based and raw variables have different numerical scales.

### 3. Choosing the number of clusters
The workflow evaluates a small set of candidate cluster counts and reports both:

- **Inertia / elbow curve:** lower is better, but improvements diminish after a point.
- **Silhouette score:** higher values indicate better within-cluster cohesion and between-cluster separation.

For the full UCI Wine dataset, the analysis commonly identifies **3 clusters** as a strong choice. The code still computes the metrics dynamically rather than hard-coding a result.

### 4. K-Means
The final K-Means model uses:
- `n_clusters = selected k`
- `n_init = 20`
- deterministic `random_state`

### 5. Hierarchical clustering
Agglomerative clustering with Ward linkage is fitted as a complementary method. Agreement is measured with **Adjusted Rand Index (ARI)** between the two unsupervised partitions.

### 6. Visualization
PCA reduces the standardized 13-dimensional feature space to two components for a 2-D visual summary. Cluster profiles are also visualized using a heatmap of standardized feature means.

## Interpretation guidance

Clusters are **segments of similarity**, not automatically meaningful business categories. Interpretation should focus on:

- Which features are relatively high or low in each cluster.
- How cluster sizes differ.
- Whether the groups are well separated.
- Whether the solution is stable across algorithms.
- Whether domain knowledge supports a meaningful interpretation.

The original class labels may be compared after clustering, but they must not be used to form the clusters.

## Tests

Run:

```bash
python -m pytest -q
```

The tests cover:
- Feature scaling.
- Valid cluster labels.
- Metric table structure.
- PCA output.
- Cluster-profile completeness.
- Hierarchical comparison.

## Responsible interpretation

This project is educational. Clustering results depend on feature selection, scaling, distance metric, algorithm, and selected cluster count. A visually separated cluster is not proof of a real-world causal or categorical distinction.

## Attribution

UCI Machine Learning Repository, Wine dataset.

See `DATA_ATTRIBUTION.md` for source and attribution information.
