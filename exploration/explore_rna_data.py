import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "sample_type_id"
EPSILON = 1e-5
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Liver RNA Data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "exploration"


def load_and_prepare_dataframe(data_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(data_path)
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    features = dataframe.drop(columns=[TARGET_COLUMN])
    non_constant_mask = features.var(axis=0) > 0
    cleaned_features = features.loc[:, non_constant_mask]

    prepared_dataframe = cleaned_features.copy()
    prepared_dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN]
    return prepared_dataframe


def compute_embeddings(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_frame = np.log(dataframe.drop(columns=[TARGET_COLUMN]).astype(float) + EPSILON)

    pca = PCA(n_components=3)
    pca_embedding = pca.fit_transform(feature_frame)

    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE)
    tsne_embedding = tsne.fit_transform(feature_frame)

    return pca_embedding, tsne_embedding


def run_anova_on_principal_components(
    dataframe: pd.DataFrame,
    n_components: int,
    alpha: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from statsmodels.formula.api import ols
        import statsmodels.api as sm
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ANOVA requires statsmodels. Install it with: pip install statsmodels"
        ) from error

    feature_frame = np.log(dataframe.drop(columns=[TARGET_COLUMN]).astype(float) + EPSILON)
    target = dataframe[TARGET_COLUMN]

    component_count = min(n_components, feature_frame.shape[0], feature_frame.shape[1])
    if component_count < 1:
        raise ValueError("Not enough data to compute PCA components for ANOVA.")

    pca = PCA(n_components=component_count)
    pca_scores = pca.fit_transform(feature_frame)

    pc_columns = [f"PC{i + 1}" for i in range(component_count)]
    pca_frame = pd.DataFrame(pca_scores, columns=pc_columns)
    pca_frame["target"] = target.values

    records = []
    for pc in pc_columns:
        model = ols(f"{pc} ~ C(target)", data=pca_frame).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        records.append(
            {
                "pc": pc,
                "f_statistic": float(anova_table.loc["C(target)", "F"]),
                "p_value": float(anova_table.loc["C(target)", "PR(>F)"]),
            }
        )

    anova_results = pd.DataFrame(records).sort_values("p_value").reset_index(drop=True)
    significant = anova_results[anova_results["p_value"] < alpha].copy()
    significant["pc_index"] = significant["pc"].str.replace("PC", "", regex=False).astype(int) - 1

    output_dir.mkdir(parents=True, exist_ok=True)
    anova_results.to_csv(output_dir / "anova_results.csv", index=False)
    significant.to_csv(output_dir / "anova_significant_pcs.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_frame.columns,
        columns=pc_columns,
    )
    loadings.to_csv(output_dir / "pca_loadings.csv")

    return significant, loadings


def build_top_genes_by_pc(
    significant_pcs: pd.DataFrame,
    loadings: pd.DataFrame,
    top_k_genes: int,
) -> dict[str, list[str]]:
    top_genes_by_pc: dict[str, list[str]] = {}

    for _, row in significant_pcs.iterrows():
        pc_name = row["pc"]
        absolute_loadings = loadings[pc_name].abs().sort_values(ascending=False)
        top_genes_by_pc[pc_name] = absolute_loadings.head(top_k_genes).index.tolist()

    return top_genes_by_pc


def run_gprofiler_enrichment(
    top_genes_by_pc: dict[str, list[str]],
    output_dir: Path,
) -> None:
    try:
        from gprofiler import GProfiler
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "g:Profiler enrichment requires gprofiler-official. "
            "Install it with: pip install gprofiler-official"
        ) from error

    profiler = GProfiler(return_dataframe=True)

    rows = []
    for pc_name, genes in top_genes_by_pc.items():
        enrichment = profiler.profile(
            organism="hsapiens",
            query=genes,
            sources=["GO:BP", "KEGG", "REAC"],
        )
        if enrichment is None or enrichment.empty:
            continue

        subset = enrichment[["source", "native", "name", "p_value"]].copy()
        subset.insert(0, "pc", pc_name)
        rows.append(subset)

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(output_dir / "gprofiler_enrichment.csv", index=False)


def evaluate_baseline(embedding: np.ndarray, target: pd.Series, label: str) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        embedding,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    classifier = RandomForestClassifier(random_state=RANDOM_STATE)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    accuracy = classifier.score(X_test, y_test)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    metrics = {
        "embedding": label,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
    }
    return metrics


def save_plots(
    dataframe: pd.DataFrame,
    pca_embedding: np.ndarray,
    tsne_embedding: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    target = dataframe[TARGET_COLUMN]

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=target, s=12)
    plt.title("PCA Projection of RNA Samples")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_scatter.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=target, s=12)
    plt.title("t-SNE Projection of RNA Samples")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_scatter.png", dpi=150)
    plt.close()

    variance_log = np.log(dataframe.drop(columns=[TARGET_COLUMN]).var(axis=0) + EPSILON)
    plt.figure(figsize=(8, 6))
    plt.hist(variance_log, bins=100)
    plt.title("Log Variance Distribution of Gene Features")
    plt.xlabel("log(variance)")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "variance_histogram.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory analysis on TCGA liver RNA data")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to CSV data file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and metrics will be written",
    )
    parser.add_argument(
        "--run-anova",
        action="store_true",
        help="Run ANOVA on PCA components and save significance tables",
    )
    parser.add_argument(
        "--run-enrichment",
        action="store_true",
        help="Run g:Profiler enrichment (requires --run-anova)",
    )
    parser.add_argument(
        "--anova-components",
        type=int,
        default=100,
        help="Maximum PCA components to test in ANOVA",
    )
    parser.add_argument(
        "--anova-alpha",
        type=float,
        default=0.05,
        help="Significance threshold for ANOVA",
    )
    parser.add_argument(
        "--top-k-genes",
        type=int,
        default=10,
        help="Number of top-loading genes used per significant PC for enrichment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_enrichment and not args.run_anova:
        raise ValueError("--run-enrichment requires --run-anova.")

    dataframe = load_and_prepare_dataframe(args.data_path)
    pca_embedding, tsne_embedding = compute_embeddings(dataframe)

    target = dataframe[TARGET_COLUMN]
    metrics = [
        evaluate_baseline(pca_embedding, target, "PCA"),
        evaluate_baseline(tsne_embedding, target, "tSNE"),
    ]

    save_plots(dataframe, pca_embedding, tsne_embedding, args.output_dir)

    metrics_frame = pd.DataFrame(metrics)
    metrics_path = args.output_dir / "baseline_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)

    if args.run_anova:
        significant_pcs, loadings = run_anova_on_principal_components(
            dataframe=dataframe,
            n_components=args.anova_components,
            alpha=args.anova_alpha,
            output_dir=args.output_dir,
        )
        print(f"- ANOVA results: {args.output_dir / 'anova_results.csv'}")
        print(f"- Significant PCs: {args.output_dir / 'anova_significant_pcs.csv'}")
        print(f"- PCA loadings: {args.output_dir / 'pca_loadings.csv'}")

        if args.run_enrichment:
            top_genes_by_pc = build_top_genes_by_pc(
                significant_pcs=significant_pcs,
                loadings=loadings,
                top_k_genes=args.top_k_genes,
            )
            run_gprofiler_enrichment(top_genes_by_pc=top_genes_by_pc, output_dir=args.output_dir)
            print(f"- g:Profiler enrichment: {args.output_dir / 'gprofiler_enrichment.csv'}")

    print("Saved exploration outputs:")
    print(f"- Plots in: {args.output_dir}")
    print(f"- Metrics: {metrics_path}")
    for row in metrics:
        print(
            f"- {row['embedding']}: "
            f"accuracy={row['accuracy']:.4f}, balanced_accuracy={row['balanced_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
