"""
Penguin clustering pipeline steps. Plain functions with no Airflow imports, so they can be
unit tested and run without a scheduler; the DAG in penguins_dag.py wires them together.

Steps pass file paths and small JSON-safe dicts to each other (never pickled DataFrames),
so nothing large or binary goes through Airflow's XCom.
"""
import base64
import io
import json
import os
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # no display inside Airflow workers
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA = DATA_DIR / "penguins.csv"
NEW_PENGUINS = DATA_DIR / "new_penguins.csv"
# Inside Docker this is /opt/airflow/output (mounted to Lab_1/output); locally Lab_1/output
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", Path(__file__).resolve().parents[2] / "output"))

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
LABEL = "species"  # only used to evaluate the clusters, never to fit them
REQUIRED_COLUMNS = FEATURES + [LABEL, "island", "sex"]
# Physically plausible ranges; values outside them mean a data error, not a big penguin
VALID_RANGES = {
    "bill_length_mm": (25, 70),
    "bill_depth_mm": (10, 25),
    "flipper_length_mm": (160, 240),
    "body_mass_g": (2500, 6500),
}
MIN_ROWS = 300
MAX_MISSING_FRACTION = 0.05
RANDOM_STATE = 42


def run_dir(run_id):
    """Folder for one DAG run's files; run ids contain ':' and '+', which aren't safe in paths."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
    path = OUTPUT_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_data(path=RAW_DATA):
    """
    Data quality gate: check the raw file before any modelling.
    Returns:
        dict: Row count and missing values per feature.
    Raises:
        ValueError: Listing every problem found, which fails the Airflow task.
    """
    df = pd.read_csv(path)
    problems = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows, expected at least {MIN_ROWS}")
    missing = df[FEATURES].isna().sum()
    for col, n in missing.items():
        if n / len(df) > MAX_MISSING_FRACTION:
            problems.append(f"{col} is {n / len(df):.1%} missing (max {MAX_MISSING_FRACTION:.0%})")
    for col, (low, high) in VALID_RANGES.items():
        bad = df[col].dropna().pipe(lambda s: s[(s < low) | (s > high)])
        if len(bad):
            problems.append(f"{len(bad)} {col} value(s) outside {low}-{high}")
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
    return {"rows": len(df), "missing": {k: int(v) for k, v in missing.items()}}


def preprocess(out_dir, path=RAW_DATA):
    """
    Drop penguins with missing measurements and save the clean data.
    Returns:
        dict: Path of the clean CSV, rows kept and rows dropped.
    """
    df = pd.read_csv(path)
    clean = df.dropna(subset=FEATURES).reset_index(drop=True)
    clean_path = Path(out_dir) / "clean.csv"
    clean.to_csv(clean_path, index=False)
    return {"clean_path": str(clean_path), "rows": len(clean), "dropped": len(df) - len(clean)}


def select_k(clean_path, k_min=2, k_max=10):
    """
    Choose the number of clusters with the Gaussian Mixture BIC (lower is better), and also
    record what the original lab's K-Means elbow and the silhouette score would pick.
    Returns:
        dict: best_k (by BIC), kmeans_elbow_k, best_silhouette_k and the score for every k.
    """
    if not 2 <= k_min < k_max:
        raise ValueError(f"Need 2 <= k_min < k_max, got k_min={k_min}, k_max={k_max}")
    X = StandardScaler().fit_transform(pd.read_csv(clean_path)[FEATURES])
    scores = []
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, n_init=5, random_state=RANDOM_STATE).fit(X)
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        scores.append({
            "k": k,
            "gmm_bic": round(float(gmm.bic(X)), 1),
            "kmeans_sse": round(float(km.inertia_), 1),
            "kmeans_silhouette": round(float(silhouette_score(X, km.labels_)), 4),
        })
    ks = [s["k"] for s in scores]
    elbow = KneeLocator(ks, [s["kmeans_sse"] for s in scores], curve="convex", direction="decreasing").elbow
    return {
        "best_k": min(scores, key=lambda s: s["gmm_bic"])["k"],
        "kmeans_elbow_k": int(elbow) if elbow is not None else None,
        "best_silhouette_k": max(scores, key=lambda s: s["kmeans_silhouette"])["k"],
        "scores": scores,
    }


def train_model(clean_path, k, out_dir):
    """
    Fit StandardScaler + GaussianMixture(k) as one pipeline, so new penguins are scaled the
    same way at prediction time, and save it.
    Returns:
        str: Path of the saved model.
    """
    X = pd.read_csv(clean_path)[FEATURES]
    model = make_pipeline(StandardScaler(),
                          GaussianMixture(n_components=k, n_init=5, random_state=RANDOM_STATE))
    model.fit(X)
    model_path = Path(out_dir) / "model.joblib"
    joblib.dump(model, model_path)
    return str(model_path)


def evaluate_model(model_path, clean_path, out_dir):
    """
    Compare the clusters with the real species (which the model never saw).
    Returns:
        dict: ARI and silhouette for the model and for a K-Means baseline with the same k,
              cluster sizes, the species each cluster mostly contains, purity and profiles.
    """
    model = joblib.load(model_path)
    df = pd.read_csv(clean_path)
    labels = model.predict(df[FEATURES])
    k = model[-1].n_components
    X_scaled = model[0].transform(df[FEATURES])
    kmeans_labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_scaled)

    crosstab = pd.crosstab(labels, df[LABEL])
    # Keys are strings because XCom stores these dicts as JSON
    cluster_species = {str(c): str(crosstab.loc[c].idxmax()) for c in crosstab.index}
    profiles = df.assign(cluster=labels).groupby("cluster")[FEATURES].mean().round(1)
    metrics = {
        "k": int(k),
        "rows": len(df),
        "gmm_ari": round(float(adjusted_rand_score(df[LABEL], labels)), 4),
        "gmm_silhouette": round(float(silhouette_score(X_scaled, labels)), 4),
        "kmeans_ari": round(float(adjusted_rand_score(df[LABEL], kmeans_labels)), 4),
        "kmeans_silhouette": round(float(silhouette_score(X_scaled, kmeans_labels)), 4),
        "purity": round(float(crosstab.max(axis=1).sum() / len(df)), 4),
        "cluster_sizes": {str(c): int(n) for c, n in pd.Series(labels).value_counts().sort_index().items()},
        "cluster_species": cluster_species,
        "crosstab": {str(c): {str(s): int(n) for s, n in row.items()} for c, row in crosstab.iterrows()},
        "profiles": {str(c): row.to_dict() for c, row in profiles.iterrows()},
    }
    (Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def predict_new(model_path, cluster_species, out_dir, path=NEW_PENGUINS):
    """
    Assign new, unlabelled penguins to clusters, with the species that cluster mostly holds
    and how confident the mixture model is.
    Returns:
        list[dict]: One prediction per penguin.
    """
    model = joblib.load(model_path)
    new = pd.read_csv(path)
    probs = model.predict_proba(new[FEATURES])
    predictions = [
        {**{f: float(row[f]) for f in FEATURES},
         "cluster": int(p.argmax()),
         "likely_species": cluster_species[str(p.argmax())],
         "confidence": round(float(p.max()), 3)}
        for (_, row), p in zip(new.iterrows(), probs)
    ]
    (Path(out_dir) / "predictions.json").write_text(json.dumps(predictions, indent=2))
    return predictions


def _chart(selection):
    """BIC, K-Means SSE and silhouette against k, as a base64 PNG for the HTML report."""
    ks = [s["k"] for s in selection["scores"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [("gmm_bic", "Gaussian Mixture BIC (lower is better)", selection["best_k"]),
              ("kmeans_sse", "K-Means SSE (elbow method)", selection["kmeans_elbow_k"]),
              ("kmeans_silhouette", "K-Means silhouette (higher is better)", selection["best_silhouette_k"])]
    for ax, (key, title, chosen) in zip(axes, panels):
        ax.plot(ks, [s[key] for s in selection["scores"]], marker="o", color="#2b6cb0")
        if chosen is not None:
            ax.axvline(chosen, color="#c05621", linestyle="--", label=f"picks k={chosen}")
            ax.legend()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("number of clusters k")
        ax.set_xticks(ks)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_report(selection, metrics, predictions, out_dir, status):
    """
    Write a self-contained HTML dashboard of the run (charts embedded, no external files).
    Returns:
        str: Path of report.html.
    """
    def table(rows, headers):
        head = "".join(f"<th>{h}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows)
        return f"<table><tr>{head}</tr>{body}</table>"

    species = sorted({s for row in metrics["crosstab"].values() for s in row})
    clusters = table(
        [[c, metrics["cluster_sizes"][c], metrics["cluster_species"][c],
          *(metrics["crosstab"][c].get(s, 0) for s in species),
          *(metrics["profiles"][c][f] for f in FEATURES)] for c in sorted(metrics["cluster_sizes"], key=int)],
        ["Cluster", "Size", "Mostly", *species, *FEATURES])
    compare = table(
        [["Gaussian Mixture (this lab)", metrics["k"], metrics["gmm_ari"], metrics["gmm_silhouette"]],
         ["K-Means baseline (original lab's model)", metrics["k"], metrics["kmeans_ari"], metrics["kmeans_silhouette"]]],
        ["Model", "k", "Adjusted Rand Index vs species", "Silhouette"])
    preds = table([[*(p[f] for f in FEATURES), p["cluster"], p["likely_species"], p["confidence"]]
                   for p in predictions], [*FEATURES, "Cluster", "Likely species", "Confidence"])
    banner = ("#276749", "PASSED quality check") if status == "passed" else ("#c53030", "BELOW quality threshold")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Penguin clustering report</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#1a202c}}table{{border-collapse:collapse;margin:8px 0 20px}}
td,th{{border:1px solid #cbd5e0;padding:4px 10px;text-align:right}}th{{background:#edf2f7}}
.banner{{color:#fff;background:{banner[0]};padding:8px 14px;display:inline-block;border-radius:4px}}</style></head><body>
<h1>Palmer Penguins clustering report</h1>
<p class="banner">{banner[1]}: ARI {metrics['gmm_ari']}</p>
<p>{metrics['rows']} penguins, {len(FEATURES)} measurements. Clusters were found without the species labels;
the labels are used only to score them.</p>
<h2>Choosing the number of clusters</h2>
<p>BIC picks <b>k={selection['best_k']}</b>. For comparison, the original lab's K-Means elbow picks
k={selection['kmeans_elbow_k']} and the best silhouette picks k={selection['best_silhouette_k']}.</p>
<img alt="Model selection charts" src="data:image/png;base64,{_chart(selection)}" style="max-width:100%">
<h2>Model comparison</h2>{compare}
<h2>Clusters (purity {metrics['purity']:.1%})</h2>{clusters}
<h2>New penguins</h2>{preds}
</body></html>"""
    report_path = Path(out_dir) / "report.html"
    report_path.write_text(html)
    return str(report_path)
