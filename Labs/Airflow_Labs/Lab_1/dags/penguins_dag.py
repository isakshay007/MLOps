"""
### Penguin clustering pipeline

Clusters the Palmer Penguins by their body measurements, without looking at the species,
then checks how well the clusters match the real species.

`validate_data` → `preprocess` → `select_k` → `train_model` → `evaluate_model` →
`predict_new` + `check_quality` → `publish_report` (ARI ≥ `min_ari`) or `flag_low_quality`

Trigger with config to change `k_min`, `k_max` or `min_ari`. Each run writes its files to
`output/<run_id>/`: `clean.csv`, `model.joblib`, `metrics.json`, `predictions.json`, `report.html`.
"""
import os
from datetime import datetime, timedelta

from airflow.sdk import Param, dag, get_current_context, task

from src import penguins


@dag(
    dag_id="penguin_clustering",
    description="Gaussian Mixture clustering of Palmer Penguins with a data check, model selection, evaluation and an HTML report",
    schedule=None,  # run on demand from the UI or CLI
    start_date=datetime(2026, 11, 1),
    catchup=False,
    default_args={"owner": "akshay", "retries": 1, "retry_delay": timedelta(minutes=1)},
    params={
        "k_min": Param(2, type="integer", minimum=2, description="Smallest number of clusters to try"),
        "k_max": Param(10, type="integer", minimum=3, maximum=15, description="Largest number of clusters to try"),
        "min_ari": Param(0.8, type="number", minimum=0, maximum=1,
                         description="Minimum Adjusted Rand Index vs the real species to pass"),
    },
    tags=["lab1", "clustering", "penguins"],
    doc_md=__doc__,
)
def penguin_clustering():

    @task(retries=0)  # bad data won't fix itself on retry
    def validate_data():
        return penguins.validate_data()

    @task
    def preprocess(validation):
        print(f"Validated {validation['rows']} rows; missing per feature: {validation['missing']}")
        return penguins.preprocess(penguins.run_dir(get_current_context()["run_id"]))

    @task
    def select_k(prep):
        params = get_current_context()["params"]
        selection = penguins.select_k(prep["clean_path"], params["k_min"], params["k_max"])
        print(f"BIC picks k={selection['best_k']}; K-Means elbow would pick k={selection['kmeans_elbow_k']}, "
              f"silhouette k={selection['best_silhouette_k']}")
        return selection

    @task
    def train_model(prep, selection):
        return penguins.train_model(prep["clean_path"], selection["best_k"], os.path.dirname(prep["clean_path"]))

    @task
    def evaluate_model(model_path, prep):
        metrics = penguins.evaluate_model(model_path, prep["clean_path"], os.path.dirname(model_path))
        print(f"ARI {metrics['gmm_ari']} (K-Means baseline {metrics['kmeans_ari']}), purity {metrics['purity']}")
        return metrics

    @task
    def predict_new(model_path, metrics):
        predictions = penguins.predict_new(model_path, metrics["cluster_species"], os.path.dirname(model_path))
        for p in predictions:
            print(f"{p['bill_length_mm']} mm bill, {p['body_mass_g']} g -> {p['likely_species']} ({p['confidence']:.0%})")
        return predictions

    @task.branch
    def check_quality(metrics):
        min_ari = get_current_context()["params"]["min_ari"]
        return "publish_report" if metrics["gmm_ari"] >= min_ari else "flag_low_quality"

    @task
    def publish_report(selection, metrics, predictions, prep):
        path = penguins.build_report(selection, metrics, predictions, os.path.dirname(prep["clean_path"]), "passed")
        print(f"Report: {path}")
        return path

    @task(retries=0)
    def flag_low_quality(selection, metrics, predictions, prep):
        path = penguins.build_report(selection, metrics, predictions, os.path.dirname(prep["clean_path"]), "below")
        raise ValueError(f"ARI {metrics['gmm_ari']} is below min_ari "
                         f"{get_current_context()['params']['min_ari']}; see {path}")

    prep = preprocess(validate_data())
    selection = select_k(prep)
    model_path = train_model(prep, selection)
    metrics = evaluate_model(model_path, prep)
    predictions = predict_new(model_path, metrics)
    branch = check_quality(metrics)
    branch >> [publish_report(selection, metrics, predictions, prep),
               flag_low_quality(selection, metrics, predictions, prep)]


penguin_clustering()
