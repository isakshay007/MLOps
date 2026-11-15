"""DAG integrity tests: the DAG file imports cleanly and is wired as intended. Needs Airflow."""
import pytest

pytest.importorskip("airflow")

from airflow.dag_processing.dagbag import DagBag  # noqa: E402

from conftest import DAGS_DIR  # noqa: E402

EXPECTED_DOWNSTREAM = {
    "validate_data": {"preprocess"},
    "preprocess": {"select_k", "train_model", "evaluate_model", "publish_report", "flag_low_quality"},
    "select_k": {"train_model", "publish_report", "flag_low_quality"},
    "train_model": {"evaluate_model", "predict_new"},
    "evaluate_model": {"predict_new", "check_quality", "publish_report", "flag_low_quality"},
    "predict_new": {"publish_report", "flag_low_quality"},
    "check_quality": {"publish_report", "flag_low_quality"},
    "publish_report": set(),
    "flag_low_quality": set(),
}


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=str(DAGS_DIR))


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_dag_loaded(dagbag):
    assert list(dagbag.dags) == ["penguin_clustering"]


def test_task_wiring(dagbag):
    dag = dagbag.dags["penguin_clustering"]
    assert {t.task_id: set(t.downstream_task_ids) for t in dag.tasks} == EXPECTED_DOWNSTREAM


def test_dag_settings(dagbag):
    dag = dagbag.dags["penguin_clustering"]
    assert dag.catchup is False
    assert set(dag.params) == {"k_min", "k_max", "min_ari"}
    assert dag.task_dict["validate_data"].retries == 0
    assert dag.task_dict["train_model"].retries == 1
