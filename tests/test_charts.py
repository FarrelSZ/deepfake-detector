import pytest
from ui_utils import charts

METRICS = {"accuracy": 0.93, "precision": 0.92, "recall": 0.95, "f1": 0.93, "auc": 0.98}


def test_metrics_chart_has_bars_and_target_ticks():
    spec = charts.metrics_vs_target_chart(METRICS).to_dict()
    assert [layer["mark"]["type"] for layer in spec["layer"]] == ["bar", "tick"]


def test_metrics_chart_tolerates_missing_metrics():
    assert charts.metrics_vs_target_chart({"accuracy": 0.9}).to_dict()


def test_comparison_chart_groups_models():
    comparison = [{"name": "A", **METRICS}, {"name": "B", **METRICS}]
    spec = charts.model_comparison_chart(comparison).to_dict()
    assert spec["encoding"]["xOffset"]["field"] == "name"


HISTORY = [
    {"model": "A", "epoch": 0, "loss": 0.7, "val_loss": 0.8, "auc": 0.6, "val_auc": 0.55},
    {"model": "A", "epoch": 1, "loss": 0.5, "val_loss": 0.6, "auc": 0.8, "val_auc": 0.7},
]


def test_available_curve_metrics_requires_train_and_val_pair():
    history = [{**HISTORY[0], "recall": 0.5}]
    assert charts.available_curve_metrics(history) == ["loss", "auc"]


@pytest.mark.parametrize("metric", ["loss", "auc"])
def test_training_curves_chart_builds(metric):
    assert charts.training_curves_chart(HISTORY, metric).to_dict()


def test_history_charts_build():
    records = [{"p_fake": 0.2, "label": "real"}, {"p_fake": 0.9, "label": "fake"}]
    assert charts.prediction_distribution_chart(records).to_dict()
    assert charts.label_counts_chart({"real": 1, "fake": 1}).to_dict()
