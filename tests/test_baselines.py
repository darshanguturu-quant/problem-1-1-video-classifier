"""
Unit tests for baseline classifiers and benchmark runner.
"""

import os
import pytest
import numpy as np

from src.classifier.baselines import MobileNetV3Classifier, BenchmarkRunner, BaselineMetricReport
from src.classifier.cascade import CascadeResult


def test_mobilenet_classifier_fit_predict(test_temp_dir):
    clf = MobileNetV3Classifier(classes=["gameplay", "sports"])
    X = np.random.randn(10, 576).astype(np.float32)
    y = ["gameplay"] * 5 + ["sports"] * 5

    clf.fit(X, y)
    assert clf.is_fitted is True

    top_cls, conf, prob_map = clf.predict_single(X[0])
    assert top_cls in ["gameplay", "sports"]
    assert 0.0 <= conf <= 1.0

    save_path = os.path.join(test_temp_dir, "mobilenet_test.joblib")
    clf.save(save_path)
    assert os.path.exists(save_path)

    loaded = MobileNetV3Classifier.load(save_path)
    assert loaded.is_fitted is True


def test_benchmark_runner_evaluate_model():
    runner = BenchmarkRunner()

    paths = ["vid1.mp4", "vid2.mp4", "vid3.mp4", "vid4.mp4"]
    ground_truth = ["sports", "sports", "podcast", "podcast"]

    # Predict function returning predictions
    def mock_predict(path):
        if "1" in path or "2" in path:
            return ("sports", 0.90)
        return ("podcast", 0.85)

    report = runner.evaluate_model(
        "MockModel",
        mock_predict,
        paths,
        ground_truth,
        model_size_mb=10.0,
    )

    assert isinstance(report, BaselineMetricReport)
    assert report.macro_f1 == 1.0
    assert report.p50_latency_ms >= 0.0
    assert report.sample_count == 4
    assert report.escalation_rate_pct == 0.0
    assert "sports" in report.per_class_f1
    assert "podcast" in report.per_class_f1


def test_format_markdown_table():
    r1 = BaselineMetricReport(
        model_name="Model A",
        macro_f1=0.88,
        per_class_precision={"cls1": 0.9},
        per_class_recall={"cls1": 0.85},
        per_class_f1={"cls1": 0.87},
        p50_latency_ms=15.2,
        p95_latency_ms=25.4,
        mean_latency_ms=16.0,
        peak_memory_mb=1.2,
        model_size_mb=0.1,
        escalation_rate_pct=0.0,
        sample_count=10,
    )
    table = BenchmarkRunner.format_markdown_table({"model_a": r1})
    assert "| Model Architecture | Macro-F1 |" in table
    assert "Model A" in table
    assert "15.2 ms" in table
