"""
Unit tests for Stage 1 Metadata & Coarse Feature Classifier.
"""

import os
import pytest
import numpy as np

from src.profiler import VideoMetadata, CoarseSignals
from src.classifier.stage1 import Stage1Features, Stage1Classifier, DEFAULT_CLASSES


def test_stage1_features_transform():
    meta = VideoMetadata(
        video_id="test_vid",
        path="dummy.mp4",
        duration_s=120.0,
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=3600,
        bitrate_kbps=4500.0,
        aspect_ratio=1.778,
        has_audio=True,
        audio_channels=2,
        file_size_bytes=67500000,
        is_valid=True
    )
    signals = CoarseSignals(
        hist_mean_r=128.0,
        hist_mean_g=110.0,
        hist_mean_b=90.0,
        hist_std_r=45.0,
        hist_std_g=40.0,
        hist_std_b=35.0,
        frame_diff_mean=15.0,
        frame_diff_std=8.0,
        cut_rate_estimate=0.1
    )

    vec = Stage1Features.transform(meta, signals)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (17,)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_stage1_classifier_fit_predict(test_temp_dir):
    # Create simple synthetic feature data
    X = np.random.randn(20, 17).astype(np.float32)
    y = ["gameplay"] * 10 + ["podcast"] * 10

    clf = Stage1Classifier(classes=["gameplay", "podcast"])
    clf.fit(X, y)

    assert clf.is_fitted is True

    # Test predict_proba
    probs = clf.predict_proba(X[:2])
    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # Test predict_single
    top_cls, conf, prob_map = clf.predict_single(X[0])
    assert top_cls in ["gameplay", "podcast"]
    assert 0.0 <= conf <= 1.0
    assert "gameplay" in prob_map and "podcast" in prob_map

    # Test serialization
    save_path = os.path.join(test_temp_dir, "stage1_test.joblib")
    clf.save(save_path)
    assert os.path.exists(save_path)

    loaded = Stage1Classifier.load(save_path)
    assert loaded.is_fitted is True
    top_cls_loaded, conf_loaded, _ = loaded.predict_single(X[0])
    assert top_cls_loaded == top_cls
    assert round(conf_loaded, 4) == round(conf, 4)


def test_stage1_classifier_unfitted():
    clf = Stage1Classifier()
    assert clf.is_fitted is False
    vec = np.zeros(17, dtype=np.float32)
    top_cls, conf, prob_map = clf.predict_single(vec)
    assert 0.0 <= conf <= 1.0
    assert len(prob_map) == len(DEFAULT_CLASSES)
