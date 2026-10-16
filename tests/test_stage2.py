"""
Unit tests for Stage 2 Adaptive Frame Sampler and CLIP Classifier.
"""

import os
import pytest
import numpy as np

from src.classifier.stage2 import AdaptiveFrameSampler, CLIPFeatureExtractor, Stage2Classifier, DEFAULT_CLASSES


def test_adaptive_frame_sampler():
    # Short videos (<= 5 min = 300s): 4 frames
    assert AdaptiveFrameSampler.get_target_frame_count(60.0) == 4
    assert AdaptiveFrameSampler.get_target_frame_count(300.0) == 4

    # Medium videos (5-30 min = 300s-1800s): 8 frames
    assert AdaptiveFrameSampler.get_target_frame_count(301.0) == 8
    assert AdaptiveFrameSampler.get_target_frame_count(1800.0) == 8

    # Long videos (> 30 min): 12 frames
    assert AdaptiveFrameSampler.get_target_frame_count(1801.0) == 12
    assert AdaptiveFrameSampler.get_target_frame_count(7200.0) == 12


def test_clip_feature_extractor_embedding(dummy_frames):
    extractor = CLIPFeatureExtractor()
    emb = extractor.extract_frame_embedding(dummy_frames)

    assert isinstance(emb, np.ndarray)
    assert len(emb.shape) == 1
    # 512 for CLIP or 576 for MobileNet fallback
    assert emb.shape[0] in [512, 576]
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)


def test_clip_feature_extractor_empty_frames():
    extractor = CLIPFeatureExtractor()
    emb = extractor.extract_frame_embedding([])

    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] in [512, 576]
    assert np.all(emb == 0.0)


def test_stage2_classifier_fit_predict(test_temp_dir):
    extractor = CLIPFeatureExtractor()
    dim = 576 if extractor._fallback_mode else 512

    X = np.random.randn(20, dim).astype(np.float32)
    # Normalize rows
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = ["sports"] * 10 + ["news_lecture"] * 10

    clf = Stage2Classifier(classes=["sports", "news_lecture"], feature_extractor=extractor)
    clf.fit(X, y)

    assert clf.is_fitted is True

    # Test predict_proba
    probs = clf.predict_proba(X[:2])
    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # Test predict_single
    top_cls, conf, prob_map = clf.predict_single(X[0])
    assert top_cls in ["sports", "news_lecture"]
    assert 0.0 <= conf <= 1.0

    # Test serialization
    save_path = os.path.join(test_temp_dir, "stage2_test.joblib")
    clf.save(save_path)
    assert os.path.exists(save_path)

    loaded = Stage2Classifier.load(save_path)
    assert loaded.is_fitted is True
    top_cls_loaded, conf_loaded, _ = loaded.predict_single(X[0])
    assert top_cls_loaded == top_cls
    assert round(conf_loaded, 4) == round(conf, 4)
