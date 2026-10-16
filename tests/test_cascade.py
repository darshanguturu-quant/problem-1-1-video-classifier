"""
Unit tests for Cascaded Hybrid Classifier, routing logic, and fallback mechanisms.
"""

import os
import pytest
import numpy as np

from src.profiler import VideoProfiler, VideoMetadata
from src.classifier.stage1 import Stage1Classifier
from src.classifier.stage2 import Stage2Classifier
from src.classifier.cascade import CascadedClassifier, CascadeResult, FeatureCache


def test_cascade_result_to_dict():
    res = CascadeResult(
        video_id="test_vid_1",
        predicted_class="sports",
        confidence=0.92,
        stage_resolved="stage1",
        escalated_to_stage2=False,
        chunking_strategy="event_and_play",
        latency_ms=12.5,
    )
    d = res.to_dict()
    assert d["video_id"] == "test_vid_1"
    assert d["predicted_class"] == "sports"
    assert d["confidence"] == 0.92
    assert d["stage_resolved"] == "stage1"
    assert d["escalated_to_stage2"] is False
    assert d["chunking_strategy"] == "event_and_play"
    assert d["latency_ms"] == 12.5


def test_cascade_corrupt_video_fallback(invalid_video_file):
    cascade = CascadedClassifier()
    res = cascade.predict_video(invalid_video_file)

    assert res.stage_resolved == "fallback"
    assert res.predicted_class == "general"
    assert res.confidence == 0.0
    assert res.escalated_to_stage2 is False
    assert res.chunking_strategy == "adaptive_scene_and_semantic"
    assert res.error_message is not None


def test_cascade_nonexistent_video_fallback():
    cascade = CascadedClassifier()
    res = cascade.predict_video("nonexistent_path/video_does_not_exist.mp4")

    assert res.stage_resolved == "fallback"
    assert res.predicted_class == "general"
    assert res.confidence == 0.0


def test_cascade_stage1_confidence_gating():
    # Mock Stage 1 with high confidence
    class MockStage1HighConf:
        def predict_single(self, feat):
            return "gameplay", 0.95, {"gameplay": 0.95, "sports": 0.05}

    class MockStage2Unreached:
        def predict_single(self, emb):
            raise AssertionError("Stage 2 should not be called when Stage 1 is confident!")

    cascade = CascadedClassifier(
        stage1=MockStage1HighConf(),
        stage2=MockStage2Unreached(),
        escalation_threshold=0.85
    )

    # Pass mock valid metadata and dummy frames
    meta = VideoMetadata(video_id="vid_high_conf", path="mock.mp4", duration_s=10.0, is_valid=True)
    frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * 4

    res = cascade.predict_video("mock.mp4", meta=meta, frames=frames)

    assert res.stage_resolved == "stage1"
    assert res.predicted_class == "gameplay"
    assert res.confidence == 0.95
    assert res.escalated_to_stage2 is False
    assert res.chunking_strategy == "state_change_and_event"


def test_cascade_escalation_to_stage2():
    # Mock Stage 1 with uncertain prediction (0.60 < 0.85)
    class MockStage1Uncertain:
        def predict_single(self, feat):
            return "sports", 0.60, {"sports": 0.60, "gameplay": 0.40}

    # Mock Stage 2 with confident prediction
    class MockStage2Confident:
        def __init__(self):
            class Extractor:
                def extract_frame_embedding(self, frames):
                    return np.zeros(512, dtype=np.float32)
            self.feature_extractor = Extractor()

        def predict_single(self, emb):
            return "gameplay", 0.91, {"gameplay": 0.91, "sports": 0.09}

    cascade = CascadedClassifier(
        stage1=MockStage1Uncertain(),
        stage2=MockStage2Confident(),
        escalation_threshold=0.85,
        fallback_threshold=0.35
    )

    meta = VideoMetadata(video_id="vid_uncertain", path="mock.mp4", duration_s=10.0, is_valid=True)
    frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * 4

    res = cascade.predict_video("mock.mp4", meta=meta, frames=frames)

    assert res.stage_resolved == "stage2"
    assert res.predicted_class == "gameplay"
    assert res.confidence == 0.91
    assert res.escalated_to_stage2 is True
    assert res.stage1_class == "sports"
    assert res.stage1_confidence == 0.60
    assert res.chunking_strategy == "state_change_and_event"


def test_cascade_stage2_low_confidence_fallback():
    # Both Stage 1 and Stage 2 have low confidence
    class MockStage1Uncertain:
        def predict_single(self, feat):
            return "wildlife", 0.50, {"wildlife": 0.50}

    class MockStage2LowConf:
        def __init__(self):
            class Extractor:
                def extract_frame_embedding(self, frames):
                    return np.zeros(512, dtype=np.float32)
            self.feature_extractor = Extractor()

        def predict_single(self, emb):
            return "wildlife", 0.20, {"wildlife": 0.20}

    cascade = CascadedClassifier(
        stage1=MockStage1Uncertain(),
        stage2=MockStage2LowConf(),
        escalation_threshold=0.85,
        fallback_threshold=0.35,
        fallback_class="general"
    )

    meta = VideoMetadata(video_id="vid_low", path="mock.mp4", duration_s=10.0, is_valid=True)
    frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * 4

    res = cascade.predict_video("mock.mp4", meta=meta, frames=frames)

    assert res.stage_resolved == "fallback"
    assert res.predicted_class == "general"
    assert res.escalated_to_stage2 is True
    assert res.chunking_strategy == "adaptive_scene_and_semantic"


def test_feature_cache_persistence(test_temp_dir):
    cache_path = os.path.join(test_temp_dir, "test_cache.joblib")
    cache = FeatureCache(cache_path)

    cache.set("dummy_vid.mp4", {"val": 12345})
    cache.save()

    assert os.path.exists(cache_path)

    loaded_cache = FeatureCache(cache_path)
    item = loaded_cache.get("dummy_vid.mp4")
    assert item is not None
    assert item.get("val") == 12345
