"""
Unit tests for VideoProfiler and CoarseSignals extraction.
"""

import os
import pytest
import numpy as np

from src.profiler import VideoProfiler, VideoMetadata, CoarseSignals


def test_profiler_probe_valid_video(sample_sports_video):
    profiler = VideoProfiler()
    meta = profiler.probe(sample_sports_video)

    assert meta.is_valid is True
    assert meta.duration_s > 0.0
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 15.0
    assert meta.frame_count >= 10
    assert meta.aspect_ratio == round(320 / 240, 3)
    assert meta.format_name == "mp4"


def test_profiler_probe_nonexistent_video():
    profiler = VideoProfiler()
    meta = profiler.probe("nonexistent_path/dummy_video_xyz.mp4")

    assert meta.is_valid is False
    assert "File not found" in meta.error_message


def test_profiler_probe_corrupt_file(invalid_video_file):
    profiler = VideoProfiler()
    meta = profiler.probe(invalid_video_file)

    assert meta.is_valid is False
    assert meta.error_message is not None


def test_profiler_extract_frames(sample_sports_video):
    profiler = VideoProfiler()
    frames = profiler.extract_frames(sample_sports_video, num_frames=4)

    assert len(frames) == 4
    for f in frames:
        assert isinstance(f, np.ndarray)
        assert f.shape == (240, 320, 3)


def test_profiler_compute_coarse_signals(dummy_frames):
    profiler = VideoProfiler()
    signals = profiler.compute_coarse_signals(dummy_frames)

    assert isinstance(signals, CoarseSignals)
    assert signals.hist_mean_r > 0.0
    assert signals.hist_mean_g > 0.0
    assert signals.hist_mean_b >= 0.0
    assert signals.frame_diff_mean >= 0.0
    assert 0.0 <= signals.cut_rate_estimate <= 1.0


def test_profiler_empty_frames():
    profiler = VideoProfiler()
    signals = profiler.compute_coarse_signals([])

    assert isinstance(signals, CoarseSignals)
    assert signals.hist_mean_r == 0.0
    assert signals.cut_rate_estimate == 0.0
