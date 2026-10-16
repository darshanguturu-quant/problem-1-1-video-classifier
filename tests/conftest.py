"""
Pytest fixtures and configuration for Task 1 Video Classifier test suite.
"""

import os
import sys
import shutil
import tempfile
import pytest
import numpy as np
import cv2
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import generate_synthetic_video


@pytest.fixture(scope="session")
def test_temp_dir():
    """Create a temporary directory for test artifacts and clean up after session."""
    temp_dir = tempfile.mkdtemp(prefix="aintropy_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_sports_video(test_temp_dir):
    """Generates a small valid sports synthetic video."""
    video_path = os.path.join(test_temp_dir, "test_sports.mp4")
    generate_synthetic_video(video_path, video_type="sports", duration_s=1.0, fps=15)
    return video_path


@pytest.fixture(scope="session")
def sample_lecture_video(test_temp_dir):
    """Generates a small valid news/lecture synthetic video."""
    video_path = os.path.join(test_temp_dir, "test_lecture.mp4")
    generate_synthetic_video(video_path, video_type="news_lecture", duration_s=1.0, fps=15)
    return video_path


@pytest.fixture(scope="session")
def invalid_video_file(test_temp_dir):
    """Creates a corrupted non-video file."""
    corrupt_path = os.path.join(test_temp_dir, "corrupt_video.mp4")
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_MP4_HEADER_RANDOM_BYTES_1234567890")
    return corrupt_path


@pytest.fixture(scope="session")
def dummy_frames():
    """Generates 4 dummy BGR frames for fast offline testing."""
    frames = []
    for i in range(4):
        f = np.zeros((240, 320, 3), dtype=np.uint8)
        f[:, :] = (i * 40, 100, 150)
        frames.append(f)
    return frames
