"""
Unit tests for CLI commands and argument handling.
"""

import os
import pytest
import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "src/main.py", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "AIntropy Video Intelligence" in result.stdout
    assert "benchmark-classifier-cascade" in result.stdout
    assert "predict" in result.stdout
    assert "inspect-dataset" in result.stdout


def test_cli_inspect_dataset():
    result = subprocess.run(
        [sys.executable, "src/main.py", "inspect-dataset"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "DATASET INSPECTION & LABEL AUDIT" in result.stdout
    assert "Native Video-Type Routing Labels Present: False" in result.stdout


def test_cli_predict_valid_video(sample_sports_video):
    result = subprocess.run(
        [sys.executable, "src/main.py", "predict", "--video", sample_sports_video],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "predicted_class" in result.stdout
    assert "chunking_strategy" in result.stdout
    assert "stage_resolved" in result.stdout
