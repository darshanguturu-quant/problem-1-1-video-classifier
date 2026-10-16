# Problem 1.1: Video Classifier

## Submission instructions

1. Install the **AIntropy Grader Bot** GitHub App (https://github.com/apps/aintropy-grader-bot) and grant it access to your fork, if you haven't already — we can't evaluate your submission without this.
2. Put your code under `src/`.
3. Put your write-up under `docs/` — markdown, slides, a notebook, or another format of your choosing are all fine, as long as it's committed in this repo.
4. Fill in `run.yaml` with the **exact** commands needed to set up and run your code. We run these commands exactly as written to evaluate your submission — this is the one part of the submission we read literally, not loosely: if `run.yaml` doesn't work, we can't see your code run.
5. Push to your own fork's default branch any time during your test's time limit. No pull request is needed or expected.

---

## Solution Overview: Cascaded Hybrid Video Type Classifier

An autonomous, two-stage cascaded classification system that infers the dominant video category upon ingestion to dynamically parameterize downstream chunking and transcoding pipelines.

### Architecture Highlights
- **Stage 1 (Lightweight Metadata Gate)**: Analyzes container metadata (duration, resolution, bitrate ratio, audio streams) and coarse keyframe color statistics. Runs in < 15 ms CPU latency. If confidence $\ge 0.85$, routes immediately.
- **Stage 2 (High-Capacity Semantic Refinement)**: Triggers adaptive frame sampling (4 frames for short $\le 5$ min, 8 frames for medium 5-30 min, 12 frames for long $> 30$ min) and visual feature embeddings (CLIP ViT-B/32 or compact MobileNetV3) for uncertain videos.
- **Taxonomy**: 11 production categories (`gameplay`, `egocentric`, `podcast`, `wildlife`, `sports`, `news_lecture`, `meetings`, `surveillance`, `concert`, `film_animation`, `general`).
- **Hardware Profile**: 100% CPU compatible, memory footprint < 600 MB RAM.

### Repository Layout
- `src/classifier/`: Stage 1 gating, Stage 2 visual refinement, cascaded coordinator, and baseline models.
- `src/profiler.py`: Sub-second container metadata and stream inspector.
- `docs/TASK1_VIDEO_CLASSIFIER.md`: Detailed architectural and benchmarking technical report.
- `specs/`: System requirements, formulas, and evaluation protocols.
- `tests/`: Automated unit and integration test suite (26 passing tests).
- `run.yaml`: Evaluator entrypoint executed non-interactively by AIntropy Grader Bot.

### Commands & Execution
```bash
# 1. Setup
pip install -r requirements.txt

# 2. Run Solution / Benchmark
python src/main.py --help
python src/main.py benchmark-classifier-cascade

# 3. Run Automated Tests
pytest tests/ -v
```
