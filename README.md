# Task 1 — Video Type Classifier

Cascaded Hybrid Video Type Classifier combining metadata-driven fast gating (<15ms) with conditional CLIP ViT-B/32 semantic refinement for uncertain predictions.

## Directory Structure
- `src/`: Core implementation and CLI entrypoint.
- `tests/`: Automated unit and integration test suite.
- `specs/`: Engineering specifications (`PRD.md`, `ARCHITECTURE.md`, `PROGRESS.md`, `RULES.md`).
- `docs/`: Technical writeups and task documentation.
- `run.yaml`: Evaluator configuration and non-interactive commands.
- `requirements.txt`: Environment dependencies.

## Execution & Verification Commands
```bash
# Setup dependencies
pip install -r requirements.txt

# Run Task Execution Commands
python src/main.py --help
python src/main.py benchmark-classifier-cascade

# Run Test Suite
pytest tests/ -v
```

## Grader Bot Integration (`run.yaml`)
This folder is fully compatible with the [AIntropy Grader Bot](https://github.com/apps/aintropy-grader-bot).
