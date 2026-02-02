import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    # CI/CD
    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",

    # Configs
    "configs/config.yaml",
    "configs/params.yaml",
    "configs/schema.yaml",

    # Data
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/split/.gitkeep",

    # Artifacts
    "artifacts/model/.gitkeep",
    "artifacts/plots/.gitkeep",
    "artifacts/metrics.json",

    # Source code
    "src/__init__.py",

    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/utils/image_ops.py",

    "src/data/__init__.py",
    "src/data/ingest.py",
    "src/data/validate.py",
    "src/data/split.py",

    "src/model/__init__.py",
    "src/model/build.py",
    "src/model/train.py",
    "src/model/evaluate.py",
    "src/model/export.py",

    "src/serving/__init__.py",
    "src/serving/predictor.py",

    # API
    "app/main.py",

    # Tests
    "tests/__init__.py",
    "tests/test_api.py",

    # Root files
    "requirements.txt",
    "Dockerfile",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if not filedir.exists():
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    if not filepath.exists():
        filepath.touch()
        logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"Already exists: {filepath}")
