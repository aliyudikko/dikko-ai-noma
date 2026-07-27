from pathlib import Path

# Project root
ROOT = Path(".")

# Directories to create
directories = [
    "data/raw",
    "data/processed",
    "data/tokenized",

    "src/preprocessing",
    "src/tokenizer",
    "src/model",
    "src/training",
    "src/inference",
    "src/rag",

    "checkpoints",
    "scripts",
]

# Python files to create
python_files = [
    "src/__init__.py",

    "src/preprocessing/__init__.py",
    "src/preprocessing/clean_text.py",
    "src/preprocessing/deduplicate.py",
    "src/preprocessing/filter_quality.py",

    "src/tokenizer/__init__.py",
    "src/tokenizer/train_tokenizer.py",
    "src/tokenizer/tokenize_dataset.py",

    "src/model/__init__.py",
    "src/model/config.py",
    "src/model/transformer.py",
    "src/model/attention.py",
    "src/model/embedding.py",

    "src/training/__init__.py",
    "src/training/train.py",
    "src/training/dataset.py",
    "src/training/checkpoint.py",

    "src/inference/__init__.py",
    "src/inference/generate.py",

    "src/rag/__init__.py",
    "src/rag/retriever.py",
    "src/rag/vector_store.py",

    "scripts/preprocess_data.py",
    "scripts/train_tokenizer.py",
    "scripts/tokenize_data.py",
    "scripts/train_model.py",
    "scripts/run_inference.py",
]

# Create directories
for directory in directories:
    path = ROOT / directory
    path.mkdir(parents=True, exist_ok=True)
    print(f"[DIR]  Created: {path}")

# Create Python files
for file in python_files:
    path = ROOT / file
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text(
            f'"""Module: {path.stem}."""\n',
            encoding="utf-8"
        )
        print(f"[FILE] Created: {path}")
    else:
        print(f"[SKIP] Already exists: {path}")

# Create empty .gitkeep files for empty directories
for directory in [
    "data/raw",
    "data/processed",
    "data/tokenized",
    "checkpoints",
]:
    gitkeep = ROOT / directory / ".gitkeep"

    if not gitkeep.exists():
        gitkeep.touch()

print("\nProject structure created successfully!")