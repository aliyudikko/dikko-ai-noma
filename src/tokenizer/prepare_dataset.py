from pathlib import Path
import numpy as np
import sentencepiece as spm


# ============================================================
# DIKKO AI NOMA DA KIWO
# HAUSA PRETRAINING DATASET PREPARATION
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# Clean Hausa pretraining corpus
INPUT_FILE = Path(
    "data/processed/hausa_pretrain_clean.txt"
)

# Trained SentencePiece tokenizer
TOKENIZER_FILE = Path(
    "data/tokenizer/hausa_tokenizer.model"
)

# Output directory
OUTPUT_DIR = Path(
    "data/pretraining"
)

# Sequence length
SEQ_LEN = 512

# Validation percentage
VAL_RATIO = 0.05

# Reproducibility
SEED = 42


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    if not TOKENIZER_FILE.exists():

        raise FileNotFoundError(
            f"Tokenizer not found:\n"
            f"{TOKENIZER_FILE}"
        )

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_FILE)
    )

    return tokenizer


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Pretraining corpus not found:\n"
            f"{INPUT_FILE}"
        )

    print("\nLoading Hausa corpus...")

    text = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    text = text.strip()

    if not text:

        raise ValueError(
            "The pretraining corpus is empty."
        )

    print(
        f"Characters: {len(text):,}"
    )

    return text


# ============================================================
# TOKENIZE CORPUS
# ============================================================

def tokenize_corpus(
    tokenizer,
    text
):

    print("\nTokenizing corpus...")

    token_ids = tokenizer.encode(
        text,
        out_type=int
    )

    print(
        f"Total tokens: "
        f"{len(token_ids):,}"
    )

    return np.array(
        token_ids,
        dtype=np.uint16
    )


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset(tokens):

    print("\nSplitting dataset...")

    rng = np.random.default_rng(
        SEED
    )

    # Number of complete sequences
    total_sequences = (
        len(tokens)
        // SEQ_LEN
    )

    if total_sequences < 2:

        raise ValueError(
            "Dataset is too small. "
            "You need at least two complete "
            f"{SEQ_LEN}-token sequences."
        )

    # Remove incomplete remainder
    usable_tokens = (
        total_sequences
        * SEQ_LEN
    )

    tokens = tokens[
        :usable_tokens
    ]

    # Reshape into sequences
    sequences = tokens.reshape(
        total_sequences,
        SEQ_LEN
    )

    # Shuffle sequences
    indices = np.arange(
        total_sequences
    )

    rng.shuffle(
        indices
    )

    sequences = sequences[
        indices
    ]

    # Validation size
    val_size = max(
        1,
        int(
            total_sequences
            * VAL_RATIO
        )
    )

    train_size = (
        total_sequences
        - val_size
    )

    train_data = sequences[
        :train_size
    ]

    val_data = sequences[
        train_size:
    ]

    print(
        f"Total sequences: "
        f"{total_sequences:,}"
    )

    print(
        f"Training sequences: "
        f"{len(train_data):,}"
    )

    print(
        f"Validation sequences: "
        f"{len(val_data):,}"
    )

    print(
        f"Training tokens: "
        f"{train_data.size:,}"
    )

    print(
        f"Validation tokens: "
        f"{val_data.size:,}"
    )

    return train_data, val_data


# ============================================================
# SAVE BIN FILES
# ============================================================

def save_dataset(
    train_data,
    val_data
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_file = (
        OUTPUT_DIR
        / "train.bin"
    )

    val_file = (
        OUTPUT_DIR
        / "val.bin"
    )

    print("\nSaving datasets...")

    # Flatten before saving
    train_data.flatten().tofile(
        train_file
    )

    val_data.flatten().tofile(
        val_file
    )

    print(
        f"\nTrain dataset saved:"
    )

    print(
        f"  {train_file}"
    )

    print(
        f"  Size: "
        f"{train_file.stat().st_size / (1024 * 1024):.2f} MB"
    )

    print(
        f"\nValidation dataset saved:"
    )

    print(
        f"  {val_file}"
    )

    print(
        f"  Size: "
        f"{val_file.stat().st_size / (1024 * 1024):.2f} MB"
    )


# ============================================================
# VERIFY DATASET
# ============================================================

def verify_dataset(
    tokenizer
):

    train_file = (
        OUTPUT_DIR
        / "train.bin"
    )

    val_file = (
        OUTPUT_DIR
        / "val.bin"
    )

    print("\n")
    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    train_tokens = np.fromfile(
        train_file,
        dtype=np.uint16
    )

    val_tokens = np.fromfile(
        val_file,
        dtype=np.uint16
    )

    print(
        "\nTrain tokens:",
        len(train_tokens)
    )

    print(
        "Validation tokens:",
        len(val_tokens)
    )

    print(
        "\nMinimum token ID:",
        train_tokens.min()
    )

    print(
        "Maximum token ID:",
        train_tokens.max()
    )

    print(
        "Tokenizer vocabulary:",
        tokenizer.vocab_size()
    )

    # Check token range
    assert (
        train_tokens.max()
        < tokenizer.vocab_size()
    ), "Invalid token ID in train.bin"

    assert (
        val_tokens.max()
        < tokenizer.vocab_size()
    ), "Invalid token ID in val.bin"

    # Show first sequence
    first_sequence = train_tokens[
        :SEQ_LEN
    ]

    decoded = tokenizer.decode(
        first_sequence.tolist()
    )

    print(
        "\nFirst 512-token sequence:"
    )

    print(
        decoded[:1000]
    )

    print(
        "\nDataset verification successful."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DIKKO AI NOMA DA KIWO")
    print("HAUSA PRETRAINING DATASET BUILDER")
    print("=" * 60)

    # Load tokenizer
    tokenizer = load_tokenizer()

    print(
        f"\nTokenizer vocabulary: "
        f"{tokenizer.vocab_size():,}"
    )

    # Load corpus
    text = load_corpus()

    # Tokenize
    tokens = tokenize_corpus(
        tokenizer,
        text
    )

    # Split
    train_data, val_data = split_dataset(
        tokens
    )

    # Save
    save_dataset(
        train_data,
        val_data
    )

    # Verify
    verify_dataset(
        tokenizer
    )

    print("\n")
    print("=" * 60)
    print("PRETRAINING DATASET READY")
    print("=" * 60)

    print(
        "\nFiles:"
    )

    print(
        "  data/pretraining/train.bin"
    )

    print(
        "  data/pretraining/val.bin"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()