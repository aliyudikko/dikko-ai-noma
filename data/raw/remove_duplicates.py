import random
from pathlib import Path


# ============================================================
# DIKKO AI NOMA
# HAUSA FINE-TUNING DATASET SPLITTER
# ============================================================

INPUT_FILE = Path(
    "hausa_finetune_deduplicated.txt"
)

TRAIN_FILE = Path(
    "data/processed/hausa_finetune_train.txt"
)

VAL_FILE = Path(
    "data/processed/hausa_finetune_val.txt"
)

# 90% Train / 10% Validation
VAL_RATIO = 0.10

# For reproducibility
SEED = 42


# ============================================================
# READ SAMPLES
# ============================================================

def read_samples(filepath):
    """
    Karanta samples daga dataset.

    Ana sa ran format:

    <|begin_of_sample|>
    <|instruction|>
    ...
    <|response|>
    ...
    <|end_of_sample|>
    """

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    # Raba samples ta BEGIN token
    raw_samples = text.split(
        "<|begin_of_sample|>"
    )

    samples = []

    for raw in raw_samples:

        raw = raw.strip()

        if not raw:
            continue

        # Sake gina sample
        sample = (
            "<|begin_of_sample|>\n"
            + raw
        )

        # Tabbatar END token yana nan
        if "<|end_of_sample|>" not in sample:
            sample += "\n<|end_of_sample|>"

        samples.append(
            sample.strip()
        )

    return samples


# ============================================================
# REMOVE EMPTY LINES INSIDE SAMPLE
# ============================================================

def remove_empty_lines(sample):
    """
    Cire dukkan blank/empty lines daga sample.

    Misali:

    <|begin_of_sample|>

    <|instruction|>

    Tambaya

    <|response|>

    Amsa

    <|end_of_sample|>

    Zai koma:

    <|begin_of_sample|>
    <|instruction|>
    Tambaya
    <|response|>
    Amsa
    <|end_of_sample|>
    """

    lines = sample.splitlines()

    # Cire whitespace daga kowanne line
    lines = [
        line.strip()
        for line in lines
    ]

    # Cire empty lines
    lines = [
        line
        for line in lines
        if line
    ]

    return "\n".join(lines)


# ============================================================
# VALIDATE SAMPLE
# ============================================================

def is_valid_sample(sample):

    required_tokens = [
        "<|begin_of_sample|>",
        "<|instruction|>",
        "<|response|>",
        "<|end_of_sample|>"
    ]

    for token in required_tokens:

        if token not in sample:
            return False

    return True


# ============================================================
# WRITE DATASET
# ============================================================

def write_samples(
    samples,
    filepath
):

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        for sample in samples:

            # Cire empty lines
            clean_sample = (
                remove_empty_lines(
                    sample
                )
            )

            # Rubuta sample
            f.write(
                clean_sample
            )

            # LINE DAYA KAWAI BAYAN SAMPLE
            f.write(
                "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "DIKKO AI NOMA"
    )

    print(
        "HAUSA FINE-TUNING DATASET SPLITTER"
    )

    print("=" * 70)


    # ========================================================
    # CHECK INPUT
    # ========================================================

    if not INPUT_FILE.exists():

        print(
            "\nERROR: Ba a samu input file ba:"
        )

        print(
            INPUT_FILE
        )

        return


    # ========================================================
    # READ
    # ========================================================

    print(
        "\n[1] Ana karanta dataset..."
    )

    samples = read_samples(
        INPUT_FILE
    )

    print(
        f"Samples da aka samu: "
        f"{len(samples):,}"
    )


    if len(samples) < 2:

        print(
            "\nERROR: Dataset ya yi ƙanƙanta."
        )

        return


    # ========================================================
    # REMOVE EMPTY LINES
    # ========================================================

    print(
        "\n[2] Ana cire empty lines..."
    )

    cleaned_samples = []

    for sample in samples:

        sample = (
            remove_empty_lines(
                sample
            )
        )

        if is_valid_sample(
            sample
        ):

            cleaned_samples.append(
                sample
            )


    print(
        f"Samples masu inganci: "
        f"{len(cleaned_samples):,}"
    )


    # ========================================================
    # SHUFFLE
    # ========================================================

    print(
        "\n[3] Ana shuffle dataset..."
    )

    random.seed(
        SEED
    )

    random.shuffle(
        cleaned_samples
    )


    # ========================================================
    # CALCULATE SPLIT
    # ========================================================

    val_size = max(
        1,
        int(
            len(cleaned_samples)
            * VAL_RATIO
        )
    )

    train_size = (
        len(cleaned_samples)
        - val_size
    )


    # ========================================================
    # SPLIT
    # ========================================================

    train_samples = (
        cleaned_samples[
            :train_size
        ]
    )

    val_samples = (
        cleaned_samples[
            train_size:
        ]
    )


    # ========================================================
    # SAVE TRAIN
    # ========================================================

    print(
        "\n[4] Ana ajiye TRAIN dataset..."
    )

    write_samples(
        train_samples,
        TRAIN_FILE
    )


    # ========================================================
    # SAVE VALIDATION
    # ========================================================

    print(
        "[5] Ana ajiye VALIDATION dataset..."
    )

    write_samples(
        val_samples,
        VAL_FILE
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SPLIT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Total samples: "
        f"{len(cleaned_samples):,}"
    )

    print(
        f"Train samples: "
        f"{len(train_samples):,}"
    )

    print(
        f"Validation samples: "
        f"{len(val_samples):,}"
    )

    print(
        f"\nTrain ratio: "
        f"{len(train_samples) / len(cleaned_samples) * 100:.1f}%"
    )

    print(
        f"Validation ratio: "
        f"{len(val_samples) / len(cleaned_samples) * 100:.1f}%"
    )

    print(
        "\nFiles created:"
    )

    print(
        f"✓ {TRAIN_FILE}"
    )

    print(
        f"✓ {VAL_FILE}"
    )

    print(
        "\nEmpty lines:"
    )

    print(
        "✓ Removed"
    )

    print(
        "\nFinal sample format:"
    )

    print(
        "<|begin_of_sample|>"
    )

    print(
        "<|instruction|>"
    )

    print(
        "Tambaya..."
    )

    print(
        "<|response|>"
    )

    print(
        "Amsa..."
    )

    print(
        "<|end_of_sample|>"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()