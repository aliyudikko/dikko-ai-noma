from pathlib import Path
import sentencepiece as spm


# ============================================================
# DIKKO AI NOMA DA KIWO
# HAUSA SENTENCEPIECE TOKENIZER
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# Clean Hausa pretraining corpus
INPUT_FILE = Path(
    "data/processed/hausa_pretrain_clean.txt"
)

# Tokenizer output directory
OUTPUT_DIR = Path(
    "data/tokenizer"
)

# Output filename prefix
MODEL_PREFIX = OUTPUT_DIR / "hausa_tokenizer"

# Vocabulary size
VOCAB_SIZE = 8000


# ============================================================
# SPECIAL TOKENS
# ============================================================

SPECIAL_TOKENS = [
    "<|begin_of_sample|>",
    "<|end_of_sample|>",
    "<|type|>",
    "<|instruction|>",
    "<|response|>",
]


# ============================================================
# TRAIN TOKENIZER
# ============================================================

def train_tokenizer():

    print("=" * 60)
    print("DIKKO AI NOMA DA KIWO")
    print("HAUSA TOKENIZER TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nPretraining corpus not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Expected file:\n"
            f"data/processed/hausa_pretrain_clean.txt"
        )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FILE INFORMATION
    # --------------------------------------------------------

    file_size_mb = (
        INPUT_FILE.stat().st_size
        / (1024 * 1024)
    )

    print(f"\nInput corpus:")
    print(f"  {INPUT_FILE}")

    print(f"\nCorpus size:")
    print(f"  {file_size_mb:.2f} MB")

    print(f"\nVocabulary size:")
    print(f"  {VOCAB_SIZE}")

    print(f"\nTokenizer:")
    print("  SentencePiece BPE")

    print("\nSpecial tokens:")

    for token in SPECIAL_TOKENS:
        print(f"  {token}")

    print("\nTraining tokenizer...\n")

    # ========================================================
    # SENTENCEPIECE TRAINING
    # ========================================================

    spm.SentencePieceTrainer.train(

        # ----------------------------------------------------
        # INPUT CORPUS
        # ----------------------------------------------------

        input=str(INPUT_FILE),

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        model_prefix=str(MODEL_PREFIX),

        # ----------------------------------------------------
        # TOKENIZER ALGORITHM
        # ----------------------------------------------------

        model_type="bpe",

        # ----------------------------------------------------
        # VOCABULARY
        # ----------------------------------------------------

        vocab_size=VOCAB_SIZE,

        # Keep all characters appearing in the corpus
        character_coverage=1.0,

        # ----------------------------------------------------
        # BASE SPECIAL TOKENS
        # ----------------------------------------------------

        # Unknown token
        unk_id=0,
        unk_piece="<unk>",

        # Beginning of sequence
        bos_id=1,
        bos_piece="<bos>",

        # End of sequence
        eos_id=2,
        eos_piece="<eos>",

        # Padding
        pad_id=3,
        pad_piece="<pad>",

        # ----------------------------------------------------
        # DIKKO AI CUSTOM CONTROL TOKENS
        # ----------------------------------------------------

        user_defined_symbols=SPECIAL_TOKENS,

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        normalization_rule_name="nmt_nfkc",

        # ----------------------------------------------------
        # WHITESPACE HANDLING
        # ----------------------------------------------------

        # Prevent automatic whitespace prefix
        add_dummy_prefix=False,

        # Preserve repeated whitespace
        remove_extra_whitespaces=False,

        # ----------------------------------------------------
        # TRAINING SETTINGS
        # ----------------------------------------------------

        max_sentence_length=4192,

        shuffle_input_sentence=True,

        # Number of CPU threads
        num_threads=8,

        # Reduce unnecessary logs
        minloglevel=1,
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    MODEL_FILE = Path(
        f"{MODEL_PREFIX}.model"
    )

    VOCAB_FILE = Path(
        f"{MODEL_PREFIX}.vocab"
    )

    # ========================================================
    # VERIFY TOKENIZER
    # ========================================================

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(MODEL_FILE)
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 60)

    print("\nModel saved to:")
    print(f"  {MODEL_FILE}")

    print("\nVocabulary saved to:")
    print(f"  {VOCAB_FILE}")

    print("\nActual vocabulary size:")
    print(
        f"  {tokenizer.vocab_size()}"
    )

    print("\nSpecial token IDs:")

    for token in SPECIAL_TOKENS:

        token_id = tokenizer.piece_to_id(
            token
        )

        print(
            f"  {token:25} -> {token_id}"
        )

    print("\nBase token IDs:")

    print(
        f"  <unk> -> "
        f"{tokenizer.unk_id()}"
    )

    print(
        f"  <bos> -> "
        f"{tokenizer.bos_id()}"
    )

    print(
        f"  <eos> -> "
        f"{tokenizer.eos_id()}"
    )

    print(
        f"  <pad> -> "
        f"{tokenizer.pad_id()}"
    )

    print("\nTokenizer is ready.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_tokenizer()