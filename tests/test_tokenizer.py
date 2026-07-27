from pathlib import Path
import sentencepiece as spm


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "data/tokenizer/hausa_tokenizer.model"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = spm.SentencePieceProcessor(
    model_file=str(MODEL_PATH)
)


# ============================================================
# DISPLAY TOKENIZATION
# ============================================================

def test_text(text, title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("\nORIGINAL TEXT:")
    print(text)

    # Token pieces
    tokens = tokenizer.encode(
        text,
        out_type=str
    )

    # Token IDs
    token_ids = tokenizer.encode(
        text,
        out_type=int
    )

    # Decode back
    decoded = tokenizer.decode(
        token_ids
    )

    print("\nTOKENS:")
    print(tokens)

    print("\nTOKEN IDS:")
    print(token_ids)

    print("\nNUMBER OF TOKENS:")
    print(len(token_ids))

    print("\nDECODED TEXT:")
    print(decoded)

    print("\nROUND-TRIP MATCH:")
    print(text == decoded)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 70)
    print("DIKKO AI NOMA")
    print("HAUSA SENTENCEPIECE TOKENIZER TEST")
    print("=" * 70)

    print(
        "\nVocabulary size:",
        tokenizer.vocab_size()
    )

    # ========================================================
    # TEST 1: CASUAL HAUSA
    # ========================================================

    casual_text = (
        "Sannu, ya ya kake? "
        "Ina lafiya, na gode. "
        "Yau ina son zuwa kasuwa domin in sayi kayan abinci."
    )

    test_text(
        casual_text,
        "TEST 1: CASUAL HAUSA"
    )

    # ========================================================
    # TEST 2: HAUSA AGRICULTURE
    # ========================================================

    agriculture_text = (
        "Manomi yana shuka masara a farkon damina. "
        "Yana bukatar ya shirya kasa sosai kafin ya dasa iri. "
        "Isasshen ruwa da taki suna taimakawa wajen samun amfanin gona mai kyau."
    )

    test_text(
        agriculture_text,
        "TEST 2: HAUSA AGRICULTURE"
    )

    # ========================================================
    # TEST 3: HAUSA SPECIAL CHARACTERS
    # ========================================================

    special_char_text = (
        "Ƙasa tana da muhimmanci ga noma. "
        "Manomi yana kula da ƙasa domin samun amfanin gona mai kyau. "
        "Ɗan adam yana bukatar abinci domin rayuwa."
    )

    test_text(
        special_char_text,
        "TEST 3: HAUSA SPECIAL CHARACTERS"
    )

    # ========================================================
    # TEST 4: YOUR FINE-TUNING FORMAT
    # ========================================================

    finetune_text = """<|begin_of_sample|>
<|type|>question_answering
<|instruction|>
Menene mafi kyawun lokacin dashen masara a arewacin Najeriya?
<|response|>
Mafi kyawun lokacin dashen masara shi ne farkon damina, lokacin da ruwan sama ya fara sauka akai-akai kuma ƙasa ta sami isasshen danshi.
<|end_of_sample|>"""

    test_text(
        finetune_text,
        "TEST 4: FINE-TUNING FORMAT"
    )

    # ========================================================
    # TEST 5: ANOTHER FINE-TUNING SAMPLE
    # ========================================================

    finetune_text_2 = """<|begin_of_sample|>
<|type|>agricultural_advice
<|instruction|>
Ni manomi ne. Ina son sanin yadda zan kula da gonar masara.
<|response|>
Ya kamata manomi ya tabbatar da cewa gonar tana da isasshen ruwa, ya sarrafa ciyawa, kuma ya kula da amfanin gonar daga kwari da cututtuka.
<|end_of_sample|>"""

    test_text(
        finetune_text_2,
        "TEST 5: AGRICULTURAL FINE-TUNING FORMAT"
    )

    # ========================================================
    # TEST SPECIAL TOKENS INDIVIDUALLY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SPECIAL TOKEN TEST")
    print("=" * 70)

    special_tokens = [
        "<|begin_of_sample|>",
        "<|end_of_sample|>",
        "<|type|>",
        "<|instruction|>",
        "<|response|>",
    ]

    for token in special_tokens:

        token_id = tokenizer.piece_to_id(token)

        encoded = tokenizer.encode(
            token,
            out_type=int
        )

        print(
            f"\n{token}"
        )

        print(
            f"Vocabulary ID: {token_id}"
        )

        print(
            f"Encoded IDs: {encoded}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()