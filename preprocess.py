import re
import hashlib
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("datasets/")
OUTPUT_DIR = Path("datasets/processed")
OUTPUT_FILE = OUTPUT_DIR / "hausa_pretrain_clean.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. NORMALIZE UNICODE AND WHITESPACE
# ============================================================

def normalize_text(text):
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove zero-width and invisible characters
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    # Normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 2. REMOVE URLS
# ============================================================

def remove_urls(text):
    patterns = [
        r"https?://\S+",
        r"http://\S+",
        r"www\.\S+",
        r"\b\S+\.(com|org|net|edu|gov|ng|uk|io)\S*\b",
    ]

    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return text


# ============================================================
# 3. REMOVE EMAIL ADDRESSES
# ============================================================

def remove_emails(text):
    return re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )


# ============================================================
# 4. REMOVE DOI REFERENCES
# ============================================================

def remove_doi(text):
    return re.sub(
        r"\bdoi\s*:\s*\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )


# ============================================================
# 5. REMOVE ACADEMIC CITATIONS
#
# Examples:
# (Author, 2020)
# (Author et al., 2021)
# [1]
# [12, 15]
# [1-5]
# ============================================================

def remove_citations(text):

    # [1]
    text = re.sub(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]", " ", text)

    # (Author, 2020)
    text = re.sub(
        r"\([A-Z][A-Za-zÀ-ÿ'-]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\)",
        " ",
        text,
        flags=re.IGNORECASE
    )

    return text


# ============================================================
# 6. REMOVE IEEE-STYLE REFERENCES
#
# Removes lines such as:
#
# References
# REFERENCES
# Bibliography
# [1] Author...
# [2] Author...
# ============================================================

REFERENCE_HEADERS = [
    "references",
    "reference",
    "bibliography",
    "works cited",
    "sources",
    "littafin manazarta",
    "manazarta",
    "tushen bayani",
]


def remove_reference_sections(text):

    lines = text.splitlines()

    cleaned_lines = []

    inside_references = False

    for line in lines:

        stripped = line.strip()

        # Detect reference section
        if stripped.lower().rstrip(":") in REFERENCE_HEADERS:
            inside_references = True
            continue

        if inside_references:

            # Stop if a clearly new section starts
            if (
                stripped
                and len(stripped) < 100
                and stripped.endswith(":")
            ):
                inside_references = False
            else:
                # Skip reference lines
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ============================================================
# 7. REMOVE INDIVIDUAL REFERENCE LINES
# ============================================================

def remove_reference_lines(text):

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        stripped = line.strip()

        # IEEE numbering
        if re.match(r"^\[\d+\]\s+", stripped):
            continue

        # Numbered bibliography
        if re.match(r"^\d+\.\s+.*\b(19|20)\d{2}\b", stripped):
            continue

        # DOI
        if re.search(r"\bdoi\b", stripped, re.IGNORECASE):
            continue

        # URL-only lines
        if re.match(r"^(https?://|www\.)", stripped, re.IGNORECASE):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ============================================================
# 8. REMOVE COMMON ACADEMIC METADATA
# ============================================================

def remove_academic_metadata(text):

    patterns = [
        r"(?im)^title\s*:.*$",
        r"(?im)^author[s]?\s*:.*$",
        r"(?im)^abstract\s*:.*$",
        r"(?im)^keywords?\s*:.*$",
        r"(?im)^affiliation\s*:.*$",
        r"(?im)^email\s*:.*$",
        r"(?im)^corresponding author\s*:.*$",
        r"(?im)^journal\s*:.*$",
        r"(?im)^volume\s*:.*$",
        r"(?im)^issue\s*:.*$",
        r"(?im)^issn\s*:.*$",
        r"(?im)^isbn\s*:.*$",
        r"(?im)^published\s*:.*$",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text)

    return text


# ============================================================
# 9. REMOVE MARKDOWN / HTML NOISE
# ============================================================

def remove_markup(text):

    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Markdown links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Markdown headings
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)

    # Markdown emphasis
    text = re.sub(r"[*_]{1,3}", "", text)

    return text


# ============================================================
# 10. REMOVE VERY NOISY LINES
# ============================================================

def remove_noise_lines(text):

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            cleaned.append("")
            continue

        # Ignore extremely short lines
        if len(stripped) < 10:
            continue

        # Ignore lines that are mostly symbols/numbers
        alphanumeric = sum(c.isalnum() for c in stripped)

        if alphanumeric < 5:
            continue

        # Ignore common page numbers
        if re.fullmatch(r"(page|shafi)?\s*\d+", stripped, re.IGNORECASE):
            continue

        cleaned.append(stripped)

    return "\n".join(cleaned)


# ============================================================
# 11. CLEAN INDIVIDUAL DOCUMENT
# ============================================================

def clean_document(text):

    text = normalize_text(text)

    text = remove_urls(text)

    text = remove_emails(text)

    text = remove_doi(text)

    text = remove_citations(text)

    text = remove_reference_sections(text)

    text = remove_reference_lines(text)

    text = remove_academic_metadata(text)

    text = remove_markup(text)

    text = remove_noise_lines(text)

    text = normalize_text(text)

    return text


# ============================================================
# 12. REMOVE DUPLICATE DOCUMENTS
# ============================================================

def document_hash(text):
    """
    Exact duplicate detection.
    """
    normalized = re.sub(r"\s+", " ", text.lower()).strip()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# 13. REMOVE DUPLICATE PARAGRAPHS
# ============================================================

def remove_duplicate_paragraphs(text, seen_paragraphs):

    paragraphs = re.split(r"\n\s*\n", text)

    unique_paragraphs = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        normalized = re.sub(
            r"\s+",
            " ",
            paragraph.lower()
        ).strip()

        paragraph_hash = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        if paragraph_hash in seen_paragraphs:
            continue

        seen_paragraphs.add(paragraph_hash)

        unique_paragraphs.append(paragraph)

    return "\n\n".join(unique_paragraphs)


# ============================================================
# 14. PROCESS ALL TXT FILES
# ============================================================

def process_dataset():

    seen_documents = set()
    seen_paragraphs = set()

    total_files = 0
    kept_files = 0
    duplicate_files = 0

    all_documents = []

    txt_files = list(INPUT_DIR.rglob("*.txt"))

    print(f"Found {len(txt_files)} TXT files.")

    for file_path in txt_files:

        total_files += 1

        try:

            raw_text = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            cleaned_text = clean_document(raw_text)

            if not cleaned_text:
                continue

            # Remove duplicate paragraphs
            cleaned_text = remove_duplicate_paragraphs(
                cleaned_text,
                seen_paragraphs
            )

            if not cleaned_text:
                continue

            # Exact document duplicate detection
            doc_hash = document_hash(cleaned_text)

            if doc_hash in seen_documents:

                duplicate_files += 1

                continue

            seen_documents.add(doc_hash)

            all_documents.append(cleaned_text)

            kept_files += 1

        except Exception as e:

            print(
                f"Error processing {file_path}: {e}"
            )

    # Write final corpus
    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        for document in all_documents:

            f.write(document)

            f.write("\n\n")

    print("\n========== DONE ==========")

    print(
        f"Total files: {total_files}"
    )

    print(
        f"Kept documents: {kept_files}"
    )

    print(
        f"Duplicate documents removed: {duplicate_files}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    process_dataset()