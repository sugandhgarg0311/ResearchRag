from pathlib import Path

from src.loaders.pdf_loader import PDFLoader
from src.chunking.researchpaper import SectionAwareChunker


# ============================================================
# 1. LOAD PDF
# ============================================================

loader = PDFLoader()

document = loader.load(
    Path("data/raw/test_paper.pdf")
)

print("=" * 80)
print("DOCUMENT")
print("=" * 80)

print("Document:", document.document_name)
print("Document ID:", document.document_id)
print("Pages:", len(document.pages))
print("Blocks:", document.metadata["block_count"])

print(
    "Block types:",
    document.metadata["block_types"]
)


# ============================================================
# 2. INSPECT BLOCKS
# ============================================================

print()
print("=" * 80)
print("BLOCK INSPECTION")
print("=" * 80)

for page in document.pages:

    print("\n======================")
    print("PAGE:", page.page_number)
    print("======================")

    for block in page.blocks:

        print(
            f"\nTYPE: {block.block_type}"
        )

        print(
            f"TEXT: {block.text[:300]}"
        )

        print(
            f"METADATA: {block.metadata}"
        )


# ============================================================
# 3. CHUNK
# ============================================================

chunker = SectionAwareChunker(
    max_tokens=800,
)

chunks = chunker.chunk(document)


# ============================================================
# 4. CHUNK SUMMARY
# ============================================================

print()
print("=" * 80)
print("CHUNK SUMMARY")
print("=" * 80)

print(
    "Total chunks:",
    len(chunks)
)


# ============================================================
# 5. CHUNKS
# ============================================================

print()
print("=" * 80)
print("CHUNKS")
print("=" * 80)

for chunk in chunks[:10]:

    print("\n--- CHUNK ---")

    print(
        "ID:",
        chunk.id[:16],
        "..."
    )

    print(
        "Document ID:",
        chunk.document_id[:16],
        "..."
    )

    print(
        "Page:",
        chunk.metadata["page_number"]
    )

    print(
        "Chunk index:",
        chunk.metadata["chunk_index"]
    )

    print(
        "Section:",
        " > ".join(
            chunk.metadata.get(
                "section_path",
                []
            )
        )
    )

    print(
        "Type:",
        chunk.metadata.get(
            "chunk_type"
        )
    )

    print(
        "Tokens:",
        chunk.metadata["token_count"]
    )

    print(
        "Strategy:",
        chunk.metadata["strategy"]
    )

    print(
        "Text:"
    )

    print(chunk.text)


# ============================================================
# 6. VALIDATION
# ============================================================

print()
print("=" * 80)
print("VALIDATION")
print("=" * 80)


# ------------------------------------------------------------
# Token limit
# ------------------------------------------------------------

oversized = [
    chunk
    for chunk in chunks
    if chunk.metadata["token_count"]
    > chunker.max_tokens
]

print(
    "Maximum chunk tokens:",
    max(
        chunk.metadata["token_count"]
        for chunk in chunks
    )
)

print(
    "Configured maximum:",
    chunker.max_tokens
)

print(
    "Oversized chunks:",
    len(oversized)
)


# ------------------------------------------------------------
# Required metadata
# ------------------------------------------------------------

required_metadata = {
    "document_name",
    "source_path",
    "mime_type",
    "page_number",
    "chunk_index",
    "strategy",
}

invalid_metadata = []

for chunk in chunks:

    missing = (
        required_metadata
        - chunk.metadata.keys()
    )

    if missing:
        invalid_metadata.append(
            (chunk.id, missing)
        )

print(
    "Chunks missing required metadata:",
    len(invalid_metadata)
)


# ------------------------------------------------------------
# IDs
# ------------------------------------------------------------

missing_ids = [
    chunk
    for chunk in chunks
    if not chunk.id
]

print(
    "Chunks without IDs:",
    len(missing_ids)
)


# ------------------------------------------------------------
# Section context
# ------------------------------------------------------------

sectionless = [
    chunk
    for chunk in chunks
    if not chunk.metadata.get(
        "section_path"
    )
]

print(
    "Chunks without section path:",
    len(sectionless)
)


# ------------------------------------------------------------
# Atomic chunks
# ------------------------------------------------------------

atomic_chunks = [
    chunk
    for chunk in chunks
    if chunk.metadata.get(
        "atomic",
        False
    )
]

print(
    "Atomic chunks:",
    len(atomic_chunks)
)


# ============================================================
# 7. PAGE CHUNK INDEX VALIDATION
# ============================================================

print()
print("=" * 80)
print("PAGE INDEX VALIDATION")
print("=" * 80)

chunks_by_page = {}

for chunk in chunks:

    page = chunk.metadata["page_number"]

    chunks_by_page.setdefault(
        page,
        []
    ).append(chunk)


page_index_errors = []

for page, page_chunks in chunks_by_page.items():

    expected = list(
        range(len(page_chunks))
    )

    actual = [
        chunk.metadata["chunk_index"]
        for chunk in page_chunks
    ]

    if actual != expected:

        page_index_errors.append(
            {
                "page": page,
                "expected": expected,
                "actual": actual,
            }
        )

print(
    "Pages with chunk-index errors:",
    len(page_index_errors)
)


# ============================================================
# 8. FINAL RESULT
# ============================================================

print()
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)

if not chunks:

    print(
        "❌ FAIL: No chunks were produced."
    )

elif oversized:

    print(
        "❌ FAIL: Some chunks exceed max_tokens."
    )

elif invalid_metadata:

    print(
        "❌ FAIL: Some chunks are missing required metadata."
    )

elif missing_ids:

    print(
        "❌ FAIL: Some chunks have no ID."
    )

elif page_index_errors:

    print(
        "❌ FAIL: chunk_index is incorrect."
    )

else:

    print(
        "✅ LOADER + SECTION-AWARE CHUNKER TEST PASSED"
    )