from pathlib import Path

from src.loaders.pdf_loader import PDFLoader
from src.chunking.reserachpaper import SectionAwareChunker


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
    overlap_tokens=100,
    min_chunk_tokens=50,
)

parents, chunks = chunker.chunk(
    document
)


# ============================================================
# 4. CHUNK SUMMARY
# ============================================================

print()
print("=" * 80)
print("CHUNK SUMMARY")
print("=" * 80)

print(
    "Parent sections:",
    len(parents)
)

print(
    "Child chunks:",
    len(chunks)
)


# ============================================================
# 5. PARENT SECTIONS
# ============================================================

print()
print("=" * 80)
print("PARENT SECTIONS")
print("=" * 80)

for parent in parents:

    print("\n--- PARENT ---")

    print(
        "ID:",
        parent.parent_id[:16],
        "..."
    )

    print(
        "Title:",
        parent.title
    )

    print(
        "Section:",
        " > ".join(
            parent.section_path
        )
    )

    print(
        "Pages:",
        parent.page_numbers
    )

    print(
        "Tokens:",
        parent.token_count
    )


# ============================================================
# 6. CHILD CHUNKS
# ============================================================

print()
print("=" * 80)
print("CHILD CHUNKS")
print("=" * 80)

for chunk in chunks[:10]:

    print("\n--- CHUNK ---")

    print(
        "Index:",
        chunk.chunk_index
    )

    print(
        "ID:",
        chunk.chunk_id[:16],
        "..."
    )

    print(
        "Parent:",
        chunk.parent_id[:16],
        "..."
    )

    print(
        "Pages:",
        chunk.page_numbers
    )

    print(
        "Section:",
        " > ".join(
            chunk.section_path
        )
    )

    print(
        "Block types:",
        chunk.block_types
    )

    print(
        "Words:",
        chunk.word_count
    )

    print(
        "Tokens:",
        chunk.token_count
    )

    print(
        "Text:"
    )

    print(chunk.text)

    print(
        "\nEmbedding text:"
    )

    print(chunk.embedding_text)


# ============================================================
# 7. VALIDATION
# ============================================================

print()
print("=" * 80)
print("VALIDATION")
print("=" * 80)


# ---- Token limit --------------------------------------------

oversized = [
    chunk
    for chunk in chunks
    if chunk.token_count > chunker.max_tokens
]

print(
    "Maximum chunk tokens:",
    max(
        chunk.token_count
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


# ---- Atomic blocks ------------------------------------------

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


# ---- Deterministic IDs --------------------------------------

missing_ids = [
    chunk
    for chunk in chunks
    if not chunk.chunk_id
]

print(
    "Chunks without IDs:",
    len(missing_ids)
)


# ---- Parent IDs ---------------------------------------------

missing_parents = [
    chunk
    for chunk in chunks
    if not chunk.parent_id
]

print(
    "Chunks without parent:",
    len(missing_parents)
)


# ---- Section context ----------------------------------------

contextless = [
    chunk
    for chunk in chunks
    if (
        chunk.section_path
        and not chunk.embedding_text.startswith(
            "Section:"
        )
    )
]

print(
    "Chunks without embedding section context:",
    len(contextless)
)


# ============================================================
# 8. FINAL RESULT
# ============================================================

print()
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)

if oversized:
    print("❌ FAIL: Some chunks exceed max_tokens")

elif missing_ids:
    print("❌ FAIL: Some chunks have no chunk ID")

elif missing_parents:
    print("❌ FAIL: Some chunks have no parent ID")

elif contextless:
    print("❌ FAIL: Some chunks lack section context")

else:
    print("✅ LOADER + CHUNKER TEST PASSED")