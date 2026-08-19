from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from src.loaders.pdf_loader import PDFLoader
from src.chunking.researchpaper import SectionAwareChunker
from src.embeddings.embedding import LocalEmbedder


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

PDF_PATH = Path("data/raw/test_paper.pdf")

TOP_K = 5

QUERY = "What is Retrieval-Augmented Generation?"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    # ==================================================
    # 1. LOAD DOCUMENT
    # ==================================================

    print("\n" + "=" * 60)
    print("DOCUMENT")
    print("=" * 60)

    loader = PDFLoader()

    document = loader.load(PDF_PATH)

    print("Name:", document.document_name)
    print("Document ID:", document.document_id)
    print("Pages:", len(document.pages))


    # ==================================================
    # 2. CHUNK DOCUMENT
    # ==================================================

    print("\n" + "=" * 60)
    print("CHUNKING")
    print("=" * 60)

    chunker = SectionAwareChunker()

    chunks = chunker.chunk(document)

    print("Total chunks:", len(chunks))


    # ==================================================
    # 3. CREATE EMBEDDING MODEL
    # ==================================================

    print("\n" + "=" * 60)
    print("EMBEDDING MODEL")
    print("=" * 60)

    embedder = LocalEmbedder()

    print("Model:", embedder.model)
    print("Dimension:", embedder.dimension)


    # ==================================================
    # 4. EXTRACT TEXT FROM CHUNKS
    # ==================================================

    texts = [
        chunk.text
        for chunk in chunks
    ]

    print("\nTexts prepared:", len(texts))


    # ==================================================
    # 5. CREATE DOCUMENT EMBEDDINGS
    # ==================================================

    print("\n" + "=" * 60)
    print("CREATING DOCUMENT EMBEDDINGS")
    print("=" * 60)

    vectors = embedder.embed_documents(texts)

    print("Embeddings created.")


    # ==================================================
    # 6. VERIFY CHUNK ↔ VECTOR MAPPING
    # ==================================================

    print("\n" + "=" * 60)
    print("EMBEDDING VERIFICATION")
    print("=" * 60)

    # Number of chunks must equal number of vectors
    assert len(chunks) == len(vectors), (
        f"Mismatch: {len(chunks)} chunks "
        f"but {len(vectors)} vectors"
    )

    # Every vector must have the same dimension
    assert all(
        len(vector) == embedder.dimension
        for vector in vectors
    ), "Vector dimension mismatch"

    print("Chunks:", len(chunks))
    print("Vectors:", len(vectors))
    print("Vector dimension:", embedder.dimension)

    print("\n✓ Chunk count == vector count")
    print("✓ All vectors have correct dimension")


    # ==================================================
    # 7. SHOW SAMPLE CHUNKS + VECTORS
    # ==================================================

    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS")
    print("=" * 60)

    for chunk, vector in zip(chunks[:5], vectors[:5]):

        print("\n" + "-" * 60)

        print("Chunk ID:", chunk.id)

        print("Document ID:", chunk.document_id)

        print(
            "Page:",
            chunk.metadata.get("page_number")
        )

        print(
            "Block Type:",
            chunk.metadata.get("block_type")
        )

        print(
            "Section:",
            chunk.metadata.get("heading_path")
        )

        print(
            "Text:",
            chunk.text[:300]
            .replace("\n", " ")
        )

        print(
            "Vector dimension:",
            len(vector)
        )

        print(
            "First 5 vector values:",
            vector[:5]
        )


    # ==================================================
    # 8. EMBED USER QUERY
    # ==================================================

    print("\n" + "=" * 60)
    print("QUERY EMBEDDING")
    print("=" * 60)

    print("Query:", QUERY)

    query_vector = embedder.embed_query(QUERY)

    print(
        "Query vector dimension:",
        len(query_vector)
    )

    assert len(query_vector) == embedder.dimension

    print("✓ Query vector dimension is correct")


    # ==================================================
    # 9. CALCULATE COSINE SIMILARITY
    # ==================================================

    print("\n" + "=" * 60)
    print("SIMILARITY SEARCH")
    print("=" * 60)

    scores = cosine_similarity(
        [query_vector],
        vectors
    )[0]

    print("Similarity scores calculated.")


    # ==================================================
    # 10. RANK CHUNKS
    # ==================================================

    ranked_indices = scores.argsort()[::-1]


    # ==================================================
    # 11. DISPLAY TOP-K RESULTS
    # ==================================================

    print("\n" + "=" * 60)
    print(f"TOP {TOP_K} RESULTS")
    print("=" * 60)

    for rank, index in enumerate(
        ranked_indices[:TOP_K],
        start=1
    ):

        chunk = chunks[index]

        print("\n" + "-" * 60)

        print("Rank:", rank)

        print(
            "Similarity:",
            round(float(scores[index]), 4)
        )

        print("Chunk ID:", chunk.id)

        print(
            "Document ID:",
            chunk.document_id
        )

        print(
            "Page:",
            chunk.metadata.get("page_number")
        )

        print(
            "Block Type:",
            chunk.metadata.get("block_type")
        )

        print(
            "Section:",
            chunk.metadata.get("heading_path")
        )

        print(
            "Text:",
            chunk.text[:500]
            .replace("\n", " ")
        )


    # ==================================================
    # 12. FINAL RESULT
    # ==================================================

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print("Document loading       ✓")
    print("Chunking               ✓")
    print("Document embeddings    ✓")
    print("Vector verification    ✓")
    print("Query embedding        ✓")
    print("Cosine similarity      ✓")
    print("Top-K retrieval        ✓")


if __name__ == "__main__":
    main()