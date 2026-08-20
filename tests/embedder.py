from pathlib import Path
from time import perf_counter

import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.loaders.pdf_loader import PDFLoader
from src.chunking.researchpaper import SectionAwareChunker
from src.embeddings.embedding import LocalEmbedder


# ============================================================
# CONFIG
# ============================================================

PDF_PATH = Path("data/raw/test_paper.pdf")

QUERY = "What is Retrieval-Augmented Generation?"

TOP_K = 5


# ============================================================
# TIMER HELPER
# ============================================================

def elapsed(start):
    return perf_counter() - start


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = perf_counter()

    timings = {}


    # ========================================================
    # 0. HARDWARE
    # ========================================================

    print("\n" + "=" * 60)
    print("HARDWARE")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()

    print("CUDA available:", cuda_available)

    if cuda_available:
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
    else:
        print("GPU: CPU only")


    # ========================================================
    # 1. LOADER INITIALIZATION
    # ========================================================

    print("\n" + "=" * 60)
    print("LOADER INITIALIZATION")
    print("=" * 60)

    start = perf_counter()

    loader = PDFLoader()

    timings["loader_init"] = elapsed(start)

    print(
        "Loader initialized in:",
        f"{timings['loader_init']:.2f}s"
    )


    # ========================================================
    # 2. PDF CONVERSION + EXTRACTION
    # ========================================================

    print("\n" + "=" * 60)
    print("PDF LOADING")
    print("=" * 60)

    start = perf_counter()

    document = loader.load(PDF_PATH)

    timings["pdf_load"] = elapsed(start)

    print("Name:", document.document_name)
    print("Document ID:", document.document_id)
    print("Pages:", len(document.pages))

    print(
        "\nPDF load time:",
        f"{timings['pdf_load']:.2f}s"
    )


    # ========================================================
    # 3. CHUNKING
    # ========================================================

    print("\n" + "=" * 60)
    print("CHUNKING")
    print("=" * 60)

    start = perf_counter()

    chunker = SectionAwareChunker()

    chunks = chunker.chunk(document)

    timings["chunking"] = elapsed(start)

    print("Total chunks:", len(chunks))

    print(
        "\nChunking time:",
        f"{timings['chunking']:.2f}s"
    )


    # ========================================================
    # 4. CHUNK METADATA CHECK
    # ========================================================

    print("\n" + "=" * 60)
    print("CHUNK METADATA")
    print("=" * 60)

    for chunk in chunks[:5]:

        print("\n" + "-" * 60)

        print("Chunk ID:", chunk.id)

        print("Document ID:", chunk.document_id)

        print(
            "Page:",
            chunk.metadata.get("page_number")
        )

        print(
            "Chunk type:",
            chunk.metadata.get("chunk_type")
        )

        print(
            "Section:",
            chunk.metadata.get("section_path")
        )

        print(
            "Strategy:",
            chunk.metadata.get("strategy")
        )

        print(
            "Text:",
            chunk.text[:250]
            .replace("\n", " ")
        )


    # ========================================================
    # 5. EMBEDDING MODEL INITIALIZATION
    # ========================================================

    print("\n" + "=" * 60)
    print("EMBEDDING MODEL INITIALIZATION")
    print("=" * 60)

    start = perf_counter()

    embedder = LocalEmbedder()

    timings["model_init"] = elapsed(start)

    print("Dimension:", embedder.dimension)

    print(
        "\nModel initialization:",
        f"{timings['model_init']:.2f}s"
    )


    # ========================================================
    # 6. PREPARE TEXT
    # ========================================================

    start = perf_counter()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    timings["text_prepare"] = elapsed(start)

    print("\nTexts prepared:", len(texts))

    print(
        "Text preparation:",
        f"{timings['text_prepare']:.2f}s"
    )


    # ========================================================
    # 7. DOCUMENT EMBEDDINGS
    # ========================================================

    print("\n" + "=" * 60)
    print("DOCUMENT EMBEDDINGS")
    print("=" * 60)

    start = perf_counter()

    vectors = embedder.embed_documents(texts)

    timings["document_embedding"] = elapsed(start)

    print("Embeddings created.")

    print(
        "\nDocument embedding:",
        f"{timings['document_embedding']:.2f}s"
    )


    # ========================================================
    # 8. EMBEDDING VERIFICATION
    # ========================================================

    print("\n" + "=" * 60)
    print("EMBEDDING VERIFICATION")
    print("=" * 60)

    assert len(chunks) == len(vectors), (
        f"Chunk/vector mismatch: "
        f"{len(chunks)} chunks vs "
        f"{len(vectors)} vectors"
    )

    assert all(
        len(vector) == embedder.dimension
        for vector in vectors
    ), "Vector dimension mismatch"

    print("Chunks:", len(chunks))
    print("Vectors:", len(vectors))
    print("Vector dimension:", embedder.dimension)

    print("\n✓ Chunk count == vector count")
    print("✓ All vectors have correct dimension")


    # ========================================================
    # 9. QUERY EMBEDDING
    # ========================================================

    print("\n" + "=" * 60)
    print("QUERY EMBEDDING")
    print("=" * 60)

    print("Query:", QUERY)

    start = perf_counter()

    query_vector = embedder.embed_query(QUERY)

    timings["query_embedding"] = elapsed(start)

    print(
        "Query vector dimension:",
        len(query_vector)
    )

    assert len(query_vector) == embedder.dimension

    print("✓ Query vector dimension is correct")

    print(
        "\nQuery embedding:",
        f"{timings['query_embedding']:.2f}s"
    )


    # ========================================================
    # 10. COSINE SIMILARITY
    # ========================================================

    print("\n" + "=" * 60)
    print("SIMILARITY SEARCH")
    print("=" * 60)

    start = perf_counter()

    scores = cosine_similarity(
        [query_vector],
        vectors
    )[0]

    timings["similarity"] = elapsed(start)

    print("Similarity scores calculated.")

    print(
        "\nSimilarity calculation:",
        f"{timings['similarity']:.2f}s"
    )


    # ========================================================
    # 11. RANK RESULTS
    # ========================================================

    start = perf_counter()

    ranked_indices = scores.argsort()[::-1]

    timings["ranking"] = elapsed(start)


    # ========================================================
    # 12. TOP-K RESULTS
    # ========================================================

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
            "Page:",
            chunk.metadata.get("page_number")
        )

        print(
            "Chunk type:",
            chunk.metadata.get("chunk_type")
        )

        print(
            "Section:",
            chunk.metadata.get("section_path")
        )

        print(
            "Text:",
            chunk.text[:500]
            .replace("\n", " ")
        )


    # ========================================================
    # 13. TOTAL PERFORMANCE
    # ========================================================

    timings["total"] = elapsed(total_start)


    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    print(
        f"{'Loader initialization:':30}",
        f"{timings['loader_init']:.2f}s"
    )

    print(
        f"{'PDF loading:':30}",
        f"{timings['pdf_load']:.2f}s"
    )

    print(
        f"{'Chunking:':30}",
        f"{timings['chunking']:.2f}s"
    )

    print(
        f"{'Model initialization:':30}",
        f"{timings['model_init']:.2f}s"
    )

    print(
        f"{'Text preparation:':30}",
        f"{timings['text_prepare']:.2f}s"
    )

    print(
        f"{'Document embedding:':30}",
        f"{timings['document_embedding']:.2f}s"
    )

    print(
        f"{'Query embedding:':30}",
        f"{timings['query_embedding']:.2f}s"
    )

    print(
        f"{'Similarity search:':30}",
        f"{timings['similarity']:.2f}s"
    )

    print(
        f"{'Ranking:':30}",
        f"{timings['ranking']:.2f}s"
    )

    print("-" * 60)

    print(
        f"{'TOTAL:':30}",
        f"{timings['total']:.2f}s"
    )


    # ========================================================
    # 14. SUCCESS
    # ========================================================

    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)

    print("Document loading       ✓")
    print("Chunking               ✓")
    print("Chunk metadata         ✓")
    print("Document embeddings    ✓")
    print("Vector verification    ✓")
    print("Query embedding        ✓")
    print("Similarity calculation ✓")
    print("Top-K retrieval        ✓")


if __name__ == "__main__":
    main()