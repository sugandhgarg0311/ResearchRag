from src.chunking.base import Chunk
from src.vector_store.qdrant import QdrantVectorStore


def main():

    print("=" * 60)
    print("QDRANT UPSERT TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Initialize Qdrant
    # ---------------------------------------------------------

    print("\n[1] Initializing collection...")

    store = QdrantVectorStore(
        collection_name="researchrag_chunks",
        vector_size=768,
    )

    store.create_collection()

    # ---------------------------------------------------------
    # 2. Create a test chunk
    # ---------------------------------------------------------

    chunk = Chunk(
        id="2a56893cdb4560021a4ccb044e3083e3a0202e6a5faf2ed81d17834ac0a098c6",
        document_id="test-document-001",
        text="Retrieval-Augmented Generation combines retrieval with generation.",
        metadata={
            "document_name": "test_paper.pdf",
            "source_path": "data/test_paper.pdf",
            "mime_type": "application/pdf",
            "page_number": 1,
            "chunk_index": 0,
            "strategy": "baseline",
            "chunk_type": "text",
            "section_path": ["Introduction"],
        },
    )

    print("\n[2] Chunk created")
    print(f"Chunk ID:       {chunk.id}")
    print(f"Document ID:   {chunk.document_id}")
    print(f"Page:          {chunk.metadata['page_number']}")
    print(f"Text:          {chunk.text}")

    # ---------------------------------------------------------
    # 3. Create fake 768-dimensional vector
    # ---------------------------------------------------------

    vector = [0.01] * 768

    print("\n[3] Vector created")
    print(f"Vector dimension: {len(vector)}")

    # ---------------------------------------------------------
    # 4. Upsert
    # ---------------------------------------------------------

    print("\n[4] Upserting point...")

    store.upsert(
        chunks=[chunk],
        vectors=[vector],
    )

    print("✓ Upsert completed")

    # ---------------------------------------------------------
    # 5. Convert canonical chunk ID → Qdrant ID
    # ---------------------------------------------------------

    qdrant_id = store._to_qdrant_id(chunk.id)

    print("\n[5] ID mapping")

    print("Canonical Chunk ID:")
    print(f"  {chunk.id}")

    print("\nQdrant Point ID:")
    print(f"  {qdrant_id}")

    # ---------------------------------------------------------
    # 6. Retrieve using Qdrant ID
    # ---------------------------------------------------------

    print("\n[6] Retrieving point...")

    result = store.client.retrieve(
        collection_name="researchrag_chunks",
        ids=[qdrant_id],
        with_vectors=True,
    )

    # ---------------------------------------------------------
    # 7. Verify
    # ---------------------------------------------------------

    if not result:
        print("❌ Point was not found.")
        return

    point = result[0]

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    print(f"\nQdrant Point ID:")
    print(f"  {point.id}")

    print(f"\nVector dimension:")
    print(f"  {len(point.vector)}")

    print("\nPayload:")

    for key, value in point.payload.items():
        print(f"  {key}: {value}")

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    assert point.id == qdrant_id

    assert len(point.vector) == 768

    assert point.payload["chunk_id"] == chunk.id

    assert point.payload["text"] == chunk.text

    assert point.payload["document_id"] == chunk.document_id

    assert point.payload["page_number"] == 1

    assert point.payload["chunk_type"] == "text"

    # ---------------------------------------------------------
    # Success
    # ---------------------------------------------------------

    print("\n✓ Qdrant ID is correct")
    print("✓ Vector dimension is correct")
    print("✓ Original chunk ID preserved")
    print("✓ Text stored correctly")
    print("✓ Document ID stored correctly")
    print("✓ Page number stored correctly")
    print("✓ Chunk type stored correctly")

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()