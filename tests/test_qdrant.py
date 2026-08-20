from src.vector_store.qdrant import QdrantVectorStore


def main():

    store = QdrantVectorStore(
        collection_name="researchrag_chunks",
        vector_size=768,
    )

    store.create_collection()

    print("Collection created successfully.")


if __name__ == "__main__":
    main()