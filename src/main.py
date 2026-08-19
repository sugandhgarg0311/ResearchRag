"""
RAG Learning Assistant - Main Entry Point

WHAT THIS DOES:
Orchestrates the entire RAG pipeline from loading documents to answering questions.

COMPLETE PIPELINE:
1. Load documents from docs/ folder
2. Chunk them into smaller pieces
3. Generate embeddings for each chunk
4. Add embeddings to vector database
5. When user asks a question:
   a. Retrieve relevant chunks
   b. Format as context
   c. Send to LLM with question
   d. Return answer + citations
"""

from documents import load_documents
from chunking import chunk_documents
from embeddings import embed_chunks
import vectordb as vdb
from llm import generate_answer
from config import DEBUG



        
def initialize_rag_pipeline():
    """
    Load and prepare all data for RAG.

    This runs once at startup to:
    - Load documents
    - Chunk them
    - Embed them
    - Add them to vector database
    """

    if DEBUG:
        print("\n" + "=" * 60)
        print("INITIALIZING RAG PIPELINE")
        print("=" * 60)

    # Step 1: Load documents
    if DEBUG:
        print("\n📄 Step 1: Loading documents...")

    documents = load_documents()

    if not documents:
        print("❌ No documents found in docs/ folder!")
        return False

    # Step 2: Chunk documents
    if DEBUG:
        print("\n✂️  Step 2: Chunking documents...")

    chunks = chunk_documents(documents)

    if not chunks:
        print("❌ No chunks created!")
        return False

    # Step 3: Generate embeddings
    if DEBUG:
        print("\n🔢 Step 3: Generating embeddings...")

    chunk_embeddings = embed_chunks(chunks)

    # Step 4: Create Qdrant collection
    if DEBUG:
        print("\n💾 Step 4: Creating Qdrant collection...")

    vdb.create_collection()

    # Step 5: Add chunks to Qdrant
    if DEBUG:
        print("💾 Step 5: Adding to vector database...")

    vdb.add_chunks(chunk_embeddings)

    # Step 6: Check database
    if DEBUG:
        print("\n✅ RAG Pipeline Ready!")

        stats = vdb.get_stats()

        print(
            f"   Total chunks in DB: "
            f"{stats['total_chunks']}"
        )

    return True


def ask_question(question: str):
    """
    Ask a question and get an answer.
    
    Args:
        question: User's question
    
    Returns:
        Dictionary with answer and sources
    """
    result = generate_answer(question)
    
    # Print result
    print(f"\n{'='*60}")
    print(f"ANSWER")
    print(f"{'='*60}")
    print(f"\n{result['answer']}")
    
    print(f"\n{'='*60}")
    print(f"SOURCES")
    print(f"{'='*60}")
    for i, (source, score) in enumerate(zip(result['sources'], result['scores']), 1):
        print(f"[{i}] {source} (similarity: {score:.4f})")
    
    return result


def interactive_mode():
    """
    Run the RAG assistant in interactive mode.
    
    User can ask multiple questions until they type 'quit'.
    """
    print("\n" + "="*60)
    print("RAG LEARNING ASSISTANT - INTERACTIVE MODE")
    print("="*60)
    print("Type your question and press Enter")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() == 'quit':
            print("\nGoodbye! 👋")
            break
        
        if not question:
            continue
        
        try:
            ask_question(question)
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """
    Main entry point for the RAG assistant.
    
    Initializes the pipeline and starts interactive mode.
    """
    # Initialize RAG pipeline
    success = initialize_rag_pipeline()
    
    if not success:
        print("\n❌ Failed to initialize RAG pipeline")
        return
    
    # Start interactive mode
    interactive_mode()


if __name__ == "__main__":
    main()
