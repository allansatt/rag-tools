import argparse
import ollama
import chromadb

def main():
    """
    Processes a user query by generating an embedding, querying ChromaDB,
    and then passing the query and retrieved context to an Ollama LLM.
    """
    parser = argparse.ArgumentParser(
        description="Query a ChromaDB collection and generate a response using an LLM."
    )
    parser.add_argument("query", type=str, help="The query to process.")
    args = parser.parse_args()

    query_text = args.query
    embedding_model_name = "nomic-embed-text"
    llm_model_name = "llama3.2"
    collection_name = "lancer"
    num_results = 20

    # 1. Create embedding for the query
    try:
        print(f"Generating embedding for query using {embedding_model_name}...")
        query_embedding_response = ollama.embed(
            model=embedding_model_name,
            input=query_text
        )
        query_embeddings = query_embedding_response.get("embeddings")
        if not query_embeddings:
            print("Error: Could not generate embedding for the query. Response did not contain 'embedding'.")
            return
        print("Embedding generated successfully.")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        print(f"Ensure Ollama is running and the model '{embedding_model_name}' is available (e.g., run 'ollama pull {embedding_model_name}').")
        return

    # 2. Query ChromaDB
    try:
        print(f"Connecting to ChromaDB and querying collection '{collection_name}'...")
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Check if collection exists
        try:
            collection = chroma_client.get_collection(name=collection_name)
        except Exception: # Catches specific errors like collection not found
            print(f"Error: Collection '{collection_name}' not found in ChromaDB.")
            print(f"Please ensure the collection has been created and populated (e.g., using a script like 'load_pdf.py').")
            return

        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=num_results,
            include=['documents']  # We only need the document texts for context
        )

        retrieved_documents = results.get('documents', [[]])[0]
        
        if not retrieved_documents:
            print("No documents found in ChromaDB for the query.")
            context_str = "No relevant documents found in the knowledge base."
        else:
            print(f"Retrieved {len(retrieved_documents)} documents from ChromaDB.")
            context_str = "\n\n---\n\n".join(retrieved_documents)

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        print("Ensure ChromaDB server is running at localhost:8000.")
        return
 
    # 3. Pass original query and context to LLM
    prompt = f"""Use the following context to anwer questions about the Tabletop Role Playing Game "Lancer". Respond only with the contents of the rules.

Context from documents:
---
{context_str}
---

User Query: {query_text}

Answer:"""

    try:
        print(f"Sending query and context to LLM model '{llm_model_name}'...")
        response = ollama.chat(
            model=llm_model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ]
        )
        llm_response = response.get('message', {}).get('content', '')
        
        print("\nLLM Response:")
        print(llm_response)

    except Exception as e:
        print(f"Error communicating with Ollama LLM: {e}")
        print(f"Ensure Ollama is running and the model '{llm_model_name}' is available (e.g., run 'ollama pull {llm_model_name}').")
        return

if __name__ == "__main__":
    main()