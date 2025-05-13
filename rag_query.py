import argparse
import ollama
import chromadb

def query_with_rag(embedding_model_name, collection_name, chunk_length, num_results, query_text):
    """
    Processes a user query by generating an embedding, querying ChromaDB,
    and then passing the query and retrieved context to an Ollama LLM.
    """

    llm_model_name = "llama3.2"

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
            collection = chroma_client.get_collection(name=f'{collection_name}')
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
            context_str = "\n---------\n".join(retrieved_documents)

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        print("Ensure ChromaDB server is running at localhost:8000.")
        return
 
    # 3. Pass original query and context to LLM
    prompt = f"""Use the following context to anwer questions about the Tabletop Role Playing Game "Lancer".

Context from documents:
---
{context_str}
--
User Query: {query_text}"""

    # 4. Write the query to a file named in the queries directory named "lancer_{timestamp}.txt"
    # (timestamp is the current time in seconds since epoch)
    import time
    import os # Import the os module
    timestamp = int(time.time())
    try:
        print(f"Sending query and context to LLM model '{llm_model_name}'...")
        response: ollama.ChatResponse = ollama.chat(
            model=llm_model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ]
        )
        llm_response = response.get('message', {}).get('content', '')
        storage_path = f'./responses/{collection_name}/{chunk_length}_chunks/{num_results}_docs/{timestamp}'

        # Create the queries directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        print("\nLLM Response:")
        print(llm_response)

    except Exception as e:
        print(f"Error communicating with Ollama LLM: {e}")
        print(f"Ensure Ollama is running and the model '{llm_model_name}' is available (e.g., run 'ollama pull {llm_model_name}').")
        return
    query_filename = f"{storage_path}/query.txt"
    with open(query_filename, 'w') as f:
        f.write(prompt)
    response_filename = f"{storage_path}/response.txt"
    with open(response_filename, 'w') as f:
        f.write(response.message.content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query a ChromaDB collection and generate a response using an LLM."
    )
    parser.add_argument('-e','--embedding_model_name', type=str, help="The name of the collection to query.", default="nomic-embed-text")
    parser.add_argument('-d','--documents', type=str, help="The number of documents to use", default=15)
    parser.add_argument('-c','--chunk_length', type=str, help="The length of chunks in the collection, for grouping outputs.", default="unknown")
    parser.add_argument('collection', type=str, help="The name of the collection to query.")
    parser.add_argument("query", type=str, help="The query to process.")
    args = parser.parse_args()

    
    query_with_rag(embedding_model_name=args.embedding_model_name, collection_name=args.collection, num_results=args.documents, chunk_length=args.chunk_length, query_text=args.query)