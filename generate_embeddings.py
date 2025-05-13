import json
import os
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
import ollama
import chromadb
import argparse

def _load_embeddings(embedings_path, algorithm, chunks, overwrite):
    if not overwrite and os.path.exists(embedings_path):
        print(f"Loading existing embeddings from {embedings_path}")
        with open(embedings_path, 'r') as f:
            document = json.load(f)
        return document
    embeddings = ollama.embed(model=algorithm, input=chunks).get("embeddings", [])
    with open(embedings_path, 'w') as f:
        json.dump(embeddings, f)
    return embeddings
 
def get_chunks_from_input(input_filename, collection_prefix, chunk_length, algorithm, overwrite=False):
    collection_name = f'{collection_prefix}_{algorithm}_{chunk_length}'
    file_path = f'./data/{input_filename}'
    embeddings_path = f'./embeddings/{collection_prefix}/{algorithm}/{chunk_length}'

    with open(file_path, 'r') as f:
        markdown = f.read()
    
    mardown_splitter = MarkdownTextSplitter()
    character_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_length, chunk_overlap=chunk_length//4)
    header_split = mardown_splitter.create_documents([markdown])
    chunks =[c.page_content for c in character_splitter.split_documents(header_split)]
    metadata = [{'filepath':file_path} for _ in chunks]
    if not chunks:
        print(f"No processable content found in {input_filename} after PDF processing and filtering.")
        return # Exit if processing occurred but yielded no chunksz
    
    embeddings = _load_embeddings(embeddings_path, algorithm, chunks, overwrite)
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collections = chroma_client.list_collections()
    if any(c.name == collection_name for c in collections):
        if not overwrite:
            print(f"Collection {collection_name} already exists. Use --overwrite to overwrite.")
            return
        try:
            chroma_client.delete_collection(collection_name)
        except Exception as e:
            breakpoint()
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space":"cosine"})
    except Exception as e:
        breakpoint()
    ids = [f"{input_filename}_{i}" for i in range(len(embeddings))]
    batch_size = 1000
    for i in range(0, len(embeddings), batch_size):
        try:
            collection.add(ids=ids[i:i+batch_size],documents=chunks[i:i+batch_size],metadatas=metadata[i:i+batch_size],embeddings=embeddings[i:i+batch_size]
            )
        except Exception as e:
            breakpoint()
            raise e
        print(f"Batch {i//batch_size + 1}  of size {batch_size} added to collection {collection_name}.")
    print(f"Successfully loaded {len(ids)} chunks from {input_filename} into collection {collection_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a PDF, process its content, and store embeddings in ChromaDB.")
    parser.add_argument("input_filename", help="The name of the input PDF file (located in ./data/ directory).")
    parser.add_argument("collection_prefix", help="The prefix of the ChromaDB collection to create/use.")
    parser.add_argument("chunk_length", help="The chunk length for splitting the text.", type=int)
    parser.add_argument("-a","--algorithm", required=False, help="The name of the embedding algorithm to use.", default="nomic-embed-text")
    parser.add_argument("-o", "--overwrite",action="store_true", help="Overwrite existing collection if it exists.")
    args = parser.parse_args()
    get_chunks_from_input(args.input_filename, args.collection_prefix, args.chunk_length, args.algorithm, args.overwrite)
