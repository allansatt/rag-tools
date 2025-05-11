import json
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import ollama
import pymupdf4llm
import chromadb
import argparse

def _load_document_markdown(file_path, output_path, overwrite):
    if not overwrite and os.path.exists(output_path):
        with open(output_path, 'r') as f:
            document = f.read()
        return document

    document = pymupdf4llm.to_markdown(file_path, pages=range(17,332))
    with open(output_path, 'w') as f:
        f.write(document)
    return document

def _load_embeddings(embedings_path, algorithm, chunks, overwrite):
    if not overwrite and os.path.exists(embedings_path):
        with open(embedings_path, 'r') as f:
            document = json.load(f)
        return document
    embeddings = ollama.embed(model=algorithm, input=chunks).get("embeddings", [])
    with open(embedings_path, 'w') as f:
        json.dump(embeddings, f)
    return embeddings
 
def get_chunks_from_input(input_filename, collection_name, overwrite=False):
    """
    Load a PDF file using PyMuPDF, process it, and store embeddings in ChromaDB.
    
    Args:
        input_filename (str): The name of the PDF file in the './data/' directory.
        collection_name (str): The name of the ChromaDB collection.
    """
    # Open the PDF file
    file_path = f'./data/{input_filename}'
    output_path = f'./outputs/{collection_name}.md'
    embeddings_path = f'./embeddings/{collection_name}.json'

    # Process the PDF if overwrite is enabled or if the output directory doesn't exist (or is not a directory).
    markdown = _load_document_markdown(file_path, output_path, overwrite)
    # header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##",'h1')], strip_headers=False)
    # header_splits = header_splitter.split_text(markdown)
    character_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = character_splitter.split_text(markdown)
    metadata = [{'filepath':file_path} for _ in chunks]
    if not chunks:
        print(f"No processable content found in {input_filename} after PDF processing and filtering.")
        return # Exit if processing occurred but yielded no chunksz

    embeddings = _load_embeddings(embeddings_path, "nomic-embed-text", chunks, overwrite)
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collections = chroma_client.list_collections()
    if any(c.name == collection_name for c in collections):
        print(f"Deleting existing collection: {collection_name}")
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
    parser.add_argument("collection_name", help="The name of the ChromaDB collection to create/use.")
    parser.add_argument("-o", "--overwrite",action="store_true", help="Overwrite existing collection if it exists.")
    args = parser.parse_args()
    get_chunks_from_input(args.input_filename, args.collection_name, args.overwrite)
