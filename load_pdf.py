import re
import ollama
import pymupdf4llm
import chromadb
import argparse

def _filter_empty_pages(pages):
    """
    Filter out empty pages from the list of pages.
    
    Args:
        pages (list): List of pages to filter.
    
    Returns:
        list: Filtered list of pages with non-empty text.
    """
    results = []
    for page in pages:
        if re.sub(r'\[\d+\]', '', page['text'].replace('\n','').replace('*','')):
            results.append(page)
    return results

def _split_pages_with_metadata(pages):
    """
    Split pages into chunks on subheaders identified by *** if chunks exceed a certain length
    """
    results = []
    metatadatas = []
    for page in pages:
        metadata = {k:page['metadata'][k] for k in page['metadata'] if page['metadata'][k]}
        subsections = re.split(r'(\*\*\*[^\*]*\*\*\*)', page['text'])
        prefix = ''
        for subsection in subsections:
            if subsection.startswith('***'):
                prefix+=subsection
                continue
            results.append(prefix+subsection)
            metatadatas.append(metadata)
            prefix = ''
    return results, metatadatas

def load_pdf(input_filename, collection_name):
    """
    Load a PDF file using PyMuPDF, process it, and store embeddings in ChromaDB.
    
    Args:
        input_filename (str): The name of the PDF file in the './data/' directory.
        collection_name (str): The name of the ChromaDB collection.
    """
    # Open the PDF file
    file_path = f'./data/{input_filename}'
    pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    pages = _filter_empty_pages(pages)
    chunks, metadata = _split_pages_with_metadata(pages)
    
    if not chunks:
        print(f"No processable content found in {input_filename} after filtering.")
        return

    embeds = ollama.embed(model="nomic-embed-text", input=chunks).get("embeddings",[])
    if not embeds:
        raise ValueError("No embeddings found in the response from Ollama.")
        
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collections = chroma_client.list_collections()
    if any(c.name == collection_name for c in collections):
        print(f"Deleting existing collection: {collection_name}")
        chroma_client.delete_collection(collection_name)
        
    collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space":"cosine"})
    ids = [f"{input_filename}_{i}" for i in range(len(embeds))]
    
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadata,
        embeddings=embeds
    )
    print(f"Successfully loaded {len(ids)} chunks from {input_filename} into collection {collection_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a PDF, process its content, and store embeddings in ChromaDB.")
    parser.add_argument("input_filename", help="The name of the input PDF file (located in ./data/ directory).")
    parser.add_argument("collection_name", help="The name of the ChromaDB collection to create/use.")
    
    args = parser.parse_args()
    
    try:
        load_pdf(args.input_filename, args.collection_name)
    except FileNotFoundError:
        print(f"Error: Input file ./data/{args.input_filename} not found.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

