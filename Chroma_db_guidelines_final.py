from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import uuid
import re


def split_by_section(text: str) -> list[tuple[str, str]]:
    """Split text into sections based on 'Overskrift:' lines.
    
    Args:
        text: The document text to split.
        
    Returns:
        A list of tuples, each containing (section_title, section_content).
        If no 'Overskrift:' lines are found, returns a single tuple with 
        ("Full Document", full_text).
    """
    sections = []
    last_known_section = None  # Track last detected section title
    current_text = []
    found_section_headers = False

    lines = text.split("\n")
    for line in lines:
        line = line.strip()

        # Detect section headers using "Overskrift:"
        match = re.match(r"Overskrift:\s*'?(.*?)'?$", line)

        if match:
            found_section_headers = True  # At least one header was found

            # Store previous section if we already have content
            if current_text:
                sections.append((last_known_section, "\n".join(current_text)))

            last_known_section = match.group(1).strip()
            current_text = []  # Start a new section
        else:
            if last_known_section is None:
                last_known_section = "Unknown Section"
            current_text.append(line)

    if current_text:
        sections.append((last_known_section, "\n".join(current_text)))

    if not found_section_headers:
        print("[INFO] No 'Overskrift:' sections found — treating full document as one chunk.")
        return [("Full Document", text.strip())]

    return sections

def load_and_process_pdfs(data_dir: str) -> list[dict]:
    """Load PDFs and extract structured sections based only on 'Overskrift:' headers.
    
    Args:
        data_dir: Directory path containing PDF files to process.
        
    Returns:
        A list of dictionaries, each containing extracted section text and metadata.
        Each dictionary has 'text' and 'metadata' keys.
    """
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # Group all pages of the same document together
    doc_texts = {}
    
    for doc in documents:
        doc_name = os.path.basename(doc.metadata.get("source", "Unknown Document"))
        if doc_name not in doc_texts:
            doc_texts[doc_name] = []
        doc_texts[doc_name].append(doc.page_content.strip())

    all_chunks = []

    print(f"\nStarting vectorization process: \n - Creating chunks based only on 'Overskrift:' sections\n")
    for doc_name, pages in doc_texts.items():
        doc_id = str(uuid.uuid4())  # Unique ID for the document

        # Join all pages into one string
        full_text = "\n".join(pages)
        structured_sections = split_by_section(full_text)
        
        print(f"[DEBUG] Found {len(structured_sections)} sections in document {doc_name} (ID: {doc_id})")
        
        is_full_doc = len(structured_sections) == 1 and structured_sections[0][0] == "Full Document"
        print(f"[DEBUG] {'No sections found' if is_full_doc else f'Found {len(structured_sections)} sections'} in document {doc_name} (ID: {doc_id})")

        # Store each section as a separate chunk without further splitting
        for i, (section_title, section_text) in enumerate(structured_sections):
            all_chunks.append({
                "text": section_text,
                "metadata": {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "section_title": section_title,
                    "section_index": i,
                    "granularity": "section"
                }
            })

    return all_chunks

def display_chunks(chunks: list[dict]) -> None:
    """Display extracted chunks for debugging purposes.
    
    Args:
        chunks: List of chunk dictionaries containing text and metadata.
    """
    print("\n[DEBUG] Displaying Extracted Chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}/{len(chunks)}:")
        print(f"Document: {chunk['metadata']['document_name']}")
        print(f"Section: {chunk['metadata']['section_title']}")
        print(f"Section Index: {chunk['metadata']['section_index']}")
        print(f"Text Preview: {chunk['text'][:100]}...")  # Print first 100 characters for readability
        print("-" * 80)

def create_vector_store(chunks: list[dict], persist_directory: str) -> Chroma:
    """Create and persist ChromaDB vector store with section metadata.
    
    Args:
        chunks: List of chunk dictionaries containing text and metadata.
        persist_directory: Directory path to store the vector database.
        
    Returns:
        A Chroma vector store instance containing the embedded chunks.
    """
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    print("Creating new vector store...")
    vectordb = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    stored_chunks = vectordb.get()
    print(f"[DEBUG] Total stored chunks: {len(stored_chunks['documents'])}")
    return vectordb

def main() -> None:
    """Main function to process PDF documents and create vector database."""
    data_dir = os.path.join(os.path.dirname(__file__), "hospital_guidelines") 
    db_dir = os.path.join(os.path.dirname(__file__), "hospital_guidelines_db")

    print("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir)
    print(f"Created {len(chunks)} chunks from PDFs")
    
    # Debugging: Display chunks before storing them in vector DB
    display_chunks(chunks)
    
    print("Creating vector store...")
    vectordb = create_vector_store(chunks, db_dir)
    print(f"Vector store created and persisted at {db_dir}")

if __name__ == "__main__":
    main()