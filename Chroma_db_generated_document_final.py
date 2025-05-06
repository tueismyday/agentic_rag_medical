from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
import os
import shutil
import uuid
import re
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableConfig
from typing import List, Tuple, Dict, Any

LLM = OllamaLLM(
    model="gemma3:4b-32k-it",
    temperature=0.6,
)

def split_by_section(text: str) -> List[Tuple[str, str]]:
    """
    Splits text into sections while keeping multi-page sections together under the last detected section.
    
    Detects section headers using the pattern "Overskrift:" and groups content under them.
    If no 'Overskrift:' lines are found, returns the full text as a single section.
    
    Args:
        text: The input text to be split into sections
        
    Returns:
        List[Tuple[str, str]]: A list of tuples containing (section_title, section_content)
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

def generate_summary(text: str, llm=LLM) -> str:
    """
    Generates a clinical summary of the provided text in Danish.
    
    Uses the configured LLM to create a concise, clinically relevant summary
    of the provided document text.
    
    Args:
        text: The document text to summarize
        llm: Language model to use for summarization (defaults to the global LLM)
        
    Returns:
        str: A clinical summary of the text in Danish
    """
    prompt = f"Sammenfat følgende dokument i en kort, klinisk relevant opsummering på dansk:\n\n{text}"
    response = LLM.invoke(
        prompt,
        config=RunnableConfig(
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    )
    
    return response

def load_and_process_pdfs(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load PDFs, extract structured sections, and split into smaller chunks.
    
    This function loads all PDFs from the specified directory, extracts structured sections
    if present, and splits the content into smaller chunks for vectorization. It processes
    documents differently based on whether they contain explicit section headers.
    
    Args:
        data_dir: Directory containing PDF files to process
        
    Returns:
        List[Dict[str, Any]]: A list of chunks with their text and associated metadata
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
    )

    all_chunks = []

    print(f"\nStarting vectorization process: \n - Making text-chunks for each section if found\n - Making paragraph chunks of each section \n - Generating a summary for each document\n")
    for doc_name, pages in doc_texts.items():
        doc_id = str(uuid.uuid4())  # Unique ID for the document

        # Join all pages into one string
        full_text = "\n".join(pages)
        structured_sections = split_by_section(full_text)
        
        is_full_doc = len(structured_sections) == 1 and structured_sections[0][0] == "Full Document"
        print(f"[DEBUG] {'No sections found' if is_full_doc else f'Found {len(structured_sections)} sections'} in document {doc_name} (ID: {doc_id})")

        if is_full_doc:
            # Only do paragraph-level splitting with generic metadata
            fine_chunks = text_splitter.split_text(full_text)
            for i, chunk_text in enumerate(fine_chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "document_id": doc_id,
                        "document_name": doc_name,
                        "section_title": "Unknown Section",
                        "chunk_index": i,
                        "granularity": "paragraph"
                    }
                })
        else:
            for section_title, section_text in structured_sections:
                # Store the section as one coarse chunk
                all_chunks.append({
                    "text": section_text,
                    "metadata": {
                        "document_id": doc_id,
                        "document_name": doc_name,
                        "section_title": section_title,
                        "chunk_index": 0,
                        "granularity": "section"
                    }
                })

                # Split into fine-grained chunks
                fine_chunks = text_splitter.split_text(section_text)
                for i, chunk_text in enumerate(fine_chunks):
                    all_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "document_id": doc_id,
                            "document_name": doc_name,
                            "section_title": section_title,
                            "chunk_index": i,
                            "granularity": "paragraph"
                        }
                    })

    return all_chunks

def display_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Display extracted chunks for debugging purposes.
    
    Prints information about each chunk including document name, section title,
    chunk index, and a preview of the text content.
    
    Args:
        chunks: List of chunk dictionaries to display
        
    Returns:
        None
    """
    print("\n[DEBUG] Displaying Extracted Chunks:\n")
    for chunk in chunks:
        print(f"Document: {chunk['metadata']['document_name']}")
        print(f"Section: {chunk['metadata']['section_title']}")
        print(f"Chunk Index: {chunk['metadata']['chunk_index']}")
        print(f"Text Preview: {chunk['text'][:100]}...")  # Print first 100 characters for readability
        print("-" * 80)

def create_vector_store(chunks: List[Dict[str, Any]], persist_directory: str) -> Chroma:
    """
    Create and persist ChromaDB vector store with enhanced metadata.
    
    Creates a new vector store from the provided chunks, using the 
    specified embedding model and persistence directory. If a vector store
    already exists at the given location, it will be cleared first.
    
    Args:
        chunks: List of text chunks with their metadata
        persist_directory: Directory where the vector store will be saved
        
    Returns:
        Chroma: The created vector database instance
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
    """
    Main function to execute the PDF processing and vectorization pipeline.
    
    Loads PDFs from the specified directory, processes them into chunks,
    and creates a vector store for later retrieval. The function prints
    progress and debugging information during execution.
    
    Returns:
        None
    """
    data_dir = os.path.join(os.path.dirname(__file__), "generated_documents")
    db_dir = os.path.join(os.path.dirname(__file__), "generated_documents_db")

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
