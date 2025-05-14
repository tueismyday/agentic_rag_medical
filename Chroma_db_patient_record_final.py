from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
import uuid
import re
from datetime import datetime


def extract_date_from_text(text: str) -> str:
    """
    Extract the most recent date from text in DD.MM.YY format.
    
    Args:
        text: The text to search for dates
        
    Returns:
        str: The most recent date found in YYYY-MM-DD HH:MM format, or empty string if none found
    """
    matches = re.findall(r"\d{2}\.\d{2}\.\d{2}(?: \d{2}:\d{2})?", text)
    
    if not matches:
        return ""

    parsed_dates = []
    for m in matches:
        for fmt in ("%d.%m.%y %H:%M", "%d.%m.%y"):  # <--- DAY.MONTH.YEAR enforced
            try:
                dt = datetime.strptime(m, fmt)
                if dt.year > 2025:
                    continue
                parsed_dates.append(dt)
                break
            except ValueError:
                continue

    if not parsed_dates:
        return ""

    most_recent = max(parsed_dates)
    return most_recent.strftime("%Y-%m-%d %H:%M")

def infer_category(text: str) -> str:
    """
    Infer patient record category from Danish guideline keywords.
    
    Args:
        text: The text content to categorize
        
    Returns:
        str: The inferred category name, or "ukategoriseret" if no match found
    """
    
    keyword_map = {
    "indlæggelse": ["indlæggelse", "indlæggelsesårsag", "indlæggelsesforløb", "indlæggelsesdato"],
    "smitterisiko": ["smitterisiko", "isolering", "positiv dyrkning", "covid", "influenza"],
    "bolig": ["boligændring", "etager", "adgangsforhold"],
    "hjælpemidler": ["hjælpemiddel", "apv", "rollator", "badestol", "lift", "kørestol"],
    "genoptræning": ["genoptræning", "rehabilitering", "fysioterapi", "ergoterapi"],
    "hjemmesygepleje": ["hjemmesygepleje", "injektion", "sårpleje", "dauerbind"],
    "medicin_administration": ["medicinadministration", "dosering", "tablet", "øjendråber", "injektion"],
    "medicinindtagelse": ["medicinindtag", "indtager medicin", "knuses", "opløses", "slugning", "sonde"],
    "kommunal_ydelse": ["kommunal indsats", "adl vurdering", "rengøring", "indkøb", "tøjvask"],
    "udskrivningskonference": ["udskrivningskonference", "koordination", "kontakt primærsektor", "videomøde"],
    "bevægeapparat": ["bevægelse", "faldtendens", "muskelstyrke", "balance", "immobilitet", "forflytning", "mobilisering", "rehabilitering", "styrke"],
    "ernæring": ["ernæring", "væskeindtag", "kvalme", "sonde", "fejlsynkning", "kost", "diæt", "tørst", "appetit", "overvægt", "undervægt"],
    "hud": ["sår", "plaster", "trykaflastning", "hudproblem", "bandage", "eksem", "kløe", "rødme", "hævelse"],
    "kommunikation": ["kommunikation", "afasi", "dysartri", "tolk", "høreapparat", "synshandicap", "tale", "hørelse"],
    "psykosocialt": ["psykosocial", "tristhed", "angst", "depression", "demens", "forvirring", "kognitiv", "hukommelse", "socialt netværk", "livskvalitet"],
    "respiration": ["respiration", "dyspnø", "hoste", "ilt", "saturation", "pep-fløjte", "lunger", "bronkier", "krampe", "astma", "kol", "lungesygdom"],
    "cirkulation": ["cirkulation", "blodtryk", "hypertension", "hypotension", "ødem", "cyanose", "blodprop", "hjerte", "kredsløb", "hjerte-kar-sygdom"],
    "seksualitet": ["seksualitet", "erektion", "samlejesmerter", "samliv", "impotens", "seksuel dysfunktion", "libido", "prævention"],
    "smerter_sansning": ["smerte", "vas", "brændende", "stikkende", "borende", "følelse", "nedsat følelse", "smertebehandling", "analgetika", "smertestillende"],
    "søvn": ["søvn", "døgnrytme", "søvnproblemer", "mareridt", "udmattelse", "træthed", "insomni", "hypersomni"],
    "viden": ["sygdomsindsigt", "egenomsorg", "forståelse", "compliance", "medvirken", "information", "undervisning", "vejledning"],
    "udskillelse": ["vandladning", "afføring", "obstipation", "kateter", "urin", "ble", "mave", "tarm", "diarre", "inkontinens"],
    "funktionsevne": ["forflytning", "toiletbesøg", "gå", "drikke", "spise", "tøj", "vask", "funktionsevne", "adl", "selvstændighed"],
    }

    text_lower = text.lower()
    for category, keywords in keyword_map.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "ukategoriseret"

def split_by_date(text: str) -> list[tuple[str, str]]:
    """
    Splits the text into chunks at each detected timestamp.
    
    Args:
        text: The text content to split by dates
        
    Returns:
        list[tuple[str, str]]: A list of (date_string, chunk_text) pairs, 
                              or [("Unknown", text)] if no timestamps found
    """
    
    pattern = r"(\d{2}\.\d{2}\.\d{2}(?: \d{2}:\d{2})?)"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return [("Unknown", text.strip())]

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        date_str = matches[i].group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        chunks.append((date_str, chunk_text))

    return chunks

def extract_entry_type(text: str) -> str:
    """
    Extract the entry type from the first line after the timestamp.

    Example:
        From: "01.01.11 11:11 Medicinnotat, sygehusapotek\\n..."
        Extracts: "Medicinnotat, sygehusapotek"

    Args:
        text: The text chunk containing timestamp and header line

    Returns:
        str: The extracted entry type, or 'Note' if not found
    """
    pattern = r"(\\d{2}\\.\\d{2}\\.\\d{2}(?: \\d{2}:\\d{2})?)\\s+(.*)"
    lines = text.strip().splitlines()
    if not lines:
        return "Note"

    match = re.match(pattern, lines[0])
    if match:
        return match.group(2).strip()

    return "Note"

def load_and_process_pdfs(data_dir: str):
    """Load PDFs and split by date/timestamp"""
    
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    doc_texts = {}
    for doc in documents:
        doc_name = os.path.basename(doc.metadata.get("source", "Unknown Document"))
        if doc_name not in doc_texts:
            doc_texts[doc_name] = []
        doc_texts[doc_name].append(doc.page_content.strip())

    all_chunks = []

    print(f"\nStarting vectorization process: \n - Chunking by timestamps \n - Assigning category and metadata \n - Generating summaries\n")
    for doc_name, pages in doc_texts.items():
        doc_id = str(uuid.uuid4())
        full_text = "\n".join(pages)
        
        dated_chunks = split_by_date(full_text)
        print(f"[DEBUG] Found {len(dated_chunks)} timestamp chunks in document {doc_name} (ID: {doc_id})")
        
        for i, (date_str, chunk_text) in enumerate(dated_chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "chunk_index": i,
                    "granularity": "timestamp_chunk",
                    "category": infer_category(chunk_text),
                    "date": extract_date_from_text(date_str),
                    "entry_type": extract_entry_type(chunk_text)
                }
            })

    return all_chunks

def display_chunks(chunks: list[dict]) -> None:
    """
    Display extracted chunks for debugging purposes.
    
    Args:
        chunks: List of chunk dictionaries containing text and metadata
    """
    
    print("\n[DEBUG] Displaying Extracted Chunks:\n")
    for chunk in chunks:
        print(f"Document: {chunk['metadata']['document_name']}")
        print(f"Chunk Index: {chunk['metadata']['chunk_index']}")
        print(f"Date: {chunk['metadata']['date']}")
        print(f"Category: {chunk['metadata']['category']}")
        print(f"Text Preview: {chunk['text'][:200]}...")  # Display first 200 characters
        print("-" * 80)

def create_vector_store(chunks: list[dict], persist_directory: str) -> Chroma:
    """
    Create and persist ChromaDB vector store with enhanced metadata.
    
    Args:
        chunks: List of chunk dictionaries containing text and metadata
        persist_directory: Directory path where the vector store will be saved
        
    Returns:
        Chroma: The created Chroma vector database instance
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
    Main function to execute the PDF processing and vector store creation workflow.
    """
    
    data_dir = os.path.join(os.path.dirname(__file__), "patient_record")
    db_dir = os.path.join(os.path.dirname(__file__), "patient_record_db")

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
