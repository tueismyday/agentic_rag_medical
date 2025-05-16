# Imports
import time
import tracemalloc
import tiktoken
import re
import os
from PyPDF2 import PdfReader
from fpdf import FPDF
from langchain_ollama import OllamaLLM
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnableConfig
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationTokenBufferMemory
from langchain.tools import tool
from sentence_transformers import util, SentenceTransformer
from datetime import datetime

import Chroma_db_generated_document_final # Have this .py in your work folder



####################################################
# Loading the guideline vectordatabases and the LLMs
####################################################

# A modified OllamaLLM class that tracks token usage
class TokenCountedOllamaLLM(OllamaLLM):
    """
    OllamaLLM subclass that tracks token usage across invocations.
    This class extends OllamaLLM to add token counting for every invoke call,
    displaying usage statistics and warnings when approaching context limits.
    
        Methods:
            invoke(prompt, **kwargs): Override of invoke to count tokens before and after generation.
    """
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Override invoke to count tokens."""
        # Log the prompt tokens
        TOKEN_COUNTER.log_usage(prompt)
        
        # Call the original invoke method
        result = super().invoke(prompt, **kwargs)
        
        # Log the completion tokens
        TOKEN_COUNTER.log_usage("", result)
        
        # Print token usage stats
        stats = TOKEN_COUNTER.get_stats()
        print(f"\n[TOKEN USAGE] Call used approximately {TOKEN_COUNTER.count_tokens(prompt) + TOKEN_COUNTER.count_tokens(result)} tokens")
        print(f"[TOKEN USAGE] Total usage: {stats['total_tokens']} tokens")
        
        # Check if approaching token limits
        if stats["total_tokens"] > 16000:  # Adjust threshold based on your model's context window
            print(f"\n[WARNING] Token usage is high: {stats['total_tokens']} tokens")
        
        return result

# Loading LLMs with token count
LLM_RETRIEVE = TokenCountedOllamaLLM(
    model="gemma3:12b-32k",
    temperature=0.6,
)

LLM_GENERATE = TokenCountedOllamaLLM(
    model="gemma3:12b-32k",
    temperature=0.6,
)

LLM_CRITIQUE = TokenCountedOllamaLLM(
    model="gemma3:12b-32k",
    temperature=0.1,
)

####################################################
# Defining local attributes:
####################################################

# Define the directories for the vector databases
GUIDELINE_DB_DIR = os.path.join(os.path.dirname(__file__), "hospital_guidelines_db")
PATIENT_DB_DIR = os.path.join(os.path.dirname(__file__), "patient_record_db")
GENERATED_DOCS_DB_DIR = os.path.join(os.path.dirname(__file__), "generated_documents_db")
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2",
                                   model_kwargs={'device': 'cpu'})

# Load ChromaDB vectordatabases
PATIENT_VECTOR_DB = Chroma(persist_directory=PATIENT_DB_DIR, embedding_function=EMBEDDINGS)
GUILDELINE_VECTOR_DB = Chroma(persist_directory=GUIDELINE_DB_DIR, embedding_function=EMBEDDINGS)
GENERATED_DOCS_VECTOR_DB = Chroma(persist_directory=GENERATED_DOCS_DB_DIR, embedding_function=EMBEDDINGS)

# Print the number of document chunks in each vector database for debugging
print(f"\n\n[INFO] The 'patient record' vector database currently holds {PATIENT_VECTOR_DB._collection.count()} document chunks.")
print(f"[INFO] The 'guideline' vector database currently holds {GUILDELINE_VECTOR_DB._collection.count()} document chunks")
if GENERATED_DOCS_VECTOR_DB != None:
    print(f"[INFO] The 'generated documents' vector database currently holds {GENERATED_DOCS_VECTOR_DB._collection.count()} document chunks\n\n")
else:
    print(f"[INFO] The 'generated documents' vector database currently holds 0 document chunks\n\n")

# Define a global variable for the retrieved guideline sections:
RETRIEVED_GUIDELINES = None
# Define a global variable for the retrieval memory
RETRIEVAL_MEMORY = ConversationTokenBufferMemory(
        llm=LLM_RETRIEVE, 
        memory_key="chat_history",
        max_token_limit=16000,
        return_messages=True
    )

####################################################
# Token-counter layer for debugging and usage tracking
####################################################

class TokenCounter:
    """"
    Token counter class to track token usage across multiple LLM calls.
    The class maintains counts for prompt tokens, completion tokens, and total tokens,
    supporting both tiktoken-based precise counting and word-based approximation.
        Methods:
            count_tokens(text): Count tokens in a string.
            log_usage(prompt_text, completion_text): Log usage for a single call.
            get_stats(): Get current usage statistics.
            reset(): Reset all counters.
    """
    
    def __init__(self):
        """Initialize with a model name to determine tokenizer.
        
        For Ollama models, we'll use tiktoken's cl100k_base encoder as an approximation.
        """
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.calls = 0
        
        # Use cl100k_base as a reasonable approximation for most models
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback to simple word-based estimation if tiktoken is not available
            self.encoding = None
            print("[WARNING] tiktoken not available, using basic word count approximation")
    
    def count_tokens(self, text):
        """Count tokens in a string."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Simple word-based approximation (~0.75 tokens per word)
            return int(len(re.findall(r'\w+', text)) * 1.33)
    
    def log_usage(self, prompt_text, completion_text=None):
        """Log usage for a single call."""
        prompt_tokens = self.count_tokens(prompt_text)
        self.total_prompt_tokens += prompt_tokens
        
        if completion_text:
            completion_tokens = self.count_tokens(completion_text)
            self.total_completion_tokens += completion_tokens
        else:
            completion_tokens = 0
            
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.calls += 1
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    
    def get_stats(self):
        """Get current usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "calls": self.calls
        }
    
    def reset(self):
        """Reset all counters."""
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.calls = 0

# Initialize a global token counter
TOKEN_COUNTER = TokenCounter()

####################################################
# Time measure wrapper for debugging usage tracking
####################################################

# This decorator can be used to profile the execution time of functions.
def profile(func):
    """
    Decorator function that measures and reports execution time of the wrapped function.
        Args:
            func: The function to be profiled.
        Returns:
            wrapper: The wrapped function with profiling capabilities.
    """
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        tracemalloc.stop()
        print(f"\n\n[PROFILE] {func.__name__} took {end_time - start_time:.2f}s\n\n")
        return result
    return wrapper


####################################################
# Information extraction methods and tools 
####################################################

def generate_retrieval_query(section_title:str, subsection_title:str, subsection_guidelines:str) -> str:
    """
    Generates an optimized query for retrieving relevant patient data during RAG searches in the generation loop.
    
    Args:
        section_title: The title of the parent section
        subsection_title: The title of the subsection
        subsection_guidelines: The guidelines for this specific subsection
        
    Returns:
        String containing the optimized query
    """
    prompt = f"""
    Du skal hente information til {subsection_title} i sektionen {section_title} for en patient, der er indlagt på hospitalet.  
    Baseret på hospitalets retningslinjer skal du fokusere på følgende punkter:

    {subsection_guidelines}

    Din opgave:  
    - Formulér én præcis og målrettet søgeforespørgsel (på dansk), der kan bruges til at finde de nødvendige informationer i patientjournalen ved hjælp af RAG.  
    - Brug terminologi fra retningslinjerne, hvor det er relevant.  
    - Hvis retningslinjerne antyder en tidsramme (f.eks. nylige hændelser), så indarbejd dette i din forespørgsel.  
    - Hvis emnet er bredt, bør du inkludere relevante synonymer eller relaterede kliniske termer.
    """
    
    query = LLM_RETRIEVE.invoke(
        prompt,
        config=RunnableConfig(
            callbacks=[StreamingStdOutCallbackHandler()],
            tags=["rag_query_generation"]
        )
    )
    
    # Clean up the query (remove quotes if present, etc.)
    query = query.strip().strip('"\'')
    
    print(f"[RAG] Generated retrieval query: {query}")
    return query

def retrieve_guidelines_by_section(query:str) -> dict:
    """
    First finds the most relevant document in the vector database, then retrieves all its sections while skipping 'Unknown Section'.
    Used as a helper function for 'start_document_generation' gathering all sections of the guideline before proceeding to generate it section by section.
    
    Args:
        query (str): The query to search for relevant guidelines.
    
    Returns:
        dict: Dictionary of guideline sections with section titles as keys and their content as values.
        Returns an error message string if no guidelines are found.
    """
    # Find the most relevant document
    print(f"[INFO] Searching for the relevant guideline.")
    doc_results = GUILDELINE_VECTOR_DB.similarity_search(query, k=3)
    if not doc_results:
        return "No relevant guidelines found."
    
    best_doc = doc_results[0].metadata["document_name"]  # Identify the most relevant document
    print(f"[DEBUG] Identified most relevant guideline document: {best_doc}")
    
    # Retrieve all sections for the identified document
    section_results = GUILDELINE_VECTOR_DB.get(where={"document_name": best_doc})
    
    if not section_results["documents"]:
        return "No relevant sections found in the selected guideline document."
    
    # Organize sections by title while skipping 'Unknown Section'
    # Unknown sections are defined in the vectordatabase logic as sections without a headline
    print(f"[INFO] Fetching all sections from guideline: {best_doc}")
    sections = {}
    for chunk_text, metadata in zip(section_results["documents"], section_results["metadatas"]):
        section_title = metadata.get("section_title", "Unknown Section")
        
        if section_title == "Unknown Section":
            print(f"[WARNING] Skipping 'Unknown Section' in {best_doc}.")
            continue  # Skip processing this section
        
        if section_title not in sections:
            sections[section_title] = ""
        sections[section_title] += chunk_text + "\n"

    # **DEBUG OUTPUT: Verify the final structure**
    print(f"[DEBUG] Retrieved Sections: {sections.keys()}")
    print("[RESULT] All sections has been gathered.")
    return sections

def split_section_into_subsections(section_text:str) -> tuple:
    """
    Splits section text into subsections based on 'Sub_section' markers.
    Returns intro text and a list of (subsection_title, subsection_content) tuples.
    """
    # First, check if there are any subsections
    if "Sub_section" not in section_text:
        # No subsections found, return the entire section as one
        return section_text, [("Main Content", section_text)]
    
    # Split the text by "Sub_section" markers
    parts = re.split(r'(Sub_section[^:]*:)', section_text)
    
    # The first part (before any Sub_section) is the introduction
    intro_text = parts[0].strip()
    
    subsections = []
    for i in range(1, len(parts), 2):
        # Check if we have both the subsection header and content
        if i < len(parts) - 1:
            subsection_title = parts[i].strip()
            subsection_content = parts[i + 1].strip()
            
            # Extract the actual title from "Sub_section: title"
            title_match = re.match(r'Sub_section[^:]*:\s*(.+)', subsection_title)
            if title_match:
                clean_title = title_match.group(1).strip()
            else:
                clean_title = f"Subsection {i//2 + 1}"
                
            subsections.append((clean_title, subsection_content))
    
    # If no subsections were successfully extracted, treat the whole text as one subsection
    if not subsections:
        subsections = [("Main Content", section_text)]
    
    return intro_text, subsections

@tool
def start_document_generation(query: str) -> str:
    """
    Initiates the generation of a new document by retrieving relevant guideline sections based on the input query.

    This tool searches for the most relevant guideline document, extracts its structured content (excluding sections
    titled 'Unknown Section'), and stores the result in a global variable (`RETRIEVED_GUIDELINES`) for future reference.

    Args:
        query (str): A user-provided prompt or question that describes what kind of document is needed 
                     (e.g., "Skriv en plejeforløbsplan for denne patient").

    Returns:
        str: A confirmation message indicating that the document has been created and is available for further queries.
    """
    global RETRIEVED_GUIDELINES
    RETRIEVED_GUIDELINES = retrieve_guidelines_by_section(query)
    return """The requested document has been created and is now avaiable for search with the tool 'retrieve_generated_document_info'. 
            Tell the human that the document has been created and saved, and that you now are ready to answer additional questions about the document."""

@tool
def retrieve_patient_info(query: str, initial_k: int = 20, final_k: int = 15) -> str:
    """
    Retrieve relevant patient information from a vectorized journal using a tiered hybrid RAG approach.

    The function performs semantic vector search with similarity scoring, filters out low-relevance matches,
    and reranks results based on both semantic closeness and clinical category alignment. It then limits
    results to only the most relevant documents. If temporal expressions are detected in the query, 
    results are additionally sorted by date to prioritize recent entries.

    Returns a structured summary of the most relevant journal content for clinical use.

    Args:
        query (str): A natural language query about the patient's condition, treatment, or care needs.
        initial_k (int): Number of documents to retrieve in the initial search (default: 20).
        final_k (int): Maximum number of documents to include after reranking (default: 15).

    Returns:
        str: A Markdown-formatted summary of high-relevance journal content, grouped and optionally time-ordered.
    """
    
    # Adjustment of the functionalities:
    chronological_rerank = True
    query_categories_rerank = True
    
    
    # Load embedding model and cache for reuse
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    # Cache the query embedding so we don't need to recompute it when reranking
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    
    # Initial semantic search with score filtering - retrieve more documents initially
    print(f"[INFO] Performing initial semantic search with k={initial_k}...")
    results_with_scores = PATIENT_VECTOR_DB.similarity_search_with_score(query, k=initial_k)

    # Define a threshold for filtering results
    score_threshold = 0.3

    filtered_results = [doc for doc, score in results_with_scores if score >= score_threshold]

    if not filtered_results:
        return "No matching patient information found for the query."
    
    # Cache document embeddings during filtering to avoid recomputation during reranking
    doc_embeddings = {}
    for doc in filtered_results:
        content = doc.page_content.lower()
        doc_embeddings[id(doc)] = embedding_model.encode(content, convert_to_tensor=True)

    # Define Danish keywords for categories
    query_categories = {
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


    def parse_date_safe(date_str):
        for fmt in ("%y.%m.%d %H:%M", "%y.%m.%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return datetime.min

    # Relevance and category-based reranking - now using cached embeddings
    def rank_score(doc):
        content = doc.page_content.lower()
        content_embedding = doc_embeddings[id(doc)]
        
        # Base similarity score
        score = util.cos_sim(query_embedding, content_embedding)[0][0].item()
        
        # Add bonus for category match
        if query_categories_rerank:
            category = doc.metadata.get("category", "").lower()
            for cat, keywords in query_categories.items():
                if cat in category or any(kw in content for kw in keywords):
                    score += 0.3
        
        # Add bonus for recency if the query has temporal keywords
        if chronological_rerank:
            date_str = doc.metadata.get("date", "")
            try:
                doc_date = parse_date_safe(date_str)
                days_ago = (datetime.now() - doc_date).days
                # Linear decay: more recent = higher bonus, older = lower
                recency_bonus = max(0, 0.3 - (days_ago / 365.0) * 0.3)  # Max +0.3 if today, 0 if older than 1 year
                score += recency_bonus
            except:
                pass  # Skip if no valid date

        return score


    print(f"[INFO] Reranking {len(filtered_results)} documents...")
    reranked_results = sorted(filtered_results, key=rank_score, reverse=True)
    print(f"[INFO] Limiting to top {final_k} documents after reranking...")
    limited_results = reranked_results[:final_k]
    print(f"[INFO] Selected {len(limited_results)} documents out of {len(filtered_results)} filtered documents")

    query_lower = query.lower()
    target_categories = [cat for cat, kw in query_categories.items() if any(k in query_lower for k in kw)]


    # Identify query targets
    query_lower = query.lower()
    target_categories = [cat for cat, kw in query_categories.items()
                        if any(k in query_lower for k in kw)]

    # Format per-chunk output
    lines = ["# Patientoplysninger\n"]
    if target_categories:
        lines.append(f"Filtreret efter kategorier: {', '.join(target_categories)}\n")

    lines.append(f"*Søgning fandt {len(filtered_results)} relevante dokumenter og præsenterer de {len(limited_results)} mest relevante.*\n")

    scores = [rank_score(doc) for doc in limited_results]
    max_score = max(scores) if scores else 1
    min_score = min(scores) if scores else 0
    score_range = max(max_score - min_score, 0.001)

    for doc in limited_results:
        entry_type = doc.metadata.get("entry_type", "Note")
        date_str = doc.metadata.get("date", "") or "Ukendt dato"
        category = doc.metadata.get("category", "Ukategoriseret")
        content = doc.page_content.strip()

        raw_score = rank_score(doc)
        relevance = int((raw_score - min_score) / score_range * 100)

        lines.append(f"## {entry_type} ({date_str})")
        meta_summary = [f"**Relevans:** {relevance}% match til forespørgsel"]

        try:
            doc_date = parse_date_safe(date_str)
            now = datetime.now()
            days_ago = (now - doc_date).days
            if days_ago < 7:
                meta_summary.append("**Periode:** Meget nylig (<1 uge)")
            elif days_ago < 30:
                meta_summary.append("**Periode:** Nylig (<1 måned)")
            elif days_ago < 180:
                meta_summary.append("**Periode:** Inden for 6 måneder")
            elif days_ago < 365:
                meta_summary.append("**Periode:** Inden for 1 år")
            else:
                meta_summary.append(f"**Periode:** {days_ago // 365} år gammel")
        except:
            meta_summary.append(f"**Dato:** {date_str}")

        meta_summary.append(f"**Kategori:** {category}")

        content_length = len(content)
        if content_length < 200:
            meta_summary.append("**Omfang:** Kort note")
        elif content_length < 500:
            meta_summary.append("**Omfang:** Mellemlang note")
        else:
            meta_summary.append("**Omfang:** Detaljeret dokumentation")

        lines.append(f"*{' | '.join(meta_summary)}*\n")
        lines.append(content)
        lines.append("\n---\n")

    lines.append("**Tip**: Brug specifikke termer som 'medicin', 'blodprøver' eller 'operationer' for mere målrettede resultater.")
    return "\n".join(lines)


@tool
def retrieve_generated_document_info(query: str) -> str:
    """
    Retrieve relevant content from previously generated medical documents based on a user query.

    This tool performs semantic similarity search across vectorized final documents, filters out
    low-relevance matches using a similarity score threshold, and groups high-quality results
    by section and document origin. If specific sections or document names are mentioned in the
    query, the output is filtered accordingly. The final result presents a structured overview
    of matched content and a listing of available sections for further exploration.

    Args:
        query (str): A natural language query concerning information in the generated medical document.

    Returns:
        str: A structured Markdown-formatted overview of relevant document content,
            filtered by semantic score and organized for clinical interpretation.
    """
    # Check if final document exists
    if {GENERATED_DOCS_VECTOR_DB._collection.count()} == 0 or None:
        return "No final document has been generated yet. Please use the 'start_document_generation' tool to create a medical document first."
    
    # Perform similarity search with metadata filter capability
    # Try multiple granularities and include summaries
    # Perform similarity search with scores
    SCORE_THRESHOLD = 0.3

    paragraph_raw = GENERATED_DOCS_VECTOR_DB.similarity_search_with_score(query, k=10, filter={"granularity": "paragraph"})
    section_raw = GENERATED_DOCS_VECTOR_DB.similarity_search_with_score(query, k=10, filter={"granularity": "section"})
    summary_raw = GENERATED_DOCS_VECTOR_DB.similarity_search_with_score(query, k=1, filter={"section_title": "Document Summary"})

    # Filter out low-scoring matches
    paragraph_results = [doc for doc, score in paragraph_raw if score >= SCORE_THRESHOLD]
    section_results = [doc for doc, score in section_raw if score >= SCORE_THRESHOLD]
    summary_results = [doc for doc, score in summary_raw if score >= SCORE_THRESHOLD]
    
    # Merge all 
    results = []
    for res in paragraph_results + section_results + summary_results:
            results.append(res)

    
    if not results:
        return "No relevant document found for this query."

    
    # Try to identify specific sections or documents mentioned in the query
    query_lower = query.lower()
    likely_section = None
    likely_document = None
    
    # Extract all unique sections and document names for analysis
    all_sections = set()
    all_documents = set()
    
    # Track all sections by document for the complete listing
    document_sections = {}

    for doc in results: # extract metadata for later filtering and display
        section = doc.metadata.get("section_title", "").lower()
        doc_name = doc.metadata.get("document_name", "").lower()
        doc_id = doc.metadata.get("document_id", "")
        
        if section:
            all_sections.add(section)
        if doc_name:
            all_documents.add(doc_name)
        
        # Build a mapping of document_id -> [section_titles]
        if doc_id not in document_sections:
            document_sections[doc_id] = {
                "name": doc.metadata.get("document_name", "Unknown Document"),
                "sections": set()
            }
        if section:
            document_sections[doc_id]["sections"].add(section)
    
    # Check if query contains any specific section or document references
    likely_sections = [section for section in all_sections if section in query_lower]
    likely_documents = [doc_name for doc_name in all_documents if doc_name.replace(".pdf", "") in query_lower]

    
    # Group results by document and section
    grouped = {}
    for doc in results:
        doc_name = doc.metadata.get("document_name", "Unknown Document")
        section = doc.metadata.get("section_title", "Unknown Section").strip()
        doc_id = doc.metadata.get("document_id", "")
        chunk_index = doc.metadata.get("chunk_index", 0)
        
        # Apply filters if specific section or document was detected
        if (likely_sections and section.lower() not in likely_sections) or (likely_documents and doc_name.lower() not in likely_documents):
            continue
            
        key = f"{doc_name} | {section}"
        if key not in grouped:
            grouped[key] = {
                "chunks": [],
                "doc_id": doc_id,
                "indices": [],
                "content": []
            }
        
        grouped[key]["chunks"].append({
            "index": chunk_index,
            "content": doc.page_content
        })
        grouped[key]["indices"].append(chunk_index)
        grouped[key]["content"].append(doc.page_content)
    
    if not grouped:
        return "No relevant document sections matched your specific query criteria."
    
    # Sort chunks within each section by their index to maintain document order
    for key in grouped:
        grouped[key]["chunks"].sort(key=lambda x: x["index"])
        grouped[key]["content"] = [chunk["content"] for chunk in grouped[key]["chunks"]]
    
    # Build formatted output
    output_lines = ["# Generated Document Search Results\n"]
    
    # Add filter information if applicable
    filters_applied = []

    if likely_sections:
        filters_applied.append(f"Sections: {', '.join([f'‘{s}’' for s in likely_sections])}")
    if likely_documents:
        filters_applied.append(f"Documents: {', '.join([f'‘{d.replace('.pdf', '')}’' for d in likely_documents])}")

    
    # Sort groups by document name for consistent output
    sorted_keys = sorted(grouped.keys())
    
    # Track which documents we've processed for the content sections
    processed_doc_ids = set()
    
    for key in sorted_keys:
        doc_info = grouped[key]
        doc_id = doc_info["doc_id"]
        processed_doc_ids.add(doc_id)
        
        # Extract document and section names
        doc_name, section = key.split(" | ")
        
        # Create section header with document reference
        output_lines.append(f"## {section}")
        output_lines.append(f"*From document: {doc_name}*\n")
        
        # Join content with paragraph breaks for readability
        section_content = "\n\n".join(doc_info["content"])
        output_lines.append(section_content)
        output_lines.append("\n---\n")
    
    # Add section listing for each document found
    output_lines.append("\n# Available Sections by Document\n")
    output_lines.append("The following sections are available in the documents matched by your query:\n")
    
    for doc_id, doc_data in document_sections.items():
        # Highlight which sections were included vs. not included in results
        doc_name = doc_data["name"]
        output_lines.append(f"## {doc_name}\n")
        
        if len(doc_data["sections"]) > 0:
            sections_list = sorted(doc_data["sections"])
            for section in sections_list:
                # Check if this section was included in the results
                section_key = f"{doc_name} | {section}"
                if section_key in grouped:
                    output_lines.append(f"+ {section} (included in results)")
                else:
                        output_lines.append(f"? {section} (not included in results)")
        else:
            output_lines.append("- No section information available for this document")
        
        output_lines.append("")
    
    # Add usage tips
    output_lines.append("\n**Tips:**")
    output_lines.append("- For more specific results, mention the section or document name in your query")
    output_lines.append("- To see content from sections marked '?', try rephrasing your query to include those section names")
    output_lines.append("- For full document content, include the document name in your query without specific section terms")
    
    return "\n".join(output_lines)

@tool
def retrieve_guideline_knowledge(query: str) -> str:
    """
    Retrieve relevant sections from official hospital guidelines based on a natural language query.

    This tool performs semantic search with similarity scoring to identify and rank relevant content
    from vectorized hospital guideline documents. Low-relevance results are filtered out using a
    similarity threshold. Matching content is grouped by section and document origin, and further
    filtered when specific terms (e.g., section titles or document names) appear in the query.
    The final result includes structured, context-aware content along with a listing of available
    sections for further exploration.

    Args:
        query (str): A natural language query related to clinical guidelines or procedures.

    Returns:
        str: A Markdown-formatted summary of guideline content, filtered by semantic relevance
            and organized by section and document for clinical reference.
    """
    # Perform similarity search with metadata filter capability
    # Try multiple granularities and include summaries
    # Perform similarity search with scores
    SCORE_THRESHOLD = 0.3

    paragraph_raw = GENERATED_DOCS_VECTOR_DB.similarity_search_with_score(query, k=10, filter={"granularity": "paragraph"})
    section_raw = GENERATED_DOCS_VECTOR_DB.similarity_search_with_score(query, k=10, filter={"granularity": "section"})
    summary_raw = GENERATED_DOCS_VECTOR_DB.similarity_search_with_score(query, k=1, filter={"section_title": "Document Summary"})

    # Filter out low-scoring matches
    paragraph_results = [doc for doc, score in paragraph_raw if score >= SCORE_THRESHOLD]
    section_results = [doc for doc, score in section_raw if score >= SCORE_THRESHOLD]
    summary_results = [doc for doc, score in summary_raw if score >= SCORE_THRESHOLD]
    
    # Merge all 
    results = []
    for res in paragraph_results + section_results + summary_results:
        results.append(res)

    if not results:
        return "No relevant document found for this query."
    
    # Try to identify specific sections or documents mentioned in the query
    query_lower = query.lower()
    likely_section = None
    likely_document = None
    
    # Extract all unique sections and document names for analysis
    all_sections = set()
    all_documents = set()
    
    # Track all sections by document for the complete listing
    document_sections = {}

    for doc in results: # extract metadata for later filtering and display
        section = doc.metadata.get("section_title", "").lower()
        doc_name = doc.metadata.get("document_name", "").lower()
        doc_id = doc.metadata.get("document_id", "")
        
        if section:
            all_sections.add(section)
        if doc_name:
            all_documents.add(doc_name)
        
        # Build a mapping of document_id -> [section_titles]
        if doc_id not in document_sections:
            document_sections[doc_id] = {
                "name": doc.metadata.get("document_name", "Unknown Document"),
                "sections": set()
            }
        if section:
            document_sections[doc_id]["sections"].add(section)
    
    # Check if query contains any specific section or document references
    likely_sections = [section for section in all_sections if section in query_lower]
    likely_documents = [doc_name for doc_name in all_documents if doc_name.replace(".pdf", "") in query_lower]

    
    # Group results by document and section
    grouped = {}
    for doc in results:
        doc_name = doc.metadata.get("document_name", "Unknown Document")
        section = doc.metadata.get("section_title", "Unknown Section").strip()
        doc_id = doc.metadata.get("document_id", "")
        chunk_index = doc.metadata.get("chunk_index", 0)
        
        # Apply filters if specific section or document was detected
        if (likely_sections and section.lower() not in likely_sections) or (likely_documents and doc_name.lower() not in likely_documents):
            continue
            
        key = f"{doc_name} | {section}"
        if key not in grouped:
            grouped[key] = {
                "chunks": [],
                "doc_id": doc_id,
                "indices": [],
                "content": []
            }
        
        grouped[key]["chunks"].append({
            "index": chunk_index,
            "content": doc.page_content
        })
        grouped[key]["indices"].append(chunk_index)
        grouped[key]["content"].append(doc.page_content)
    
    if not grouped:
        return "No relevant document sections matched your specific query criteria."
    
    # Sort chunks within each section by their index to maintain document order
    for key in grouped:
        grouped[key]["chunks"].sort(key=lambda x: x["index"])
        grouped[key]["content"] = [chunk["content"] for chunk in grouped[key]["chunks"]]
    
    # Build formatted output
    output_lines = ["# Generated Document Search Results\n"]
    
    # Add filter information if applicable
    filters_applied = []

    if likely_sections:
        filters_applied.append(f"Sections: {', '.join([f'‘{s}’' for s in likely_sections])}")
    if likely_documents:
        filters_applied.append(f"Documents: {', '.join([f'‘{d.replace('.pdf', '')}’' for d in likely_documents])}")

    
    # Sort groups by document name for consistent output
    sorted_keys = sorted(grouped.keys())
    
    # Track which documents we've processed for the content sections
    processed_doc_ids = set()
    
    for key in sorted_keys:
        doc_info = grouped[key]
        doc_id = doc_info["doc_id"]
        processed_doc_ids.add(doc_id)
        
        # Extract document and section names
        doc_name, section = key.split(" | ")
        
        # Create section header with document reference
        output_lines.append(f"## {section}")
        output_lines.append(f"*From document: {doc_name}*\n")
        
        # Join content with paragraph breaks for readability
        section_content = "\n\n".join(doc_info["content"])
        output_lines.append(section_content)
        output_lines.append("\n---\n")
    
    # Add section listing for each document found
    output_lines.append("\n# Available Sections by Document\n")
    output_lines.append("The following sections are available in the documents matched by your query:\n")
    
    for doc_id, doc_data in document_sections.items():
        # Highlight which sections were included vs. not included in results
        doc_name = doc_data["name"]
        output_lines.append(f"## {doc_name}\n")
        
        if len(doc_data["sections"]) > 0:
            sections_list = sorted(doc_data["sections"])
            for section in sections_list:
                # Check if this section was included in the results
                section_key = f"{doc_name} | {section}"
                if section_key in grouped:
                    output_lines.append(f"+ {section} (included in results)")
                else:
                        output_lines.append(f"? {section} (not included in results)")
        else:
            output_lines.append("- No section information available for this document")
        
        output_lines.append("")
    
    # Add usage tips
    output_lines.append("\n**Tips:**")
    output_lines.append("- For more specific results, mention the section or document name in your query")
    output_lines.append("- To see content from sections marked '?', try rephrasing your query to include those section names")
    output_lines.append("- For full document content, include the document name in your query without specific section terms")
    
    return "\n".join(output_lines)
    
    return "\n".join(output_lines)


####################################################
# Defining the retrieval agent
####################################################

# Retrieval decision agent
def create_retrieval_agent():
    """
    Creates a retrieval agent with appropriate tools, memory, and configuration.
    The agent uses a conversational memory buffer and multiple specialized tools
    to retrieve and process information from guidelines, patient records, and
    generated documents.
    
        Returns:
            AgentExecutor: An initialized agent executor ready to process queries.
    """
    global RETRIEVAL_MEMORY
    
    # Load memory content
    memory_variables = RETRIEVAL_MEMORY.load_memory_variables({})
    chat_history = memory_variables.get("chat_history", [])
    
    # Limit chat history to the last 10 conversations
    chat_history = chat_history[-20:]
    
    # Format chat history for inclusion in the prompt
    formatted_history = ""
    if chat_history:
        for message in chat_history:
            if hasattr(message, "content") and hasattr(message, "type"):
                role = "Human" if message.type == "human" else "Assistant"
                formatted_history += f"{role}: {message.content}\n"
    
    # Initialize tools for the agent
    tools = [retrieve_guideline_knowledge, retrieve_patient_info, retrieve_generated_document_info, start_document_generation]
    memory = RETRIEVAL_MEMORY
    
    agent_executor = initialize_agent(
    tools=tools,
    llm=LLM_RETRIEVE,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    agent_kwargs={
        "prefix": f"""You are an intelligent multi-step retrieval (RAG) agent working in a hospital. 
        You are expected to think aloud, reason step-by-step, and use multiple tools if needed to fully understand and answer the user's question. 
        
        Recent conversation history:
        {formatted_history}
    
        Please refer to this conversation history when answering questions that reference previous interactions.
        Your goal is to simulate how an expert clinician-researcher would retrieve and synthesize information from multiple sources to give a clear, accurate answer — in Danish."""
,
        
        "system_message": """
        You are an intelligent hospital retrieval agent.

        Rules:
            - Use the 'start_document_generation' tool if the user asks to create a document
            - Never repeat the same tool with the same input more than once.
            - Stop when you have enough information.
        
        Available tools:
            - retrieve_patient_info: Access patient records
            - retrieve_generated_document_info: Get information about finalized documents
            - start_document_generation: Begin creating a clinical document
            - retrieve_guideline_knowledge: Get information about medical guidelines
        
        Think step-by-step:
            - First, analyze the user's question in detail.
            - Then hypothesize what information you may need to answer it.
            - Then decide which tool is best for retrieving that information.
            - After each tool, reflect: did this bring you closer to the answer? Should you expand, follow up, or rerank?
            - You are allowed to perform multiple tool actions in sequence before finalizing your answer, especially when the question requires combining patient-specific and guideline-based knowledge.
            - Before you finalize, make sure to check if the information you have is sufficient to answer the question. If any information is redundant or irrelevant, you should remove it.
        """,
        
        "format_instructions": """You must use this format for tool usage:

        IMPORTANT: To use a tool: (In the following, replace 'curly braces' with the curly braces symbols)
```json
curly braces
"action": "tool_name",
"action_input": "input string"
curly braces
```


        IMPORTANT: When you are done: (In the following, replace 'curly braces' with the curly braces symbols)
```json
curly braces
"action": "Final Answer",
"action_input": "your full answer in Danish"
curly braces
```

""" }, 
    handle_parsing_errors=True, 
    verbose=True,
    max_iterations=10,
    early_stopping_method="generate"
    )
    
    return agent_executor

@profile
def invoke_retrieval_agent(query: str):
    """
    Uses an LLM agent to collect information for answering the user query aswell as generating sub-sections of the document.
    This function is profiled for performance monitoring.
        Args:
            query (str): The user's query to be processed by the agent.
        Returns:
            str: The agent's response to the query.
    """
    agent = create_retrieval_agent()
    
    try:
        print(f"[DEBUG] Invoking retrieval agent with query: {query}")
        response = agent.invoke(
        input={"input": query},
        config=RunnableConfig(
            callbacks=[StreamingStdOutCallbackHandler()])
        )

        # Check if the response is a dictionary or string
        if isinstance(response, dict) and 'output' in response:
            return response['output']
        elif isinstance(response, str):
            return response
        else:
            return "Unexpected agent response format."
    except Exception as e:
        return f"Retrieval agent failed: {str(e)}"


####################################################
# Agent for generation and its critique module
####################################################

def critique_section_guideline(section_title: str, generated_text: str, section_guidelines: str) -> str:
    """
    Evaluates the generated text against hospital guidelines and provides a critique.
    
    Args:
        section_title: The title of the section being evaluated.
        generated_text: The text content of the section to be critiqued.
        section_guidelines: The hospital guidelines that the section should follow.
    
    Returns:
        The critique result as a string.
    """
    critique_prompt = f"""
    Du har til opgave at give kritik på en genereret sektion '{section_title}' i et medicinsk dokument.
    
    Du ved KUN hvordan denne sektion skal være ud fra følgende retningslinjer:
    {section_guidelines}

    Dette er sektionen du skal kritisere:
    {generated_text}

    ## HUSK
    - Eksemplerne i guidelines er ikke nødvendigvis relevante for den specifikke patient.
    - Hvis der er skrevet '[Information forefindes ikke]' som svar til et punkt, skal dette punkt ikke kritiseres.
    
    Vurdér om sektionen følger retningslinjerne, følg denne struktur:
    - Du skal kun nævne punkter der skal forbedres
    - Giv konkrete forslag til forbedringer
    - Skriv kort og præcist svar
    """
    
    response = LLM_CRITIQUE.invoke(
        critique_prompt,
        config=RunnableConfig(
            callbacks=[StreamingStdOutCallbackHandler()],
            tags=["streaming_critique", section_title]
        )
    )
    return response

def critique_section_patient_record(section_title: str, generated_text: str, patient_journal: str) -> str:
    """
    Evaluates the generated text against the patient record and provides a critique.
    
    Args:
        section_title: The title of the section being evaluated.
        generated_text: The text content of the section to be critiqued.
        section_guidelines: The hospital guidelines that the section should follow.
    
    Returns:
        The critique result as a string.
    """
    critique_prompt = f"""
    Du har til opgave at give kritik på en genereret sektion '{section_title}' i et medicinsk dokument.
    Du har patients journalen tilgængelig, og du skal vurdere om den genererede sektion er korrekt i forhold til journalen. 

    Patientens journal:
    {patient_journal}

    Dette er sektionen du skal kritisere:
    {generated_text}

    Følg denne struktur når du kritiserer:
    - Du skal lokalisere hvis der er fejl i sektionen, det vil sige hvis et faktum ikke stemmer overens med journalen
    - Undersøg om det er den nyeste relevante information i journalen der er brugt til at udfylde sektionen
    - Du skal ikke give forslag til forbedringer af struktur
    - Du må gerne give forslag til tilføjelser eller ændringer der er relevante for sektionen
    - Skriv kort og præcist svar
    - Du skal ikke give et komplet forslag til sektionen, kun kritik
    - Du skal ikke stille spørgsmål til, giv direkte besked om hvad der skal rettes fordi det ikke stemmer overens med journalen
    """
    
    response = LLM_CRITIQUE.invoke(
        critique_prompt,
        config=RunnableConfig(
            callbacks=[StreamingStdOutCallbackHandler()],
            tags=["streaming_critique", section_title]
        )
    )
    return response

@profile
def generate_subsection(section_title:str, subsection_title:str, section_intro:str, subsection_guidelines:str, patient_data:str) -> str:
    """
    Generates a single subsection based on guidelines and retrieved relevant patient data.
    
    Args:
        section_title: The title of the parent section
        subsection_title: The title of the subsection
        section_intro: The introductory text of the parent section
        subsection_guidelines: The guidelines for this specific subsection
        patient_pdf_path: Path to the patient PDF for critique
        
    Returns:
        The generated subsection text
    """
    # Check token usage before starting
    current_usage = TOKEN_COUNTER.get_stats()
    full_title = f"{section_title} - {subsection_title}"
    print(f"[TOKEN INFO] Before generating '{full_title}': {current_usage['total_tokens']} tokens used")
    
    # Generate an optimized retrieval query for this subsection
    retrieval_query = generate_retrieval_query(section_title, subsection_title, subsection_guidelines)
    
    # Retrieve relevant patient data
    relevant_patient_data = retrieve_patient_info(retrieval_query)
    if relevant_patient_data is None:
        relevant_patient_data = patient_data
    
    # Calculate estimated tokens for this subsection
    estimated_prompt_tokens = TOKEN_COUNTER.count_tokens(f"{full_title}{section_intro}{subsection_guidelines}{relevant_patient_data}")
    print(f"[TOKEN INFO] Estimated prompt size: ~{estimated_prompt_tokens} tokens")
    
    # Check if we might hit token limits
    max_tokens = 32000
    if current_usage['total_tokens'] + estimated_prompt_tokens > max_tokens * 0.8:
        print(f"\n[WARNING] Approaching token limit! Consider splitting document generation.")
        print(f"[WARNING] Current usage: {current_usage['total_tokens']}, Estimated new tokens: {estimated_prompt_tokens}")
    
    # Create the prompt for subsection generation, including section intro
    subsection_prompt = f"""
    Du er en sygeplejeske, der skriver underafsnittet '{subsection_title}' under hovesektionen '{section_title}' baseret på hospitalets retningslinjer.

    ## INDLEDNING TIL SEKTION '{section_title}':
    {section_intro}

    ## RETNINGSLINJER FOR DETTE UNDERAFSNIT:
    {subsection_guidelines}

    ## RELEVANT PATIENT INFORMATION:
    {relevant_patient_data}

    ## DIN OPGAVE:
    1. Generer underafsnittet '{subsection_title}' PÅ DANSK 
    2. Udfyld ALLE nødvendige felter fra retningslinjerne, men uden at gentage retningslinjerne
    3. Du skal bruge information fra patientjournalen til at udfylde underafsnittet
    4. Brug altid de nyeste informationer fra journalen, hvis der er modstridende oplysninger
    5. Formatér dit svar med punktopstilling for hvert punkt i retningslinjerne som skal besvares
    6. Tænk dig om, og find en sammenhæng mellem patientens tilstand i patientjournalen og hvad der skal stå i underafsnittet
    7. Hvis du ikke kan finde informationen i det tilgængelige uddrag af patientjournalen, så skriv at du ikke kan finde informationen, du skal ikke lave en eksempel-udfyldning.
    
    ## HUSK
    - Eksemplerne i guidelines er ikke nødvendigvis relevante for den specifikke patient
    - Vær kort og præcis i dine svar
    - Dine svar skal være relevante for patientens nuværende indlæggelsesforløb.
    - DU SKAL FØLGE RETNINGSLINJERNE NØJE OG IKKE TILFØJE PUNKTER SOM IKKE ER RELEVANTE FOR UNDERAFSNITTET
    - Lad vær med at inkludere * formatering
    - ikke alt information i patientjournalen er relevant for underafsnittet, så vær selektiv i hvad du inkluderer

    ## VIGTIGT
    - Du SKAL følge denne proces og formatering præcist som beskrevet
    - Du SKAL følge kravene i retningslinjerne meget nøje og ikke tilføje punkter som ikke er relevant for underafsnittet.
    """
    
    print(f"\n\n [INFO] Generating subsection '{full_title}' \n\n:")
    # Generator agent reasons the task and generate first draft
    output = LLM_GENERATE.invoke(
        subsection_prompt,
        config=RunnableConfig(
            callbacks=[StreamingStdOutCallbackHandler()],
            tags=["subsection_generation", full_title]
        )
    )
    
    # Check token usage after initial generation
    current_usage = TOKEN_COUNTER.get_stats()
    print(f"[TOKEN INFO] After initial generation: {current_usage['total_tokens']} tokens used")
    
    return output

@profile
def generate_section_with_subsections(section_title:str, section_guidelines:str, patient_data:str)-> str:
    """
    Processes a section by splitting it into subsections, generating each subsection,
    and then combining them into a complete section.
    """
    print(f"\n\n [INFO] Processing section '{section_title}' with subsections\n\n:")
    
    # Split the section into introduction and subsections
    section_intro, subsections = split_section_into_subsections(section_guidelines)
    
    print(f"[INFO] Found {len(subsections)} subsections in '{section_title}'")
    
    # Process each subsection
    generated_subsections = []
    for i, (subsection_title, subsection_guidelines) in enumerate(subsections, 1):
        print(f"[INFO] Processing subsection {i}/{len(subsections)}: '{subsection_title}'")
        
        # Reset token counter for each subsection to avoid token limit issues
        TOKEN_COUNTER.reset()
        print("[TOKEN INFO] Token counter reset for new subsection generation")
        
        # Generate the subsection content
        subsection_content = generate_subsection(
            section_title=section_title,
            subsection_title=subsection_title,
            section_intro=section_intro,
            subsection_guidelines=subsection_guidelines,
            patient_data=patient_data,
        )
        
        # Save the generated subsection
        generated_subsections.append((subsection_title, subsection_content))
        print(f"[RESULT] Subsection '{subsection_title}' completed!")
    
    # Combine all subsections into a complete section
    if len(subsections) == 1 and subsections[0][0] == "Main Content":
        # If there's only one subsection and it contains the original content,
        # return just the generated content without subsection headers
        complete_section = f"{section_intro}\n\n{generated_subsections[0][1]}"
    else:
        # Otherwise, combine subsections with their titles
        combined_parts = [section_intro]
        for subsection_title, subsection_content in generated_subsections:
            if subsection_title != "Main Content":
                combined_parts.append(f"## {subsection_title}\n{subsection_content}")
            else:
                combined_parts.append(subsection_content)
        
        complete_section = "\n\n".join(combined_parts)
    
    print(f"[RESULT] Created section '{section_title}' assembled from {len(generated_subsections)} subsections!")
    
    print(f"\n\n [INFO] Generating guideline critique to the section '{section_title}'\n\n:")
    # Critique Agent gives critique of the generated subsection
    critique_result_guideline = critique_section_guideline(
        section_title=section_title,
        generated_text=complete_section,
        section_guidelines=section_guidelines
    )

    # Use full patient data for critique (critique still uses full record for better verification)
    print(f"\n\n [INFO] Generating patient information critique to the section '{section_title}'\n\n:")
    critique_result_patient_data = critique_section_patient_record(
        section_title=section_title,
        generated_text=complete_section,
        patient_journal=patient_data
    )
    
    # Check token usage after critique
    current_usage = TOKEN_COUNTER.get_stats()
    print(f"[TOKEN INFO] After critique: {current_usage['total_tokens']} tokens used")
    
    # Define the critique prompt for adjusting the generated text
    adjustment_prompt = f"""
    Du har skrevet dette:
    {complete_section}

    Du skal rette det underafsnit '{subsection_title}' du skrev før efter følgende kritik:
    {critique_result_guideline}
    og
    {critique_result_patient_data} 

    VIGTIG:
    - Du skal kun rette det du har skrevet tidligere, svar intet andet end din rettelse.
    - Brug kritikken til at forbedre dit svar ved at tilføje, ændre eller fjerne information.
    - Sørg for at dit svar er skrevet på dansk!
    - Dit svar skal stadig dække alle punkter fra guidelines som du har fulgt da du lavede din første version.
    - Husk at inkludere samtlige subsections fra din første version
    """

    # The generated subsection is adjusted to accommodate the critique
    print(f"\n\n [INFO] Adjusting section '{section_title}' to critique. \n\n")
    improved_output = LLM_GENERATE.invoke(
        adjustment_prompt,
        config=RunnableConfig(
            callbacks=[StreamingStdOutCallbackHandler()],
            tags=["section_improvement", section_title]
        )
    )
 
    return improved_output


####################################################
# Extract PDF information, final answer and indexing/outputting of document creation
####################################################

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text information from a PDF patient record.
        Args:
            pdf_path (str): Path to the PDF file containing patient record.
        Returns:
            str: Extracted text from all pages of the PDF.
    """
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    print("[RESULT] Patient data has been extracted")
    return text

def extract_final_section(response: str) -> str:
    """
    Extracts the final improved section from the response object.
    Handles different response formats from the LLM.
        Args:
            response (str or dict): The response from the LLM containing the final section.
        Returns:
            str: The extracted section text.
    """
    if isinstance(response, dict) and 'output' in response:
        return response['output']
    elif isinstance(response, str):
        return response
    else:
        print("[ERROR] Unable to extract final section from response.")
        return ""

def assemble_final_document(section_outputs: dict) -> str:
    """
    Assemble the final document from individual section texts.
        Args:
            section_outputs (dict): Dictionary with section titles as keys and their content as values.
        Returns:
            str: The complete document with all sections properly formatted.
    """
    print("[RESULT] Final document has been created from guidelines!")
    return "\n\n".join([f"Overskrift: {title}:\n{text}\n\n" for title, text in section_outputs.items()])

def index_final_document():
    """
    Indexes the generated final document into a vector database for future queries.
    This function creates or updates the vector database of generated documents,
    enabling the system to reference and search within previously generated documents.
    """
    # Gather the global values
    global GENERATED_DOCS_VECTOR_DB
    global EMBEDDINGS
    global GENERATED_DOCS_DB_DIR
    
    # Creating vectordatabase for the generated documents, enabling retrieval
    Chroma_db_generated_document_final.main()
    GENERATED_DOCS_VECTOR_DB = Chroma(persist_directory=GENERATED_DOCS_DB_DIR, embedding_function=EMBEDDINGS)
    
def save_to_pdf(text:str, output_name:str) -> None:
    """
    Save document to a PDF file with full Danish character support.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add a Unicode font (e.g., DejaVuSans)
    font_path = "DejaVuSans.ttf"  # Make sure this TTF file is available in your project directory
    if not os.path.isfile(font_path):
        raise FileNotFoundError(f"Font file {font_path} not found. Please download it and place it in the working directory.")
    
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)
    
    # Ensure proper UTF-8 encoding (no character loss)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    pdf.multi_cell(0, 10, text)  # Use multi_cell for proper text wrapping
    pdf.output(output_name, "F")
    print(f"PDF saved as {output_name}")
    return None


####################################################
# Main workflow
####################################################

# Generate output
def generate_answer(query: str, output_name: str, patient_pdf_path=None) -> None:
    """
    Main workflow function that processes a query, retrieves information, and generates documents as needed.
    This function orchestrates query handling:
    - activates Retriever agent to use tools to gather information to answer query
    - directs answer to query if so or orchistrates document creation
    This function orchestrates the entire document generation process:
    - Token usage tracking
    - Patient data extraction (if PDF provided)
    - Invoking the retrieval agent to gather information
    - Creating empty guidelines (if no patient data) or complete document (if patient data available)
    - Saving and indexing the final document

        Args:
            query (str): The user's query to process.
            output_name (str): Filename for saving the generated document.
            patient_pdf_path (str, optional): Path to PDF containing patient data. Defaults to None.
        Returns:
            None: Prints results and saves document to PDF if applicable.
    """
    
    # Track query tokens
    TOKEN_COUNTER.log_usage(query)

    retrieval_answer = invoke_retrieval_agent(query)

    # If the retrieved data is a guideline and there IS NOT a patient record as PDF:
    # The guidelines are created but not filled, and saved as a PDF-file.
    if RETRIEVED_GUIDELINES is not None and patient_pdf_path is None:
        print("[INFO] No patient data provided. Returning concatenated guideline sections.")
        concatenated_guidelines = "\n\n".join(
            f"Section: {title}\nGuidelines:\n{guideline}" for title, guideline in RETRIEVED_GUIDELINES.items()
        )
        save_to_pdf(concatenated_guidelines, f"medical_document_guidelines.pdf")
        return print(retrieval_answer)
    
    # If the retrieved data is a guideline and there IS a patient record as PDF:
    if RETRIEVED_GUIDELINES is not None and patient_pdf_path is not None:
        print("[INFO] Creating document from retrieved guideline sections from the patient record.")
        
        if isinstance(retrieval_answer, str):
            print(retrieval_answer)
        
        # Sections from the retrieved guidelines has been extracted with the retriever agent triggered the generation tool
        sections = RETRIEVED_GUIDELINES
        
        # Collecting the patient data from the PDF
        patient_data = extract_text_from_pdf(patient_pdf_path)
        # Estimate patient data tokens
        patient_tokens = TOKEN_COUNTER.count_tokens(patient_data)
        print(f"[TOKEN INFO] Patient data size: ~{patient_tokens} tokens")
        
        # Check if patient data alone might exceed token limits
        if patient_tokens > 32000:
            print(f"[WARNING] Patient data is very large ({patient_tokens} tokens)")
            print("[WARNING] Consider truncating or summarizing the patient data")
        
        # Calculate total estimated token usage for all sections
        total_estimated = 0
        for title, guideline in sections.items():
            section_est = TOKEN_COUNTER.count_tokens(f"{title}{guideline}{patient_data}")
            total_estimated += section_est
            print(f"[TOKEN INFO] Estimated tokens for '{title}': ~{section_est}")
        
        print(f"[TOKEN INFO] Total estimated tokens for all sections: ~{total_estimated}")
        
        generated_sections = {}
        for i, (title, guideline) in enumerate(sections.items(), 1):
            TOKEN_COUNTER.reset()
            print("[TOKEN INFO] Token counter reset for new section generation")
            print(f"[INFO] Creating section {i}/{len(sections)}: '{title}'")
                
            section_output = generate_section_with_subsections(title, guideline, patient_data)
            print("[RESULT] Section is completed!")
            # Extract the final section text from the response, depending on the response format
            # The response can be a string or a dictionary, so we handle both cases
            final_section = extract_final_section(section_output)
            generated_sections[title] = final_section
            
        print("[RESULT] All sections has been completed!")
        
        # Finalize the document by assembling all sections
        final_document = assemble_final_document(generated_sections)
        return save_to_pdf(final_document, output_name)
    
    # If the retrieved data is information for question answering from API or other of the retriever agent's tools
    else: 
        print("[RESULT] The final answer from data-collection is:\n")
        # Log the token usage for this response
        
        return print(retrieval_answer)


##########################
# Start RAG-system from user query
##########################


# Retrieval router agent decides course of action from query
if __name__ == "__main__":
    
    # PDF is optional, but needed for creating a medical document for a specific patient
    if os.path.join(os.path.dirname(__file__), "patient_record", "Patientjournal GPT6.pdf"):
        patient_pdf_path = os.path.join(os.path.dirname(__file__), "patient_record", "Patientjournal GPT6.pdf")
    else:
        patient_pdf_path = None
    
    # Name of the output file, if a document is created
    output_name = "generated_medical_document.pdf"
    
    # Examples of quries to test the system:
    
    # "Generér en plejeforløbsplan."
    # "Hvorfor får paitenten proteindrik?"
    # "Hvorfor har patienten nedsat kognetiv evne?"
    # "Hvilke sektioner skal en plejeforløbsplan indeholde?"
    # "Hvorfor er patienten blevet indlagt?"
    # "Hvem har tilset patienten?"
    # "Uden at bruge det genererede dokument, brug patientjournalen til at besvare punkterne under 'sygepleje' i guideline for plejeforløbsplan."

    # Insert one or more here:
    
    queries = ["Generér en plejeforløbsplan.","Hvad fortæller den genrerede plejeforløbsplan om patientens psykosociale forhold?"]
    #queries = ["Hvordan passer patientens funktionsevne i forhold til at kunne vaske sig ifølge patientjournalen, med vurderingen skrevet i det genererede plejeforløbsplan dokument? Tilsidst, vurdér ifølge hospitalets retningslinjer for en plejeforløbsplan, om denne vurdering skal have en højere eller lavere score end den har nu, eller om den er som den skal være.", "I mit forrige spørgsmål, hvad er din forklaring på hvorfor du synes vurderingen er korrekt?"]
    
    
    for query in queries:
        answer = generate_answer(query, output_name, patient_pdf_path)
        
        if answer != None:
            print(answer)
    
