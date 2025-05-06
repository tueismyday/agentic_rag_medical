# agentic_rag_medical
Agentic RAG workflow for medical documentation automation - a proof of concept

This project implements an **agentic Retrieval-Augmented Generation (RAG)** system for generating post-discharge medical documents and answering clinical queries using patient journal records and hospital guidelines. Built in Python with LangChain and local vector databases, the system simulates a multi-step, expert-level clinical reasoning process to reduce administrative burden in healthcare.

The repository includes files for generating a medical document: 
* Synthethic patient record created with ChatGPT 4o
* Guideline example for a medical document with structure needed for optimal performance
* An example of a generated document with the synthetic patient record (this example is generated using gemma3:4b-32k)

---

## Installation Guide

### Prerequisites

* Python 3.9+ installed
* Pip package manager
* Sufficient disk space for language models (at least 75GB recommended)
* Have the font pack in your work folder to ensure compatability with danish characters

### 1. Set up a Python virtual environment

```bash
mkdir Agentic_RAG
cd Agentic_RAG
python -m venv .venv
# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install dependencies

Ensure you have a valid `requirements.txt`, then run:

```bash
pip install -r requirements.txt
```
or use the 
```bash
setup.py
```

### 3. Install and set up Ollama

* Linux: `curl -fsSL https://ollama.com/install.sh | sh`
* macOS/Windows: Download and install from [https://ollama.com/download](https://ollama.com/download)

### 4. Download required models

```bash
ollama pull gemma3:12b
```

### 5. Create custom models with extended context windows

Create the file `ollama_models/gemma3-12b-32k` containing:

```
FROM gemma3:12b
PARAMETER num_ctx 32000
```

Then run:

```bash
ollama create gemma3:12b-32k -f ollama_models/gemma3-12b-32k
```

### 6. Prepare the vector databases

Ensure your PDFs are placed in the correct folders:

* `hospital_guidelines/`
* `patient_record/`
* `generated_documents/`

Then run the database setup scripts:

```bash
python Chroma_db_guidelines_final.py
python Chroma_db_patient_record_final.py
python Chroma_db_generated_document_final.py
```

### 7. Prepare test data

Place a PDF like `Patientjournal_test.pdf` in your `patient_record/` folder.

### 8. Run the system

Before running the system, you can edit the list of queries in the `__main__` section of `Agentic_RAG_Final_fo_real_fo_real.py`. For example:

```python
queries = [
    "Generér en plejeforløbsplan.",
    "Hvad fortæller den genererede plejeforløbsplan om patientens psykosociale forhold?"
]
```

Then run:

```bash
python Agentic_RAG_Final.py
```

The system will either generate a medical document or answer your query using RAG from the three vector databases, depending on the nature of the input.

### 9. Troubleshooting

* **Model loading errors**:

  * Ensure Ollama is running (`ollama serve`)
  * Check models are downloaded custom versions created (`ollama list`)
    
* **Vector DB errors**:

  * Verify path and permissions to ChromaDB directories

---

## Use Cases

* Generate structured post-discharge care plans from patient data.
* Answer clinical questions using:

  * Hospital guidelines
  * Patient records
  * Previously generated documents
* Evaluate generated content against both guidelines and patient notes.

---

## Features

* Multi-agent RAG pipeline (retrieval, generation, critique)
* Local Chroma vector stores (guidelines, patient records, generated docs)
* LangChain tool-based decision-making
* Token counting and profiling
* Iterative critique and refinement workflow

---


## Code Structure

### LLMs and Embeddings

* `TokenCountedOllamaLLM`: Tracks token usage
* `LLM_RETRIEVE`, `LLM_GENERATE`, `LLM_CRITIQUE`: Role-specific models
* `HuggingFaceEmbeddings`: All-MPNET model

### Vector Databases

* `PATIENT_VECTOR_DB`: Vectorized patient records
* `GUILDELINE_VECTOR_DB`: Vectorized hospital guidelines
* `GENERATED_DOCS_VECTOR_DB`: Vectorized generated documents
* Scripts to initialize databases:

  * `Chroma_db_guidelines_final.py`
  * `Chroma_db_patient_record_final.py`
  * `Chroma_db_generated_document_final.py`

### LangChain Tools

* `retrieve_guideline_knowledge()`: Retrieves relevant hospital guideline excerpts.
* `retrieve_patient_info()`: Extracts timestamped, categorized information from patient records.
* `retrieve_generated_document_info()`: Enables Q\&A over previously generated PDFs.
* `start_document_generation()`: Begins document drafting based on guideline structure.

### Generation and Critique

* `generate_section_with_subsections()`: Generates each document section and refines it with critiques.
* `generate_subsection()`: Creates one text subsection from guideline and patient input.
* `critique_section_guideline()`: Evaluates output against guideline expectations.
* `critique_section_patient_record()`: Checks if the generated text aligns with patient data.

### Utilities

* `TokenCounter`: Logs and limits token usage.
* `profile`: Decorator for timing and memory monitoring.
* `extract_text_from_pdf()`: Loads and extracts text from patient PDFs.
* `save_to_pdf()`: Outputs UTF-8 encoded PDF with support for Danish fonts.
* `index_final_document()`: Adds generated document to vector DB for future Q\&A.

### Agents

* `create_retrieval_agent()`: Builds a LangChain multi-tool agent with memory and reasoning.
* `invoke_retrieval_agent()`: Runs the agent with performance logging.
* `generate_answer()`: Core entry point; coordinates agent use, document generation, and output.

---


