# agentic_rag_medical
Agentic RAG workflow for medical documentation automation - a proof of concept

This project implements an **agentic Retrieval-Augmented Generation (RAG)** system for generating post-discharge medical documents and answering clinical queries using patient journal records and hospital guidelines. Built in Python with LangChain and local vector databases, the system simulates a multi-step, expert-level clinical reasoning process to reduce administrative burden in healthcare.

---

## Installation Guide

### Prerequisites

* Python 3.12+ installed
* Pip package manager
* Sufficient disk space for language models (at least 75GB recommended)

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

### 3. Install and set up Ollama

* Linux: `curl -fsSL https://ollama.com/install.sh | sh`
* macOS/Windows: Download and install from [https://ollama.com/download](https://ollama.com/download)

### 4. Download required models

```bash
ollama pull gemma3:12b
```

### 5. Create custom models with extended context windows

Create the file `ollama_models/gemma3-12b-32k-it` containing:

```
FROM gemma3:12b
PARAMETER num_ctx 32000
```

Then run:

```bash
ollama create gemma3:12b-32k -f ollama_models/gemma3-12b-32k-it
ollama create qwen3:4b-32k -f ollama_models/qwen3-4b-32k
ollama create qwen2.5:14b-instruct-8k -f ollama_models/Qwen-14b-Instruct-8k
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
python Agentic_RAG_Final_fo_real_fo_real.py
```

The system will either generate a medical document or answer your query using RAG from the three vector databases, depending on the nature of the input.

### 9. Troubleshooting

* **Model loading errors**:

  * Ensure Ollama is running (`ollama serve`)
  * Check models are downloaded (`ollama list`)
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

* `PATIENT_VECTOR_DB`, `GUILDELINE_VECTOR_DB`, `GENERATED_DOCS_VECTOR_DB`
* Scripts:

  * `Chroma_db_guidelines_final.py`: Splits guidelines by 'Overskrift' sections
  * `Chroma_db_patient_record_final.py`: Chunks by timestamps, infers clinical categories
  * `Chroma_db_generated_document_final.py`: Stores both coarse and fine chunks from generated PDFs

### LangChain Tools

* `retrieve_guideline_knowledge()`: Retrieve sections from guideline database
* `retrieve_patient_info()`: Retrieve timestamped, categorized patient entries
* `retrieve_generated_document_info()`: Retrieve sections from past generated documents
* `start_document_generation()`: Start generation from guideline headings

### Generation and Critique

* `generate_section_with_subsections()`, `generate_subsection()`
* `critique_section_guideline()`, `critique_section_patient_record()`

### Utilities

* `TokenCounter`, `profile`, `extract_text_from_pdf()`, `save_to_pdf()`, `index_final_document()`

### Agents

* `create_retrieval_agent()`, `invoke_retrieval_agent()`, `generate_answer()`

---


