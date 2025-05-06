from setuptools import setup, find_packages

setup(
    name="agentic_rag_medical",
    version="0.1.0",
    author="Tue Larsen",
    author_email="tue.larsen24@example.com",
    description="Agentic RAG pipeline for generating and critiquing medical documents using LangChain and local vector stores.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tueismyday/agentic_rag_medical",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0,<0.2.0",
        "langchain-ollama>=0.0.1,<0.1.0",
        "langchain-huggingface>=0.0.1,<0.1.0",
        "langchain_community>=0.0.1,<0.1.0",
        "ollama>=0.1.0,<0.2.0",
        "sentence-transformers>=2.2.2,<3.0.0",
        "torch>=2.0.0,<3.0.0",
        "transformers>=4.38.0,<5.0.0",
        "chromadb>=0.4.18,<0.5.0",
        "PyPDF2>=3.0.0,<4.0.0",
        "fpdf>=1.7.2,<2.0.0",
        "tiktoken>=0.5.1,<1.0.0",
        "pydantic>=2.0.0,<3.0.0",
        "rich>=13.0.0,<14.0.0"
    ],
    python_requires='>=3.12',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
