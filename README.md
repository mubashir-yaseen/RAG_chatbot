# RAG Chat System

A powerful **Retrieval Augmented Generation (RAG)** system built with Python that allows you to upload PDF documents and chat with them using AI.

## Features

- 📄 **PDF Upload**: Extract text from PDF documents
- 🔍 **Semantic Search**: Find relevant content using HuggingFace embeddings
- 💾 **Local Vector Storage**: Use FAISS for fast, efficient retrieval
- 🤖 **AI-Powered Responses**: Generate answers grounded in your documents
- 🔒 **Strict Prompt Enforcement**: LLM only answers from retrieved content
- 🎨 **User-Friendly UI**: Streamlit interface for easy interaction
- 📌 **Source Attribution**: See which parts of documents were used

## Tech Stack

- **PyMuPDF**: PDF text extraction
- **LangChain**: RAG orchestration and chain management
- **FAISS**: Vector database for semantic search
- **Sentence Transformers**: HuggingFace embedding models
- **OpenAI/OpenRouter**: Language model for response generation
- **Streamlit**: Web-based user interface

## Installation

### 1. Clone or navigate to the project directory

```bash
cd d:\projects\RAG
```

### 2. Create a virtual environment (optional but recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API Key

Get your API key from either:

- **OpenAI**: [OpenAI Platform](https://platform.openai.com/account/api-keys)
- **OpenRouter**: [OpenRouter](https://openrouter.ai/keys) (free tier available)

Provide it in the Streamlit UI (recommended) or set environment variable:

```bash
# On Windows
set OPENAI_API_KEY=your_api_key_here

# On macOS/Linux
export OPENAI_API_KEY=your_api_key_here
```

**Note**: The system now supports both OpenAI and OpenRouter APIs. Select your provider in the sidebar.

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Configure Settings** (Optional):
   - Adjust embedding model, LLM model, temperature, etc.
   - Set chunk size and overlap for text splitting

2. **Initialize RAG System**:
   - Provide OpenAI API Key
   - Click "Initialize RAG System"

3. **Upload PDF**:
   - Click "Choose a PDF file"
   - Click "Process PDF" to extract and index the document

4. **Chat**:
   - Ask questions about your document
   - View responses with source attribution

## Project Structure

```
RAG/
├── app.py                 # Streamlit UI application
├── rag_system.py          # Core RAG system implementation
├── requirements.txt       # Python dependencies
├── vector_stores/         # Saved vector stores (created on first use)
└── README.md             # This file
```

## Code Overview

### RAGSystem Class

The core of the system:

```python
from rag_system import RAGSystem

# Initialize
rag = RAGSystem(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo",
    temperature=0.3
)

# Extract text from PDF
text = rag.extract_text_from_pdf("document.pdf")

# Chunk and create vector store
chunks = rag.chunk_text(text, chunk_size=1000)
rag.create_vector_store(chunks)

# Setup RAG chain with strict prompt
rag.setup_rag_chain(k=3)

# Query the system
result = rag.query("What is this document about?")
print(result["answer"])
```

## Configuration Options

### Advanced Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Embedding Model** | `all-MiniLM-L6-v2` | HuggingFace model for embeddings |
| **LLM Model** | `openai/gpt-3.5-turbo` | OpenAI/OpenRouter model to use |
| **API Provider** | `OpenRouter` | Choose between OpenAI and OpenRouter |
| **Temperature** | `0.3` | Lower = more consistent responses |
| **Chunk Size** | `1000` | Characters per chunk |
| **Chunk Overlap** | `200` | Overlap between chunks |
| **K Results** | `3` | Number of chunks to retrieve |

## Prompt Strategy

The system uses a **strict system prompt** that:

1. ✅ Forces the LLM to only answer using retrieved context
2. ✅ Explicitly tells the model to say "I cannot answer" if info is not found
3. ✅ Prevents hallucination and external knowledge usage
4. ✅ Requires source citations

### Sample Strict Prompt

```
You are a helpful assistant that answers questions ONLY based on the provided context.

IMPORTANT RULES:
1. ONLY answer questions using information from the provided context chunks below.
2. If the answer cannot be found in the context, explicitly say: 
   "I cannot answer this question based on the provided documents."
3. Do NOT use any external knowledge or make up information.
4. Always cite which part of the context you're using to answer.
```

## Troubleshooting

### "OPENAI_API_KEY not found"
- Provide your API key in the Streamlit sidebar
- Ensure you have a valid OpenAI or OpenRouter account with API access
- Select the correct API provider in the sidebar

### "FAISS installation failed"
- Try: `pip install faiss-cpu` (CPU version)
- Or: `pip install faiss-gpu` (GPU version, if CUDA available)

### "Slow embedding generation"
- This is normal for first-time use (downloads model)
- Subsequent chunks process faster
- Consider using smaller chunk sizes initially

### "Out of memory"
- Reduce chunk size
- Use a smaller embedding model
- Process documents in smaller batches

## Performance Tips

1. **Use smaller embedding models** for faster processing:
   - `all-MiniLM-L6-v2` (fastest)
   - `paraphrase-MiniLM-L6-v2`
   - `all-mpnet-base-v2` (slower but better quality)

2. **Adjust chunk size** based on document type:
   - Long documents: larger chunks (1500-2000)
   - Technical docs: smaller chunks (500-800)

3. **Use appropriate K value**:
   - k=3: Fast, good for brief answers
   - k=5: Balanced
   - k=10: More context, slower

## API Reference

### RAGSystem Methods

#### `extract_text_from_pdf(pdf_path: str) -> str`
Extract text from a PDF file.

#### `chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list`
Split text into chunks with overlap.

#### `create_vector_store(documents_list: list) -> None`
Create FAISS vector store from documents.

#### `setup_rag_chain(k: int) -> None`
Setup the RAG chain with strict prompt.

#### `query(question: str) -> dict`
Ask a question and get answer with sources.

#### `save_vector_store(save_path: str) -> None`
Save vector store to disk.

#### `load_vector_store(load_path: str) -> None`
Load previously saved vector store.

## Example Use Cases

- 📚 Research papers: Index and search academic content
- 📋 Manuals: Create searchable technical documentation
- 🏢 Company docs: Build internal knowledge bases
- 📰 News articles: Analyze and summarize content
- 📖 Books: Create interactive reading companions

## Limitations

- Requires API key (OpenAI paid or OpenRouter free)
- Performance depends on chunk size and model quality
- Large PDFs may take time to process
- Answers are limited to document content

## Future Enhancements

- [ ] Support for multiple documents
- [ ] RAG with local LLMs (Ollama, llama.cpp)
- [ ] Multi-language support
- [ ] Better source highlighting
- [ ] Export/import chat history
- [ ] Document comparison
- [ ] Batch processing

## License

MIT License - feel free to use and modify

## Support

For issues or questions:
1. Check troubleshooting section
2. Review OpenAI API documentation
3. Check LangChain documentation
