# Research Paper Analysis Toolkit

A comprehensive toolkit for analyzing research papers using RAG (Retrieval-Augmented Generation), knowledge graphs, and AI-powered text explanation. This toolkit consists of three main components:

1. **Research Paper RAG System** - Python tool for building knowledge graphs and RAG systems from research papers
2. **CLI Chat Interface** - Interactive command-line interface for querying papers
3. **Browser Extension** - Real-time text explanation for web content

## Installation

### Prerequisites

1. **Python 3.8+** with pip
2. **Node.js** (for browser extension development)
3. **Local LLM Server** (Ollama or vLLM)

### Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
PyPDF2>=3.0.1
PyMuPDF>=1.23.0
networkx>=3.1
scholarly>=1.7.11
arxiv>=1.4.8
crossref-commons>=0.0.7
nltk>=3.8.1
spacy>=3.6.1
transformers>=4.33.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
numpy>=1.24.3
tqdm>=4.65.0
requests>=2.31.0
rich>=13.5.2
click>=8.1.6
aiohttp>=3.8.5
```

### Additional Setup

1. **Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

2. **Install and setup Ollama:**
```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2
ollama pull mistral
```

3. **Alternative: Setup vLLM server:**
```bash
pip install vllm
# Start vLLM server
python -m vllm.entrypoints.api_server --model mistralai/Mistral-7B-Instruct-v0.1
```

## Usage

### 1. Research Paper RAG System

Build a RAG system from a research paper:

```bash
python research_rag_system.py --paper "path/to/paper.pdf" --depth 2 --download-dir "./papers"
```

**Parameters:**
- `--paper`: Path to the initial PDF paper
- `--depth`: Maximum reference depth (default: 2)
- `--download-dir`: Directory to store downloaded papers (default: "./papers")

**Example:**
```bash
python research_rag_system.py --paper "transformer_paper.pdf" --depth 3
```

This will:
- Extract text and references from the initial paper
- Search for and download referenced papers (up to specified depth)
- Build embeddings and FAISS index
- Create knowledge graph with paper relationships
- Save everything to SQLite database

### 2. CLI Chat Interface

Start an interactive chat session:

```bash
python cli_chat_interface.py --paper "path/to/paper.pdf" [options]
```

**Parameters:**
- `--paper`: Path to research paper PDF (required)
- `--depth`: Reference analysis depth (default: 2)
- `--server-type`: LLM server type - 'ollama' or 'vllm' (default: 'ollama')
- `--server-url`: LLM server URL (default: 'http://localhost:11434')
- `--model`: LLM model name (default: 'llama2')
- `--download-dir`: Directory for downloaded papers (default: './papers')
- `--rebuild`: Rebuild RAG system from scratch

**Example:**
```bash
python cli_chat_interface.py \
  --paper "attention_is_all_you_need.pdf" \
  --server-type ollama \
  --model mistral \
  --depth 2
```

**Chat Commands:**
- `help` - Show available commands
- `stats` - Show knowledge base statistics
- `history` - Show chat history
- `clear` - Clear chat history
- `quit`/`exit`/`q` - Exit the chat

**Example Questions:**
- "What is the main contribution of this paper?"
- "Explain the transformer architecture"
- "What are the limitations mentioned?"
- "How does this compare to previous work?"

### 3. Browser Extension

#### Installation

1. **Prepare extension files:**
```bash
mkdir text-explainer-extension
cd text-explainer-extension

# Copy the extension files:
# - manifest.json
# - content-script.js
# - popup-styles.css
# - popup.html
# - popup.js
# - background.js
# - icons/ (create 16x16, 48x48, 128x128 PNG icons)
```

2. **Load extension in Chrome:**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (top right)
   - Click "Load unpacked"
   - Select the extension directory
   - The extension should now be installed and visible in the toolbar

#### Usage

1. **Configure settings:**
   - Click the extension icon in the toolbar
   - Set your LLM server URL and model
   - Test the connection
   - Ensure the extension is enabled

2. **Explain text:**
   - Navigate to any webpage (research paper, article, etc.)
   - Select text you want explained
   - Click the "🧠 Explain" button that appears
   - View the AI-generated explanation in the popup

#### Settings

Access settings by clicking the extension icon:

- **Enable Extension**: Toggle the extension on/off
- **Server URL**: Your local LLM server (e.g., `http://localhost:11434`)
- **Model**: Model name (e.g., `llama2`, `mistral`)
- **Server Type**: Ollama or vLLM (configured in popup.js)

## Advanced Configuration

### Custom Model Configuration

**For Ollama:**
```bash
# Pull different models
ollama pull codellama
ollama pull neural-chat
ollama pull orca-mini

# Use in CLI
python cli_chat_interface.py --model codellama --paper paper.pdf
```

**For vLLM:**
```bash
# Start vLLM with different model
python -m vllm.entrypoints.api_server \
  --model microsoft/DialoGPT-large \
  --host 0.0.0.0 \
  --port 8000

# Use in CLI
python cli_chat_interface.py \
  --server-type vllm \
  --server-url http://localhost:8000 \
  --model microsoft/DialoGPT-large \
  --paper paper.pdf
```

### Database Management

The system creates several database files:
- `research_rag.db` - Main RAG database
- `knowledge_graph.db` - Knowledge graph storage
- `knowledge_graph.json` - Exportable graph format

**Query database directly:**
```python
import sqlite3

conn = sqlite3.connect('research_rag.db')
cursor = conn.cursor()

# List all papers
cursor.execute("SELECT title, year FROM papers")
papers = cursor.fetchall()

# Get paper concepts
cursor.execute("SELECT paper_id, concept FROM concepts")
concepts = cursor.fetchall()

conn.close()
```

### Extending the System

**Add new paper sources:**
```python
# In research_rag_system.py, extend PaperDownloader class
class CustomPaperDownloader(PaperDownloader):
    def search_custom_source(self, query: str) -> List[Paper]:
        # Implement your custom search logic
        pass
```

**Custom concept extraction:**
```python
# In research_rag_system.py, extend ConceptExtractor class
class CustomConceptExtractor(ConceptExtractor):
    def extract_domain_specific_concepts(self, text: str) -> List[str]:
        # Add domain-specific concept extraction
        pass
```

## Troubleshooting

### Common Issues

1. **LLM Server Connection Failed**
   - Ensure Ollama/vLLM is running: `ollama serve` or start vLLM server
   - Check server URL and port
   - Verify model is available: `ollama list`

2. **PDF Extraction Errors**
   - Install required dependencies: `pip install PyMuPDF PyPDF2`
   - Some PDFs may be protected or corrupted
   - Try alternative PDF processing libraries

3. **Memory Issues with Large Papers**
   - Reduce chunk size in RAG system
   - Limit reference depth
   - Use lighter embedding models

4. **Browser Extension Not Working**
   - Check console for errors (F12 → Console)
   - Verify content script injection
   - Ensure server is accessible from browser

### Performance Optimization

**For large document collections:**
```python
# Use GPU acceleration for embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# Use hierarchical clustering for better organization
import sklearn.cluster

# Cluster papers by similarity
embeddings = np.array([paper.embeddings for paper in papers])
clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=10)
clusters = clustering.fit_predict(embeddings)
```

**For faster retrieval:**
```python
# Use GPU FAISS index
import faiss

# GPU index (if available)
if faiss.get_num_gpus() > 0:
    gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
```

## API Reference

### ResearchPaperRAG Class

```python
rag = ResearchPaperRAG(download_dir="./papers", db_path="research.db")

# Build RAG from paper
await rag.build_rag_from_paper("paper.pdf", max_depth=2)

# Query the system
result = rag.query("What is attention mechanism?", top_k=5)
```

### ChatInterface Class

```python
llm_client = LLMClient(server_type="ollama", base_url="http://localhost:11434")
chat = ChatInterface(rag_system, llm_client)
chat.start_chat("paper.pdf")
```

### Browser Extension API

```javascript
// Content script communication
chrome.runtime.sendMessage({
    action: 'explainText',
    text: selectedText,
    settings: settings
}, (response) => {
    console.log(response.explanation);
});
```

## Contributing

1. **Fork the repository**
2. **Create feature branch:** `git checkout -b feature-name`
3. **Make changes and test thoroughly**
4. **Submit pull request with detailed description**

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-repo/research-paper-rag
cd research-paper-rag

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black research_rag_system.py cli_chat_interface.py
```

## License

MIT License - see LICENSE file for details.

## Changelog

### v1.0.0 (Initial Release)
- Research paper RAG system with knowledge graph
- CLI chat interface with local LLM support
- Browser extension for text explanation
- Support for Ollama and vLLM servers
- arXiv and Crossref integration

### Planned Features
- Support for more paper sources (PubMed, IEEE, ACM)
- Advanced citation analysis
- Paper summarization
- Multi-language support
- Cloud deployment options
- Mobile app interface

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Include logs and error messages

## Examples

### Complete Workflow Example

```bash
# 1. Start LLM server
ollama serve

# 2. Pull model
ollama pull llama2

# 3. Build RAG system
python research_rag_system.py \
  --paper "transformer_attention_paper.pdf" \
  --depth 2 \
  --download-dir "./papers"

# 4. Start chat interface
python cli_chat_interface.py \
  --paper "transformer_attention_paper.pdf" \
  --model llama2 \
  --server-url "http://localhost:11434"

# 5. Install browser extension (manual step)
# Load unpacked extension in Chrome

# 6. Use all three tools together:
# - Query papers in CLI
# - Explain text selections in browser
# - Build comprehensive knowledge base
```

### Integration with Jupyter Notebooks

```python
# notebook_integration.py
from research_rag_system import ResearchPaperRAG
import asyncio

# Initialize in notebook
rag = ResearchPaperRAG()
await rag.build_rag_from_paper("paper.pdf", max_depth=1)

# Query interactively
result = rag.query("Explain the methodology")
print(result['retrieved_documents'][0]['text'])

# Visualize knowledge graph
import matplotlib.pyplot as plt
import networkx as nx

G = rag.knowledge_graph.graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
plt.show()
```

This comprehensive toolkit provides a complete solution for research paper analysis, from automated knowledge base construction to interactive querying and real-time text explanation.