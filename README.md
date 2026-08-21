# 🎯 Placement Intelligence System

An AI-powered placement assistant that helps students analyze company job descriptions, eligibility criteria, technical requirements, and compare opportunities across companies using **Corrective Retrieval-Augmented Generation (CRAG)**.

The system combines **semantic search, vector retrieval, relevance evaluation, query correction, and local LLM generation** to provide answers grounded in uploaded placement documents.

---

## ✨ Features

### 📄 Placement Document Analysis
- Upload placement-related PDF and TXT documents.
- Automatically extract and process document content.
- Split documents into searchable chunks.
- Store document embeddings in a local vector database.

### 🤖 Placement Q&A
Ask questions about uploaded company documents, such as:
- What technical skills are required?
- Does this role require Python and SQL?
- What are the eligibility requirements?
- What qualifications are required?
- Which companies require a particular technology?

Answers are generated using the information retrieved from the uploaded documents.

### 🔄 Corrective RAG
The system does not directly generate an answer after the first retrieval. It evaluates whether the retrieved information is relevant to the user's question.

If the retrieved context is insufficient, the system:
1. **Rewrites the query** into optimized search keywords.
2. **Performs another retrieval** (or web search fallback).
3. **Checks the available evidence**.
4. **Generates an answer** only when sufficient information is available.
5. **Clearly reports** when the requested information cannot be found in the indexed documents.

This helps eliminate hallucinations and unsupported answers from irrelevant documents.

### ⚖️ Company Comparison
Compare multiple companies side-by-side using their uploaded placement documents:
- Technical Skills & Tech Stack
- Eligibility Requirements & CGPA Cutoffs
- Interview Rounds & Selection Process
- Job / Role Requirements
- Comprehensive Multi-Company Requirements

If a requirement is not mentioned in a company's document, the system explicitly reports `Not mentioned`.

### 📊 Document Analytics
Overview of all indexed placement documents, chunk metrics, and in-memory vector database specifications.

### 🔍 CRAG Pipeline Inspector
Visual accordion panel in the Streamlit UI displaying step-by-step CRAG execution traces:
- Initial retrieval
- Context relevance evaluation (Grade: `RELEVANT` vs. `INSUFFICIENT`)
- Query rewriting & transformation
- Corrective re-retrieval
- Final grounded answer generation with cited sources

---

## 🧠 How the System Works

The application follows a Corrective RAG architecture:

```text
                     User Question
                           │
                           ▼
                    Query Processing
                           │
                           ▼
                    Vector Retrieval
                           │
                           ▼
                 Context Relevance Check
                           │
                    ┌──────┴──────┐
                    │             │
                 Relevant      Insufficient
                    │             │
                    ▼             ▼
             Generate Answer   Rewrite Query
                    │             │
                    │             ▼
                    │        Re-retrieval
                    │             │
                    │             ▼
                    │       Evidence Check
                    │             │
                    └──────┬──────┘
                           ▼
                  Grounded Response
                           │
                           ▼
                    Source Documents
```

### Retrieval Flow Comparison

#### Normal Query (Sufficient Context)
```text
Question ──► Retrieve Documents ──► Check Relevance (RELEVANT) ──► Generate Answer
```

#### Corrective Query (Insufficient Context)
```text
Question ──► Retrieve Documents ──► Check Relevance (INSUFFICIENT) ──► Rewrite Query ──► Retrieve Again ──► Check Evidence ──► Generate Grounded Answer
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core application development |
| **Streamlit** | Interactive web application interface |
| **LangChain** | LLM orchestrator and retrieval integration |
| **Qdrant (In-Memory)** | Local vector database & similarity search |
| **Ollama** | Local LLM inference engine |
| **Qwen3 (1.7B)** | Local reasoning & generation model |
| **Nomic Embed Text** | Text embeddings generation |
| **PyPDF** | PDF document processing and text extraction |

---

## 📂 Project Structure

```text
placement-intelligence-system/
│
├── app.py                             # Root entry point
│
├── placement_intel/                   # Core Package
│   ├── __init__.py
│   ├── app.py                         # Streamlit UI layout & tabs
│   ├── config.py                      # Default parameters & models
│   ├── crag_engine.py                 # Corrective RAG pipeline
│   ├── doc_processor.py               # PDF/TXT extraction & chunking
│   ├── prompts.py                     # Evaluation & generation prompts
│   └── vector_store.py                # Local in-memory Qdrant client
│
├── sample_data/                       # Sample Placement Documents
│   ├── Google_Software_Engineer_JD.txt
│   ├── TCS_Ninja_Digital_Eligibility.txt
│   └── Accenture_ASE_Requirement.txt
│
├── .streamlit/
│   └── config.toml
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Jatin0977/placement-intelligence-system.git
cd placement-intelligence-system
```

### 2. Create & Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell / CMD)
.venv\Scripts\activate

# Activate on macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🤖 Ollama Setup

This project uses Ollama to run models locally on your machine with zero cloud API costs.

1. Download and install [Ollama](https://ollama.com/).
2. Pull the required models:
```bash
# Pull the language model
ollama pull qwen3:1.7b

# Pull the embedding model
ollama pull nomic-embed-text
```

3. Verify installed models:
```bash
ollama list
```

---

## ▶️ Run the Application

With your virtual environment activated:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📚 Using the Application

### Step 1 — Load Placement Documents
* In the sidebar, click **"🚀 Load Samples"** to automatically index built-in JDs (Google, TCS, Accenture), or upload your own company PDF/TXT documents and click **"📥 Index Files"**.

### Step 2 — Ask Placement Questions
Use the preset quick triggers or type your own question:
- *"Which companies require React.js?"*
- *"What are the eligibility requirements for TCS?"*
- *"What technical skills are required for Google Software Engineer?"*

### Step 3 — Inspect Corrective Retrieval
Open the **"🔍 CRAG Pipeline Execution Inspector"** expander under any response to inspect:
- Context relevance grading (`RELEVANT` vs. `INSUFFICIENT`)
- Step-by-step trace logs
- Rewritten queries (when corrective mode is triggered)
- Cited document evidence

### Step 4 — Multi-Company Comparison
Switch to the **"⚖️ Company Comparison"** tab to compare requirements side-by-side:

| Requirement | Google | Accenture | TCS |
| :--- | :--- | :--- | :--- |
| **Python** | Mentioned | Mentioned | Mentioned |
| **React.js** | Mentioned | Mentioned | Not mentioned |
| **CGPA Cutoff** | Not mentioned | 6.5 CGPA | 60% / 6.0 CGPA |

---

## 🧪 Example Queries

- **Technical Skills:**
  - *"What technical skills are required for Google?"*
  - *"Does Accenture require Python and SQL?"*
  - *"Which companies require React.js?"*

- **Eligibility & Criteria:**
  - *"What are the eligibility criteria and CGPA cutoffs for TCS?"*
  - *"What qualifications are required for Accenture ASE role?"*

- **Cross-Company Comparison:**
  - *"Compare Google and Accenture technical requirements."*
  - *"Compare TCS and Accenture selection process."*

- **Missing Information & Negative Testing:**
  - *"Which companies require Kubernetes?"*
  - *(The system will evaluate context, rewrite the query, and accurately state that the information is not present rather than hallucinating).*

---

## 🔐 Security & Privacy

- **Local Execution**: All embeddings and LLM inference run locally on your system via Ollama.
- **Zero API Costs**: No mandatory cloud LLM subscriptions or external API keys needed.
- **Ignored Files**: The following files are excluded via `.gitignore`:
```text
.env
.venv/
__pycache__/
*.pyc
```

---

## ⚠️ Limitations

- The current Qdrant vector store is configured for local in-memory storage (`:memory:`).
- Answer completeness depends on the quality and detail of uploaded placement documents.
- Local LLM response latency depends on your local hardware specifications (CPU / GPU).

---

## 🔮 Future Improvements

- Persistent vector database storage on disk
- Resume-to-JD matching and skill gap analysis
- Placement preparation roadmap recommendations
- OCR support for scanned placement notices

---

## 🎓 Project Objective

This project demonstrates a production-grade implementation of **Corrective Retrieval-Augmented Generation (CRAG)** applied to campus placement intelligence, combining:
- Self-corrective retrieval loops
- Vector similarity search
- Grounded document attribution
- Local LLM inference

---

## 👨‍💻 Author

**Jatin Kumar**  
B.Tech — Computer Science & Engineering (AI & ML)  
GitHub: [@Jatin0977](https://github.com/Jatin0977)

---

## ⭐ Key Technologies
`Python` `Streamlit` `LangChain` `Qdrant` `Ollama` `Qwen3` `CRAG` `RAG` `Semantic Search`
