# 🎓 Placement Intelligence System (Powered by Corrective RAG)

An intelligent, reliable campus placement analysis assistant built for B.Tech students. Powered by **Corrective RAG (CRAG)**, this application allows students to ingest company Job Descriptions (JDs), eligibility matrices, skill requirements, and interview preparation materials to get evidence-backed answers with zero hallucination.

---

## 🌟 Key Features

- **Multi-Document Ingestion**: Batch upload PDFs and TXT documents (JDs, eligibility notices, syllabus documents).
- **Zero-Cloud In-Memory Vector Search**: Uses local Qdrant in-memory vector storage (`location=":memory:"`) powered by OpenAI embeddings.
- **Corrective RAG Pipeline**:
  - **Relevance Grader**: LLM evaluates whether retrieved vector chunks contain relevant evidence.
  - **Query Transformer**: Rewrites underspecified or failed queries into optimized search terms.
  - **Web Search Fallback**: Integrates Tavily API for external search fallback when document context is missing.
- **Placement Analytics Dashboard**: Document list overview, chunk breakdown, and vector database specs.
- **CRAG Execution Inspector**: Visual accordion panel in Streamlit showing step-by-step CRAG execution traces (Grade: RELEVANT vs. INSUFFICIENT, rewritten queries, and source citations).
- **Student-Friendly UI**: Dark glassmorphic modern design with one-click preset query buttons ("Which companies require React.js?", "Python requirements", "Eligibility cutoffs").

---

## 🏗 Project Architecture

```
+-----------------------------------------------------------------------------------+
|                                   STREAMLIT UI                                    |
|   +--------------------------+    +-----------------------+    +--------------+   |
|   | Document Management Tab  |    | Placement Q&A Chat UI |    | CRAG Inspector|   |
|   +--------------------------+    +-----------------------+    +--------------+   |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                DOCUMENT PIPELINE                                  |
|   [ PDF / TXT Upload ] -> [ PyPDF/Text Splitter ] -> [ OpenAI Embeddings ]         |
|                                                                 |                 |
|                                                                 v                 |
|                                                     [ Qdrant Vector Store ]       |
+-----------------------------------------------------------------------------------+
                                                                  |
                                                                  v (Similarity Search)
+-----------------------------------------------------------------------------------+
|                           CORRECTIVE RAG (CRAG) ENGINE                            |
|                                                                                   |
|        +------------------------------------------------------------------+       |
|        | 1. DOCUMENT RELEVANCE EVALUATOR (LLM Grade: Relevant / Weak)     |       |
|        +------------------------------------------------------------------+       |
|                                      |                                            |
|                  +-------------------+-------------------+                        |
|                  | (If Relevant)                         | (If Irrelevant/Weak)  |
|                  v                                       v                        |
|        +-------------------+                   +-------------------+              |
|        | 2a. GENERATE      |                   | 2b. QUERY REWRITE |              |
|        |     RESPONSE      |                   |     & RE-RETRIEVE |              |
|        +-------------------+                   +-------------------+              |
|                  |                                       |                        |
|                  v                                       v                        |
|        +------------------------------------------------------------------+       |
|        | 3. FINAL ANSWER GENERATION (With Document Source Attribution)    |       |
|        +------------------------------------------------------------------+       |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 How to Setup & Run (Local Ollama)

### 1. Install & Setup Ollama
Download and install [Ollama](https://ollama.com/), then pull the required local AI models in your terminal:
```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

### 2. Activate Environment & Install Dependencies
```bash
# Activate your virtual environment
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launch the Streamlit Application
```bash
streamlit run app.py
```

### 4. Using the Application
1. Click **"🚀 Load Samples"** in the sidebar to load built-in sample placement JDs (Google, TCS, Accenture) using local `nomic-embed-text` embeddings.
2. Use the one-click preset prompt buttons or ask custom placement questions:
   - *"Which companies require React.js?"*
   - *"What are the eligibility requirements for TCS?"*
   - *"Compare Google vs Accenture requirements."*
3. Open the **"🔍 CRAG Pipeline Execution Inspector"** expander under any response to inspect how `qwen3:4b` evaluated evidence and performed query corrections!
4. Open the **"🔍 CRAG Pipeline Execution Inspector"** expander under any response to see how the Corrective RAG evaluator and query transformer operated!

---

## 📁 Directory Structure

```
CRAG/
├── placement_intel/               # Core Placement Intelligence Package
│   ├── __init__.py
│   ├── app.py                     # Streamlit application UI layout & tabs
│   ├── config.py                  # Default parameters and model definitions
│   ├── doc_processor.py           # Document loading, PDF extraction, and text splitting
│   ├── vector_store.py            # Local in-memory Qdrant client manager
│   ├── crag_engine.py             # Corrective RAG pipeline (Grader, Rewriter, Generator)
│   └── prompts.py                 # Placement-tailored prompt templates
│
├── sample_data/                   # Sample Placement Documents
│   ├── Google_Software_Engineer_JD.txt
│   ├── TCS_Ninja_Digital_Eligibility.txt
│   └── Accenture_ASE_Requirement.txt
│
├── app.py                         # Root entry point
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🎯 Technical Highlights for Job Interviews

When presenting this project in AI/ML technical interviews, emphasize:
1. **Self-Corrective Architecture**: Why standard RAG fails (hallucinations on poor context) and how CRAG solves it through evaluation and query transformation.
2. **Domain Adaptation**: How general RAG was customized for Placement Intelligence (extracting CGPA cutoffs, allowed streams, tech stacks, selection processes).
3. **Optimized Vector Infrastructure**: Using local in-memory Qdrant to eliminate cloud latency while preserving full vector similarity search capabilities.
