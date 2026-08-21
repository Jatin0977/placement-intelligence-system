import os
import glob
import streamlit as st

# Page Configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="PlaceIntel AI - Placement Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

from placement_intel.doc_processor import PlacementDocProcessor
from placement_intel.vector_store import PlacementVectorStore
from placement_intel.crag_engine import PlacementCRAGEngine
from placement_intel.config import DEFAULT_OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL

# Custom CSS for Modern Component Aesthetics
CUSTOM_CSS = """
<style>
    /* Header Card */
    .main-header {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    
    /* Badge styling */
    .badge-chip {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-chip-green {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
    }

    .badge-chip-amber {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff;
    }

    /* Trace Inspector Card */
    .trace-card {
        background: rgba(15, 23, 42, 0.9);
        border-left: 4px solid #6366f1;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        font-size: 0.9rem;
        color: #f8fafc;
    }

    /* Sources container */
    .source-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        color: #f8fafc;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State Variables
def init_session_state():
    if "tavily_api_key" not in st.session_state:
        st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "preset_prompt" not in st.session_state:
        st.session_state.preset_prompt = ""

init_session_state()


# Sidebar: Credentials & Document Management
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/graduation-cap.png", width=64)
    st.title("PlaceIntel AI")
    st.caption("Corrective Placement Intelligence System")
    st.divider()

    st.subheader("🔑 API Key Configuration")
    tavily_key_input = st.text_input(
        "Tavily API Key (Optional)",
        value=st.session_state.tavily_api_key,
        type="password",
        help="Optional fallback for web search when documents lack context"
    )
    st.session_state.tavily_api_key = tavily_key_input

    st.divider()
    st.subheader("📄 Placement Document Management")
    
    # Batch File Upload
    uploaded_files = st.file_uploader(
        "Upload Placement JDs / Notices",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload company JDs, eligibility notices, or syllabus PDFs"
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("📥 Index Files", use_container_width=True)
    with col2:
        load_sample_btn = st.button("🚀 Load Samples", use_container_width=True)

    # Process User Uploaded Files
    if process_btn and uploaded_files:
        with st.spinner("Processing & indexing placement documents into local Qdrant memory (via Ollama nomic-embed-text)..."):
            processor = PlacementDocProcessor()
            
            if st.session_state.vector_store is None:
                st.session_state.vector_store = PlacementVectorStore()
            
            total_chunks = 0
            for file in uploaded_files:
                chunks = processor.load_and_split_file(file.name, file.getvalue())
                added = st.session_state.vector_store.index_documents(chunks)
                total_chunks += added

            st.success(f"Successfully indexed {total_chunks} chunks from {len(uploaded_files)} documents!")

    # Load Default Sample Data (Google, TCS, Accenture)
    if load_sample_btn:
        with st.spinner("Indexing sample placement documents (Google, TCS, Accenture via Ollama)..."):
            sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
            sample_files = glob.glob(os.path.join(sample_dir, "*.*"))
            
            processor = PlacementDocProcessor()
            if st.session_state.vector_store is None:
                st.session_state.vector_store = PlacementVectorStore()
            
            total_chunks = 0
            for s_path in sample_files:
                chunks = processor.load_from_filepath(s_path)
                added = st.session_state.vector_store.index_documents(chunks)
                total_chunks += added
                
            st.success(f"Loaded {len(sample_files)} sample company JDs ({total_chunks} vector chunks)!")

    # Display Status of Indexed Files
    st.divider()
    st.subheader("📚 Indexed Placement Library")
    if st.session_state.vector_store and st.session_state.vector_store.get_indexed_files():
        for fname in st.session_state.vector_store.get_indexed_files():
            st.markdown(f"🟢 `<small>{fname}</small>`", unsafe_allow_html=True)
    else:
        st.info("No documents indexed yet. Upload files or click 'Load Samples'.")


# Main Dashboard Layout
st.markdown("""
<div class="main-header">
    <span class="badge-chip">Corrective RAG Engine</span>
    <span class="badge-chip badge-chip-green">Vector DB: Qdrant (Local)</span>
    <h1 style="margin-top: 0.5rem; margin-bottom: 0.2rem; font-size: 2.2rem; font-weight: 800;">
        🎓 Placement Intelligence System
    </h1>
    <p style="color: #94a3b8; margin: 0; font-size: 1rem;">
        Analyze Company Job Descriptions, CGPA Eligibility, Skill Matrices, and Interview Topics with Evidence-Backed Reliability.
    </p>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_chat, tab_compare, tab_analytics, tab_crag_explain = st.tabs([
    "💬 Placement Q&A Assistant",
    "⚖️ Company Comparison",
    "📊 Document Library & Analytics",
    "🔄 Corrective RAG Workflow"
])


# ==========================================
# TAB 1: Placement Q&A Assistant
# ==========================================
with tab_chat:
    st.markdown("### 💡 Quick Placement Triggers")
    st.caption("Click any preset query to analyze company requirements:")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("⚡ Which companies require React.js?"):
            st.session_state.preset_prompt = "Which companies require React.js?"
    with col_p2:
        if st.button("🐍 Which companies require Python?"):
            st.session_state.preset_prompt = "Which companies require Python?"
    with col_p3:
        if st.button("📋 Eligibility cutoffs for all companies"):
            st.session_state.preset_prompt = "What are the eligibility requirements and CGPA cutoffs for all uploaded companies?"
    with col_p4:
        if st.button("⚖️ Compare Google vs Accenture requirements"):
            st.session_state.preset_prompt = "Compare the technical requirements and eligibility criteria between Google and Accenture."

    st.divider()

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "trace_logs" in message:
                with st.expander("🔍 CRAG Pipeline Execution Inspector", expanded=False):
                    for log in message["trace_logs"]:
                        st.markdown(log)
            if "sources" in message and message["sources"]:
                st.markdown("**Cited Sources:** " + ", ".join([f"`{src}`" for src in message["sources"]]))

    # Input Box (User Prompt or Preset)
    default_input = st.session_state.preset_prompt
    user_query = st.chat_input("Ask any placement-related question (e.g., Which companies allow CSE with 7.0 CGPA?)...")

    # If preset was clicked, override user_query
    if st.session_state.preset_prompt and not user_query:
        user_query = st.session_state.preset_prompt
        st.session_state.preset_prompt = ""  # Reset preset after capturing

    if user_query:
        # Check prerequisites
        if st.session_state.vector_store is None or not st.session_state.vector_store.get_indexed_files():
            st.warning("Please upload placement documents or click 'Load Samples' in the sidebar first.")
        else:
            # Append User Question
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Run Corrective RAG Pipeline
            with st.chat_message("assistant"):
                with st.spinner(f"Analyzing placement evidence & evaluating relevance (via Ollama {DEFAULT_OLLAMA_MODEL})..."):
                    crag_engine = PlacementCRAGEngine(
                        tavily_api_key=st.session_state.tavily_api_key
                    )
                    
                    result = crag_engine.execute_crag_pipeline(
                        question=user_query,
                        vector_store=st.session_state.vector_store
                    )
                    
                    # Render Answer
                    st.markdown(result["answer"])
                    
                    # Render CRAG Execution Inspector
                    with st.expander("🔍 CRAG Pipeline Execution Inspector", expanded=True):
                        if result["grade"] == "RELEVANT":
                            st.markdown("<span class='badge-chip badge-chip-green'>Direct High-Relevance Generation</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span class='badge-chip badge-chip-amber'>Corrective Query Transformation Applied</span>", unsafe_allow_html=True)
                        
                        st.markdown("#### Execution Trace:")
                        for log in result["trace_logs"]:
                            st.markdown(f"- {log}")
                            
                    # Render Cited Sources
                    if result["sources"]:
                        st.markdown("<div class='source-box'><strong>📍 Cited Document Evidence:</strong> " + 
                                    ", ".join([f"<code>{src}</code>" for src in result["sources"]]) + "</div>", unsafe_allow_html=True)

            # Append Assistant Answer to Chat History
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "trace_logs": result["trace_logs"],
                "sources": result["sources"]
            })


# ==========================================
# TAB 2: Company Comparison
# ==========================================
with tab_compare:
    st.markdown("### ⚖️ Multi-Company Placement Comparison")
    st.caption("Compare technical skills, eligibility cutoffs, and hiring criteria side-by-side based strictly on uploaded documents.")

    if st.session_state.vector_store is None or not st.session_state.vector_store.get_indexed_files():
        st.warning("Please upload placement documents or click 'Load Samples' in the sidebar first.")
    else:
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            comp_1 = st.text_input("🏢 Company 1 *", value="Accenture", placeholder="e.g. Accenture")
        with col_c2:
            comp_2 = st.text_input("🏢 Company 2 *", value="Google", placeholder="e.g. Google")
        with col_c3:
            comp_3 = st.text_input("🏢 Company 3 (Optional)", value="TCS", placeholder="e.g. TCS (optional)")

        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            focus_option = st.selectbox(
                "🎯 Comparison Focus Area",
                [
                    "Technical Skills & Tech Stack",
                    "Eligibility Criteria, CGPA Cutoffs & Backlog Policy",
                    "Interview Rounds & Selection Process",
                    "Roles, Job Description & Responsibilities",
                    "Comprehensive (All Criteria)"
                ]
            )
        with col_f2:
            st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
            compare_clicked = st.button("🚀 Compare Companies", type="primary", use_container_width=True)

        if compare_clicked:
            comp_list = [comp_1.strip(), comp_2.strip()]
            if comp_3.strip():
                comp_list.append(comp_3.strip())

            with st.spinner(f"Comparing company evidence across uploaded documents (via Ollama {DEFAULT_OLLAMA_MODEL})..."):
                crag_engine = PlacementCRAGEngine(
                    tavily_api_key=st.session_state.tavily_api_key
                )
                comp_res = crag_engine.compare_companies(
                    companies=comp_list,
                    focus_area=focus_option,
                    vector_store=st.session_state.vector_store
                )

                # Missing company alerts
                if comp_res.get("missing_companies"):
                    for missing in comp_res["missing_companies"]:
                        st.info(f"ℹ️ **{missing}**: No uploaded document found for this company.")

                # Render Comparison Analysis Table & Findings
                st.markdown("#### 📊 Comparison Matrix & Analysis:")
                st.markdown(comp_res["answer"])

                # Render Cited Sources
                if comp_res.get("sources"):
                    st.markdown("<div class='source-box'><strong>📍 Cited Document Evidence:</strong> " + 
                                ", ".join([f"<code>{src}</code>" for src in comp_res["sources"]]) + "</div>", unsafe_allow_html=True)


# ==========================================
# TAB 3: Document Library & Analytics
# ==========================================
with tab_analytics:
    st.markdown("### 📊 Indexed Document Overview")
    if st.session_state.vector_store and st.session_state.vector_store.get_indexed_files():
        files = st.session_state.vector_store.get_indexed_files()
        st.success(f"Currently indexing **{len(files)} placement documents** in local Qdrant memory.")
        
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            st.markdown("#### Document List:")
            for idx, fname in enumerate(files, 1):
                st.markdown(f"**{idx}. {fname}**")
        with col_d2:
            st.markdown("#### Vector DB Specifications:")
            st.json({
                "Vector Store": "Qdrant (In-Memory)",
                "Embedding Model": f"Ollama {DEFAULT_EMBEDDING_MODEL}",
                "LLM Model": f"Ollama {DEFAULT_OLLAMA_MODEL}",
                "Total Indexed Documents": len(files),
                "Chunk Size": 800,
                "Chunk Overlap": 150
            })
    else:
        st.info("No placement documents indexed yet. Use the sidebar to upload files or load sample JDs.")


# ==========================================
# TAB 4: Corrective RAG Workflow Explanation
# ==========================================
with tab_crag_explain:
    st.markdown("### 🔄 How Corrective RAG Works in This Application")
    st.markdown("""
    Unlike standard RAG systems that blindly generate answers from retrieved context regardless of quality, 
    **Corrective RAG (CRAG)** introduces a self-correcting evaluation loop:
    """)
    
    st.markdown("""
    ```
    [ Student Question ]
             │
             ▼
    [ Vector Retrieval (Qdrant) ] ──► Retrieve Top-k Placement Chunks
             │
             ▼
    [ LLM Document Grader ] ────────► Score: Are chunks relevant & sufficient?
             │
      ┌──────┴─────────────────────────┐
      │ (If Relevant)                  │ (If Insufficient / Weak)
      ▼                                ▼
    [ Direct Generation ]       [ Query Rewriter ] ──► Transform to keyword-rich query
      │                                │
      │                                ▼
      │                         [ Secondary Search / Web Fallback ]
      │                                │
      └────────────────────────────────┴──► [ Final Evidence-Backed Answer ]
    ```
    """)
    
    st.markdown("#### Key Technical Highlights for Student Interviews:")
    st.markdown("""
    - **Document Relevance Grading**: Uses an LLM evaluator to prevent hallucination when retrieved chunks don't answer the query.
    - **Query Rewriting**: Transforms underspecified student queries (e.g. *"What to study?"*) into targeted search terms (e.g. *"Software Engineer technical interview topics Data Structures System Design"*).
    - **Zero-Cloud Local Qdrant**: Uses `QdrantClient(location=":memory:")` for fast, lightweight in-memory vector storage without external cloud cluster latency or setup overhead.
    """)
