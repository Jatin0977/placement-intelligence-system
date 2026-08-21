import json
from typing import Dict, List, Any, Optional
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_community.tools import TavilySearchResults
from placement_intel.vector_store import PlacementVectorStore
from placement_intel.prompts import (
    EVALUATOR_PROMPT,
    REWRITER_PROMPT,
    GENERATOR_PROMPT,
    COMPARISON_PROMPT
)
from placement_intel.config import DEFAULT_OLLAMA_MODEL


class PlacementCRAGEngine:
    """Corrective RAG Engine for Placement Intelligence.
    
    Workflow Steps:
    1. Retrieve relevant document chunks from Vector Store.
    2. Evaluate chunk relevance (Grade: RELEVANT or INSUFFICIENT).
    3. If INSUFFICIENT: Transform/rewrite user query and perform secondary retrieval (or web search fallback).
    4. Generate final evidence-backed answer with document source citations.
    """

    def __init__(
        self,
        tavily_api_key: Optional[str] = None,
        model_name: str = DEFAULT_OLLAMA_MODEL
    ):
        self.tavily_api_key = tavily_api_key
        self.llm = ChatOllama(
            model=model_name,
            temperature=0
        )
        
        if tavily_api_key:
            self.web_search_tool = TavilySearchResults(
                tavily_api_key=tavily_api_key,
                k=3
            )
        else:
            self.web_search_tool = None

    def evaluate_document_relevance(self, question: str, documents: List[Document]) -> bool:
        """Evaluates whether the retrieved context contains sufficient relevant evidence.
        Uses ONE single LLM call for all retrieved document chunks."""
        if not documents:
            return False

        # Combine all retrieved chunks with document identifiers into one context block
        context_parts = []
        for idx, doc in enumerate(documents, 1):
            source_name = doc.metadata.get("source_name", doc.metadata.get("source", f"Document {idx}"))
            context_parts.append(f"[{source_name}]:\n{doc.page_content}")
        combined_context = "\n\n".join(context_parts)

        formatted_prompt = EVALUATOR_PROMPT.format(context=combined_context, question=question)
        
        try:
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()
            
            # Clean thinking tags if emitted by reasoning models
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1].strip()
                
            content_lower = content.lower()
            
            # Check JSON score or direct binary keyword
            if '"score": "yes"' in content_lower or '"score":"yes"' in content_lower:
                return True
            if '"score": "no"' in content_lower or '"score":"no"' in content_lower:
                return False
            
            if "yes" in content_lower and "no" not in content_lower:
                return True
            return False
        except Exception:
            return True  # Fallback gracefully to True

    def rewrite_query(self, question: str) -> str:
        """Transforms an underspecified or failed question into an optimized search query."""
        formatted_prompt = REWRITER_PROMPT.format(question=question)
        response = self.llm.invoke(formatted_prompt)
        content = response.content.strip()
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>")[-1].strip()
        return content

    def generate_answer(self, question: str, documents: List[Document]) -> str:
        """Generates grounded final answer using retrieved document evidence and metadata citations."""
        formatted_context_blocks = []
        
        for idx, doc in enumerate(documents, 1):
            source_name = doc.metadata.get("source_name", doc.metadata.get("source", "Placement Document"))
            page_info = f" (Page {doc.metadata['page'] + 1})" if "page" in doc.metadata else ""
            header = f"--- Document {idx}: [{source_name}{page_info}] ---"
            formatted_context_blocks.append(f"{header}\n{doc.page_content}")

        context_str = "\n\n".join(formatted_context_blocks)
        formatted_prompt = GENERATOR_PROMPT.format(context=context_str, question=question)
        
        response = self.llm.invoke(formatted_prompt)
        content = response.content.strip()
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>")[-1].strip()
        return content

    def execute_crag_pipeline(
        self,
        question: str,
        vector_store: PlacementVectorStore,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """Executes the full Corrective RAG state pipeline and returns answer, sources, and execution trace."""
        trace_logs = []
        
        # Step 1: Initial Retrieval
        trace_logs.append(f"🔍 **Step 1: Initial Retrieval** - Searching vector database for query: *'{question}'*")
        retrieved_docs = vector_store.search(question, k=top_k)
        
        if not retrieved_docs:
            trace_logs.append("⚠️ No document chunks found in vector database.")
            return {
                "answer": "No relevant placement documents have been indexed yet. Please upload placement documents or JDs first.",
                "sources": [],
                "trace_logs": trace_logs,
                "grade": "NO_DOCS",
                "rewritten_query": None
            }

        # Step 2: Relevance Evaluation (Single combined LLM call)
        trace_logs.append(f"📊 **Step 2: Context Relevance Evaluation** - Scoring overall sufficiency of {len(retrieved_docs)} retrieved chunks in 1 LLM call...")
        is_relevant = self.evaluate_document_relevance(question, retrieved_docs)
        
        final_docs = retrieved_docs
        rewritten_query = None

        if is_relevant:
            grade_status = "RELEVANT"
            trace_logs.append("✅ **Grade: RELEVANT** - Retrieved context contains sufficient placement evidence. Proceeding directly to answer generation.")
        else:
            grade_status = "INSUFFICIENT"
            trace_logs.append("⚠️ **Grade: INSUFFICIENT / WEAK** - Retrieved context lacks sufficient details. Triggering Query Transformation.")
            
            # Step 3: Query Rewriting (Corrective trigger)
            rewritten_query = self.rewrite_query(question)
            trace_logs.append(f"🔄 **Step 3: Query Rewriting** - Original: *'{question}'* ➡️ Transformed: *'{rewritten_query}'*")
            
            # Step 4: Re-retrieval
            secondary_docs = vector_store.search(rewritten_query, k=top_k)
            has_web_evidence = False

            # Optional Web Search Fallback if Tavily is configured and vector store evidence is weak
            if self.web_search_tool:
                try:
                    trace_logs.append("🌐 **Step 4b: Web Search Fallback** - Executing web search via Tavily for external context...")
                    web_results = self.web_search_tool.invoke({"query": rewritten_query or question})
                    web_text = "\n".join([res.get("content", "") for res in web_results if isinstance(res, dict)])
                    if web_text:
                        web_doc = Document(
                            page_content=f"Web Search Results:\n{web_text}",
                            metadata={"source_name": "External Web Search (Tavily)"}
                        )
                        final_docs = [web_doc]
                        has_web_evidence = True
                except Exception as e:
                    trace_logs.append(f"⚠️ Web search fallback skipped: {str(e)}")

            if not has_web_evidence:
                # Check if secondary retrieval produced genuinely different chunks or if database lacks relevant data
                initial_contents = set(doc.page_content.strip() for doc in retrieved_docs)
                secondary_contents = set(doc.page_content.strip() for doc in secondary_docs) if secondary_docs else set()
                
                # If secondary chunks are identical to rejected initial chunks or empty
                if not secondary_docs or secondary_contents.issubset(initial_contents):
                    trace_logs.append("⚠️ Re-retrieval confirmed that no relevant placement documents or criteria exist for this query.")
                    return {
                        "answer": f"Information regarding '{question}' was not found in the indexed placement documents.",
                        "sources": [],
                        "trace_logs": trace_logs,
                        "grade": "INSUFFICIENT",
                        "rewritten_query": rewritten_query,
                        "context_chunks": []
                    }
                else:
                    final_docs = secondary_docs
                    trace_logs.append(f"🔎 **Step 4: Re-Retrieval** - Retrieved {len(secondary_docs)} new chunks using rewritten query.")

        # Step 5: Answer Generation (Grounded synthesis)
        trace_logs.append("📝 **Step 5: Answer Generation** - Synthesizing grounded answer from retrieved context...")
        answer = self.generate_answer(question, final_docs)
        
        # Extract unique sources used
        sources = list(set([doc.metadata.get("source_name", "Placement Document") for doc in final_docs]))

        return {
            "answer": answer,
            "sources": sources,
            "trace_logs": trace_logs,
            "grade": grade_status,
            "rewritten_query": rewritten_query,
            "context_chunks": [doc.page_content for doc in final_docs]
        }

    def compare_companies(
        self,
        companies: List[str],
        focus_area: str,
        vector_store: PlacementVectorStore
    ) -> Dict[str, Any]:
        """Compares two or more companies based strictly on uploaded placement documents."""
        valid_companies = [c.strip() for c in companies if c and c.strip()]
        if not valid_companies:
            return {
                "answer": "Please specify at least two companies to compare.",
                "sources": [],
                "missing_companies": []
            }

        combined_context_blocks = []
        sources = []
        missing_companies = []

        for company in valid_companies:
            # Query vector store specifically for this company and focus area
            company_query = f"{company} {focus_area} requirements eligibility skills cutoff job description"
            docs = vector_store.search(company_query, k=3)
            
            # Check if any retrieved document actually pertains to this company
            company_docs = [
                doc for doc in docs 
                if company.lower() in doc.metadata.get("source_name", "").lower() 
                or company.lower() in doc.page_content.lower()
            ]
            
            if company_docs:
                for doc in company_docs:
                    src_name = doc.metadata.get("source_name", "Placement Document")
                    if src_name not in sources:
                        sources.append(src_name)
                    combined_context_blocks.append(f"[{src_name} - {company}]:\n{doc.page_content}")
            else:
                missing_companies.append(company)

        if not combined_context_blocks:
            return {
                "answer": "No uploaded placement documents found for any of the specified companies.",
                "sources": [],
                "missing_companies": missing_companies
            }

        context_str = "\n\n".join(combined_context_blocks)
        companies_str = ", ".join(valid_companies)
        
        formatted_prompt = COMPARISON_PROMPT.format(
            companies=companies_str,
            focus_area=focus_area,
            context=context_str
        )
        
        response = self.llm.invoke(formatted_prompt)
        content = response.content.strip()
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>")[-1].strip()
            
        return {
            "answer": content,
            "sources": sources,
            "missing_companies": missing_companies
        }

