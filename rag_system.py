import os
import logging
import requests

import fitz as pymupdf
from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv(find_dotenv())

logger = logging.getLogger("rag_system")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RAGSystem:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        # 1. Configured default model to the verified free Nvidia endpoint
        llm_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free",
        temperature: float = 0.2,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.llm_model = llm_model

        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            logger.exception("Embeddings init failed")
            raise RuntimeError(f"Embedding initialization failed: {e}")

        # 2. Safely grab credentials prioritizing Streamlit Secrets over local .env variables
        try:
            import streamlit as st
            self.api_key = st.secrets.get("OPENROUTER_API_KEY") or st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            supabase_url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
            supabase_key = (
                st.secrets.get("SUPABASE_SERVICE_ROLE_KEY") 
                or st.secrets.get("SUPABASE_ANON_KEY") 
                or os.getenv("SUPABASE_SERVICE_ROLE_KEY") 
                or os.getenv("SUPABASE_ANON_KEY")
            )
        except Exception:
            # If Streamlit is not initialized, default cleanly back to standard os environment lookups
            self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY or OPENAI_API_KEY in configurations.")

        if not supabase_url or not supabase_key:
            raise ValueError("Missing Supabase config. Please provide SUPABASE_URL and authentication keys.")

        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key

        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            self.supabase_ok = True
            self.supabase_via_rest = False
            logger.info("Supabase client initialized successfully.")
        except Exception as e:
            logger.exception("Supabase client init failed")
            raise RuntimeError(f"Supabase client initialization failed: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        doc = pymupdf.open(pdf_path)
        try:
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            return text
        except Exception:
            logger.exception("extract_text_from_pdf failed")
            return ""
        finally:
            doc.close()

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"chunk_id": i}) for i, chunk in enumerate(chunks)]

    def chunk_and_embed_document(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200, metadata: dict = None):
        if not text:
            return []

        docs = self.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = [d.page_content for d in docs]

        try:
            embeddings = self.embeddings.embed_documents(texts)
        except Exception:
            logger.exception("embed_documents failed")
            embeddings = [None] * len(texts)

        chunks = []
        for i, doc in enumerate(docs):
            chunks.append({
                "index": i,
                "content": doc.page_content,
                "embedding": embeddings[i] if i < len(embeddings) else None,
                "metadata": metadata or {}
            })
        return chunks

    def _openrouter_chat(self, messages, model=None, temperature=None):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Mubashir & Hassan RAG Chat System"
        }
        payload = {
            "model": model or self.llm_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # 3. Updated the default value to None here so it safely reads self.llm_model inside the body
    def _openrouter_web_search(self, question: str, max_results: int = 5, model: str = None):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with web search access. "
                    "Use the web search results to answer accurately and cite sources with markdown links when possible."
                )
            },
            {"role": "user", "content": question}
        ]
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Mubashir & Hassan RAG Chat System"
        }
        payload = {
            "model": model or self.llm_model,
            "messages": messages,
            "temperature": 0.2,
            "plugins": [{"id": "web", "max_results": max_results}]
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=150)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def query_general_question(self, question: str):
        prompt = f"""You are a helpful general-purpose assistant.

Answer the user's question clearly and directly using your own knowledge.
If the question is ambiguous, explain the most likely interpretation.
If you are uncertain, say so briefly.

Question: {question}

Answer:"""
        try:
            answer = self._openrouter_chat([{"role": "user", "content": prompt}])
            return {"answer": answer, "source_documents": []}
        except Exception as e:
            logger.exception("General QA generation failed")
            return {"answer": f"Failed to generate answer: {str(e)}", "source_documents": []}

    def query_web_search(self, question: str):
        prompt = f"""Answer the following using current web information only when needed.

Question: {question}

Return a clear answer with concise supporting details."""
        try:
            # 4. Corrected to pass the chosen instance free model
            answer = self._openrouter_web_search(prompt, max_results=5, model=self.llm_model)
            return {"answer": answer, "source_documents": []}
        except Exception as e:
            logger.exception("Web search failed")
            return {"answer": f"Web search failed: {str(e)}", "source_documents": []}

    def query_financial_analysis(self, question: str):
        prompt = f"""You are a finance and stock market analyst.

Use current market/news context when needed.
Explain:
1. what the news means,
2. likely impact on the broader market,
3. likely impact on relevant sectors/stocks,
4. whether the effect is bullish, bearish, or mixed,
5. any uncertainty.

Question:
{question}

Answer in a structured, investor-friendly way."""
        try:
            # 5. Corrected to pass the chosen instance free model
            answer = self._openrouter_web_search(prompt, max_results=5, model=self.llm_model)
            return {"answer": answer, "source_documents": []}
        except Exception as e:
            logger.exception("Financial analysis failed")
            return {"answer": f"Financial analysis failed: {str(e)}", "source_documents": []}

    def _build_answer(self, context: str, question: str) -> str:
        prompt = f"""You are a helpful assistant that answers questions ONLY based on the provided context.

IMPORTANT RULES:
1. ONLY answer using the provided context.
2. If the answer cannot be found in the context, say: "I cannot answer this question based on the provided documents."
3. Do NOT use external knowledge.
4. If the question is unclear, ask for clarification.

Context:
{context}

Question: {question}

Answer:"""
        try:
            return self._openrouter_chat([{"role": "user", "content": prompt}])
        except Exception:
            logger.exception("LLM generation failed")
            return "Failed to generate answer."

    def get_company_by_symbol_or_name(self, identifier: str):
        try:
            if not identifier:
                return None

            ident = identifier.strip()
            symbol = ident.upper()

            resp = self.supabase.table("companies").select("*").eq("symbol", symbol).limit(1).execute()
            if resp.data:
                return resp.data[0]

            resp = self.supabase.table("companies").select("*").ilike("name", f"%{ident}%").limit(1).execute()
            if resp.data:
                return resp.data[0]

            return None
        except Exception:
            logger.exception("get_company_by_symbol_or_name failed")
            return None

    def upload_pdf_to_storage(self, bucket_name: str, file_path: str, storage_path: str):
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.supabase.storage.from_(bucket_name).upload(
                path=storage_path,
                file=content,
                file_options={"content-type": "application/pdf", "upsert": "true"}
            )
            logger.info("Uploaded to storage: %s", storage_path)
            return True
        except Exception:
            logger.exception("upload_pdf_to_storage failed")
            return False

    def insert_document_record(
        self,
        user_id=None,
        company_id=None,
        scope="user",
        year=None,
        report_type="annual_report",
        file_name="",
        storage_path="",
        source_url=None
    ):
        try:
            data = {
                "user_id": user_id,
                "company_id": company_id,
                "scope": scope,
                "year": year,
                "report_type": report_type,
                "file_name": file_name,
                "storage_path": storage_path,
                "source_url": source_url
            }
            resp = self.supabase.table("documents").insert(data).execute()
            if getattr(resp, "data", None):
                return resp.data[0].get("id")
            return None
        except Exception as e:
            logger.exception(f"insert_document_record failed: {e}")
            raise

    def insert_chunks_record(self, document_id, user_id=None, company_id=None, scope="user", chunks_data=None):
        if not chunks_data:
            return 0

        try:
            payload = []
            for chunk in chunks_data:
                payload.append({
                    "document_id": document_id,
                    "user_id": user_id,
                    "company_id": company_id,
                    "scope": scope,
                    "chunk_index": int(chunk.get("index", 0)),
                    "content": chunk.get("content"),
                    "metadata": chunk.get("metadata", {}),
                    "embedding": chunk.get("embedding")
                })

            resp = self.supabase.table("document_chunks").insert(payload).execute()
            if getattr(resp, "data", None):
                return len(resp.data)
            return 0
        except Exception as e:
            logger.exception(f"insert_chunks_record failed: {e}")
            raise

    def query_user_document(self, document_id: int, question: str, top_k: int = 3):
        try:
            question_emb = self.embeddings.embed_query(question)
        except Exception:
            return {"answer": "Embedding failed.", "source_documents": []}

        try:
            response = self.supabase.rpc("similar_chunks", {
                "query_embedding": question_emb,
                "document_id_filter": document_id,
                "company_id_filter": None,
                "scope_filter": "user",
                "match_count": top_k
            }).execute()
            chunks = response.data or []
        except Exception:
            logger.exception("similar_chunks RPC failed")
            return {"answer": "Search failed (RPC error).", "source_documents": []}

        if not chunks:
            return {"answer": "No relevant information found in the document.", "source_documents": []}

        context = "\n\n".join([chunk["content"] for chunk in chunks])
        answer = self._build_answer(context, question)

        source_docs = [
            Document(page_content=chunk["content"], metadata=chunk.get("metadata", {}))
            for chunk in chunks
        ]
        return {"answer": answer, "source_documents": source_docs}

    def query_company_documents(self, company_id: int, question: str, top_k: int = 3):
        try:
            question_emb = self.embeddings.embed_query(question)
        except Exception:
            return {"answer": "Embedding failed.", "source_documents": []}

        try:
            response = self.supabase.rpc("similar_chunks", {
                "query_embedding": question_emb,
                "document_id_filter": None,
                "company_id_filter": company_id,
                "scope_filter": "company",
                "match_count": top_k
            }).execute()
            chunks = response.data or []
        except Exception:
            logger.exception("similar_chunks RPC failed")
            return {"answer": "Search failed (RPC error).", "source_documents": []}

        if not chunks:
            return {"answer": "No relevant information found in the company documents.", "source_documents": []}

        context = "\n\n".join([chunk["content"] for chunk in chunks])
        answer = self._build_answer(context, question)

        source_docs = [
            Document(page_content=chunk["content"], metadata=chunk.get("metadata", {}))
            for chunk in chunks
        ]
        return {"answer": answer, "source_documents": source_docs}

    def get_documents_for_company(self, company_id: int):
        try:
            resp = self.supabase.table("documents").select("*").eq("company_id", company_id).eq("scope", "company").order("year", desc=True).execute()
            return resp.data or []
        except Exception:
            logger.exception("get_documents_for_company failed")
            return []

    def get_documents_for_user(self, user_id: int):
        try:
            resp = self.supabase.table("documents").select("*").eq("user_id", user_id).eq("scope", "user").order("id", desc=True).execute()
            return resp.data or []
        except Exception:
            logger.exception("get_documents_for_user failed")
            return []
