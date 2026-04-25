import os
import tempfile
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from rag_system import RAGSystem

load_dotenv(find_dotenv())

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_OK = bool(SUPABASE_URL and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")))

st.set_page_config(
    page_title="Mubashir & Hassan | RAG Chat System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --bg: #ffffff;
    --card: #f8fafc;
    --text: #111827;
    --muted: #6b7280;
    --border: #e5e7eb;
    --accent: #2563eb;
    --accent2: #1d4ed8;
}
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
}
.main-title {
    color: var(--accent2);
    text-align: center;
    font-size: 2.35rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    color: var(--muted);
    font-size: 1rem;
    margin-bottom: 1rem;
}
.card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.1rem 1.15rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}
.assistant-response {
    background: #0f172a;
    color: #ffffff;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0 1rem 0;
    line-height: 1.6;
    white-space: pre-wrap;
}
.user-response {
    background: #eff6ff;
    color: #0f172a;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0 1rem 0;
    line-height: 1.6;
    white-space: pre-wrap;
}
.card-soft {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.status-pill {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: #eff6ff;
    color: #1d4ed8;
    font-weight: 600;
    margin-right: 0.45rem;
    margin-bottom: 0.35rem;
}
.small-note {
    color: var(--muted);
    font-size: 0.9rem;
}
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 800;
    color: #0f172a;
}
hr {
    border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False
if "mode" not in st.session_state:
    st.session_state.mode = "User Document Mode"
if "current_company" not in st.session_state:
    st.session_state.current_company = None
if "current_document_id" not in st.session_state:
    st.session_state.current_document_id = None
if "company_options" not in st.session_state:
    st.session_state.company_options = []
if "last_company_name" not in st.session_state:
    st.session_state.last_company_name = None


def initialize_rag_system():
    try:
        embedding_model = st.session_state.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        llm_model = st.session_state.get("llm_model", "openai/gpt-3.5-turbo")
        temperature = st.session_state.get("temperature", 0.3)
        base_url = st.session_state.get("base_url", "https://openrouter.ai/api/v1")
        st.session_state.rag_system = RAGSystem(
            model_name=embedding_model,
            llm_model=llm_model,
            temperature=temperature,
            base_url=base_url
        )
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False


def process_pdf(uploaded_file):
    try:
        if st.session_state.rag_system is None:
            if not initialize_rag_system():
                return False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        rag_system = st.session_state.rag_system

        with st.spinner("Extracting text from PDF..."):
            text = rag_system.extract_text_from_pdf(tmp_path)

        if not text.strip():
            st.error("No text could be extracted from the PDF.")
            os.unlink(tmp_path)
            return False

        st.success(f"Extracted {len(text)} characters from PDF")

        with st.spinner("Chunking and embedding text..."):
            chunk_size = st.session_state.get("chunk_size", 1000)
            chunk_overlap = st.session_state.get("chunk_overlap", 200)
            chunks_data = rag_system.chunk_and_embed_document(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        st.success(f"Created {len(chunks_data)} text chunks")

        storage_path = f"user-documents/{uploaded_file.name}"
        with st.spinner("Uploading PDF to Supabase Storage..."):
            uploaded_ok = rag_system.upload_pdf_to_storage("user-documents", tmp_path, storage_path)

        if not uploaded_ok:
            st.error("Failed to upload PDF to Supabase storage.")
            os.unlink(tmp_path)
            return False

        document_id = rag_system.insert_document_record(
            scope="user",
            report_type="annual_report",
            file_name=uploaded_file.name,
            storage_path=storage_path
        )

        if not document_id:
            st.error("Failed to insert document record into Supabase.")
            os.unlink(tmp_path)
            return False

        with st.spinner("Saving chunks to database..."):
            inserted_cnt = rag_system.insert_chunks_record(
                document_id=document_id,
                scope="user",
                chunks_data=chunks_data
            )

        if inserted_cnt is None:
            st.error("Failed to insert chunks into Supabase.")
            os.unlink(tmp_path)
            return False

        st.session_state.current_document_id = document_id
        st.session_state.vector_store_loaded = True
        st.session_state.current_pdf = uploaded_file.name
        st.success(f"Document processed and saved to Supabase. Inserted {inserted_cnt} chunks.")
        os.unlink(tmp_path)
        return True

    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False


def load_companies():
    try:
        if st.session_state.rag_system is None:
            if not initialize_rag_system():
                return []
        resp = st.session_state.rag_system.supabase.table("companies").select("id, symbol, name, sector").order("name").execute()
        return resp.data or []
    except Exception:
        return []


def select_company_ui():
    companies = load_companies()
    st.session_state.company_options = companies

    if not companies:
        st.info("No companies found in the database yet.")
        return

    options = [f"{c['name']} ({c['symbol']})" for c in companies]
    current_label = None
    if st.session_state.current_company:
        current_label = f"{st.session_state.current_company['name']} ({st.session_state.current_company['symbol']})"
    default_index = options.index(current_label) if current_label in options else 0

    selected_label = st.selectbox(
        "Choose a company",
        options,
        index=default_index
    )

    selected_company = companies[options.index(selected_label)]
    if st.session_state.current_company != selected_company:
        st.session_state.current_company = selected_company
        st.session_state.current_document_id = None
        st.session_state.chat_history = []

    st.markdown(
        f"<div class='status-pill'>Selected: {selected_company['name']} ({selected_company['symbol']})</div>",
        unsafe_allow_html=True
    )
    if selected_company.get("sector"):
        st.markdown(
            f"<div class='small-note'>Sector: {selected_company['sector']}</div>",
            unsafe_allow_html=True
        )


def chat_interface():
    st.subheader("Chat with Your Document")

    if st.session_state.mode == "Company Research Mode":
        if not st.session_state.current_company:
            st.info("Please select a company first.")
            return
    else:
        if not st.session_state.vector_store_loaded:
            st.info("Please upload and process a PDF first.")
            return

    if st.session_state.chat_history:
        st.write("**Chat History:**")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-response"><strong>You:</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="assistant-response"><strong>Assistant:</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True
                )
                if message.get("sources"):
                    with st.expander("Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**Source {i}:**\n{source.page_content}")

    question = st.text_area(
        "Ask a question about the document:",
        placeholder="What is this document about?",
        key="user_input",
        height=120 if len(st.session_state.get("user_input", "")) < 180 else 220
    )

    submit_button = st.button("Ask", key="submit_button")

    if submit_button and question:
        with st.spinner("Generating response..."):
            try:
                if st.session_state.rag_system is None:
                    if not initialize_rag_system():
                        return

                if st.session_state.mode == "Company Research Mode":
                    result = st.session_state.rag_system.query_company_documents(
                        st.session_state.current_company["id"], question,
                        top_k=st.session_state.get("k_results", 3)
                    )
                else:
                    result = st.session_state.rag_system.query_user_document(
                        st.session_state.current_document_id, question,
                        top_k=st.session_state.get("k_results", 3)
                    )

                answer = result["answer"]
                sources = result.get("source_documents", [])

                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    st.markdown("---")
    st.caption("Built by Mubashir & Hassan — Mubashir & Hassan can never make mistakes so don't dare to verify the responses.")


def main():
    st.markdown("<h1 class='main-title'>Mubashir & Hassan RAG Chat System</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Created for Ma'am Madiha by Mubashir & Hassan with Love and passion 💔</div>", unsafe_allow_html=True)

    if st.session_state.get("rag_system") is not None:
        supabase_connected = getattr(st.session_state.rag_system, "supabase_ok", False)
    else:
        supabase_connected = SUPABASE_OK

    if not supabase_connected:
        st.markdown(
            """
            <div class="card-soft">
            <strong>Supabase is not connected.</strong><br>
            Check your .env file and restart the app.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Project Controls</div>", unsafe_allow_html=True)

        mode = st.selectbox(
            "Mode",
            ["User Document Mode", "Company Research Mode"],
            index=0 if st.session_state.mode == "User Document Mode" else 1
        )
        st.session_state.mode = mode

        st.markdown("---")
        st.markdown(f"<div class='status-pill'>Current Mode: {mode}</div>", unsafe_allow_html=True)
        if mode == "Company Research Mode" and st.session_state.current_company:
            st.markdown(
                f"<div class='status-pill'>Company: {st.session_state.current_company['symbol']}</div>",
                unsafe_allow_html=True
            )

        with st.expander("Advanced Settings"):
            st.session_state["embedding_model"] = st.selectbox(
                "Embedding Model",
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "sentence-transformers/paraphrase-MiniLM-L6-v2"
                ],
                index=0
            )

            st.session_state["llm_model"] = st.selectbox(
                "LLM Model",
                ["openai/gpt-3.5-turbo", "openai/gpt-4", "anthropic/claude-3-haiku", "meta-llama/llama-3-8b-instruct"],
                index=0
            )

            st.session_state["temperature"] = st.slider(
                "Temperature",
                0.0, 1.0, 0.3, 0.1
            )

            st.session_state["chunk_size"] = st.slider(
                "Chunk Size (characters)",
                200, 2000, 1000, 100
            )

            st.session_state["chunk_overlap"] = st.slider(
                "Chunk Overlap (characters)",
                0, 500, 200, 50
            )

            st.session_state["k_results"] = st.slider(
                "Number of Retrieved Chunks",
                1, 10, 3
            )

        if st.button("Initialize RAG System"):
            if not initialize_rag_system():
                st.error("RAG system failed to initialize.")
            else:
                st.success("RAG system initialized.")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.vector_store_loaded = False
            st.session_state.current_company = None
            st.session_state.current_document_id = None
            st.success("Chat history cleared!")

    tab1, tab2 = st.tabs(["Upload & Chat", "About"])

    with tab1:
        if st.session_state.mode == "Company Research Mode":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Company Selection")
            if not st.session_state.rag_system:
                if not initialize_rag_system():
                    st.stop()
            select_company_ui()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            if st.session_state.current_company:
                docs = st.session_state.rag_system.get_documents_for_company(
                    st.session_state.current_company["id"]
                )
                if docs:
                    st.write("**Available Company Reports:**")
                    for d in docs:
                        st.write(f"- {d.get('file_name')} (year: {d.get('year')})")
                else:
                    st.info("No company reports found. Upload company reports using the backend script.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            chat_interface()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([1.05, 1.4], gap="large")

            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Upload PDF")
                uploaded_file = st.file_uploader(
                    "Choose a PDF file",
                    type="pdf",
                    help="Upload a PDF document to create a searchable knowledge base"
                )

                if uploaded_file:
                    if not st.session_state.rag_system:
                        if not os.environ.get("OPENAI_API_KEY"):
                            st.error("Please set your API key in the sidebar first.")
                        else:
                            if not initialize_rag_system():
                                st.stop()

                    if st.button("Process PDF", key="process_pdf"):
                        process_pdf(uploaded_file)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                chat_interface()
                st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class='card'>
        <h3>About RAG Chat System</h3>
        <p>This app lets you upload your own PDFs or chat with selected PSX company reports stored in Supabase.</p>
        <ul>
            <li>Upload PDFs</li>
            <li>Store files in Supabase Storage</li>
            <li>Store metadata and chunks in Supabase Postgres</li>
            <li>Search and answer questions from your documents</li>
        </ul>
        <p><strong>Built by Mubashir & Hassan</strong></p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()