import os
import argparse
from dotenv import load_dotenv, find_dotenv
from supabase import create_client
import fitz as pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv(find_dotenv())

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    try:
        parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            parts.append(f"\n--- Page {page_num + 1} ---\n")
            parts.append(page.get_text())
        return "".join(parts)
    finally:
        doc.close()


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def get_company(supabase, symbol: str):
    resp = supabase.table("companies").select("*").eq("symbol", symbol.upper()).limit(1).execute()
    return resp.data[0] if resp.data else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--company-symbol", required=True)
    parser.add_argument("--year", default=None)
    parser.add_argument("--report-type", default="annual_report")
    parser.add_argument("--file-name", default=None)
    parser.add_argument("--storage-bucket", default="annual-reports")
    args = parser.parse_args()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE key.")
    if not os.path.exists(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")

    supabase = create_client(supabase_url, supabase_key)
    company = get_company(supabase, args.company_symbol)
    if not company:
        raise SystemExit(f"Company not found: {args.company_symbol}")

    model = SentenceTransformer(MODEL_NAME)
    file_name = args.file_name or os.path.basename(args.pdf)
    storage_path = f"{args.storage_bucket}/{args.company_symbol.upper()}/{file_name}"

    with open(args.pdf, "rb") as f:
        content = f.read()
    supabase.storage.from_(args.storage_bucket).upload(
        path=storage_path,
        file=content,
        file_options={"content-type": "application/pdf", "upsert": "true"}
    )

    doc_resp = supabase.table("documents").insert({
        "company_id": company["id"],
        "scope": "company",
        "year": int(args.year) if args.year else None,
        "report_type": args.report_type,
        "file_name": file_name,
        "storage_path": storage_path,
        "source_url": None
    }).execute()
    if not doc_resp.data:
        raise SystemExit("Failed to insert document row.")
    document_id = doc_resp.data[0]["id"]

    text = extract_text_from_pdf(args.pdf)
    chunks = chunk_text(text)
    vectors = model.encode(chunks, normalize_embeddings=True).tolist() if chunks else []

    payload = []
    for i, chunk in enumerate(chunks):
        payload.append({
            "document_id": document_id,
            "company_id": company["id"],
            "scope": "company",
            "chunk_index": i,
            "content": chunk,
            "metadata": {"company_id": company["id"], "symbol": company["symbol"], "name": company["name"]},
            "embedding": vectors[i] if i < len(vectors) else None
        })

    if payload:
        chunk_resp = supabase.table("document_chunks").insert(payload).execute()
        if not chunk_resp.data:
            raise SystemExit("Failed to insert chunks.")

    print("UPLOAD OK")
    print("company_id:", company["id"])
    print("document_id:", document_id)
    print("storage_path:", storage_path)


if __name__ == "__main__":
    main()