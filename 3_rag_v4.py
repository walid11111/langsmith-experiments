


# pip install -U langchain langchain-openai langchain-community 
# faiss-cpu pypdf python-dotenv langsmith langchain-groq langchain-huggingface

import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ----------------- ENV -----------------
os.environ['LANGCHAIN_PROJECT'] = 'Full-Rag-app'
load_dotenv()

PDF_PATH = "islr.pdf"
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)


# =========================================================
# ----------------- HELPERS (TRACED) ----------------------
# =========================================================

@traceable(name="load_pdf", tags=["load pdf"])
def load_pdf(path: str):
    return PyPDFLoader(path).load()


@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


@traceable(name="build_vectorstore")
def build_vectorstore(splits, embed_model_name: str):
    emb = HuggingFaceEmbeddings(model=embed_model_name)
    return FAISS.from_documents(splits, emb)


# =========================================================
# ----------------- FINGERPRINT LOGIC ---------------------
# =========================================================

def _file_fingerprint(path: str) -> dict:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {
        "sha256": h.hexdigest(),
        "size": p.stat().st_size,
        "mtime": int(p.stat().st_mtime)
    }


def _index_key(pdf_path: str, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()


# =========================================================
# ----------------- INDEX BUILD / LOAD --------------------
# =========================================================

@traceable(name="load_index", tags=["index"], run_type="chain")
def load_index_run(index_dir: Path, embed_model_name: str):
    emb = HuggingFaceEmbeddings(model=embed_model_name)
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )


@traceable(name="build_index", tags=["index"], run_type="chain")
def build_index_run(
    pdf_path: str,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embed_model_name: str
):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits, embed_model_name)

    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))

    # Save metadata with fingerprint
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "pdf_fingerprint": _file_fingerprint(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
    }, indent=2))

    return vs


def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    meta_file = index_dir / "meta.json"

    rebuild_reason = None

    if force_rebuild:
        rebuild_reason = "force_rebuild=True"

    elif not index_dir.exists():
        rebuild_reason = "index_not_found"

    elif not meta_file.exists():
        rebuild_reason = "meta_missing"

    else:
        stored_meta = json.loads(meta_file.read_text())
        current_fingerprint = _file_fingerprint(pdf_path)

        if stored_meta.get("pdf_fingerprint") != current_fingerprint:
            rebuild_reason = "pdf_changed"

    if rebuild_reason:
        print(f"\n⚡ Rebuilding index because: {rebuild_reason}")
        return build_index_run(
            pdf_path,
            index_dir,
            chunk_size,
            chunk_overlap,
            embed_model_name
        )
    else:
        print("\n🚀 Loading existing index (fast mode)")
        return load_index_run(index_dir, embed_model_name)


# =========================================================
# ----------------- LLM + PROMPT --------------------------
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# =========================================================
# ----------------- MAIN PIPELINE -------------------------
# =========================================================

@traceable(name="pdf_rag_full_run", run_type="chain", tags=["rag"])
def setup_pipeline_and_query(
    pdf_path: str,
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    force_rebuild: bool = False,
):
    vectorstore = load_or_build_index(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embed_model_name,
        force_rebuild
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | llm | StrOutputParser()

    return chain.invoke(
        question,
        config={
            "run_name": "pdf_rag_query",
            "tags": ["qa"],
            "metadata": {"k": 4}
        }
    )


# =========================================================
# ----------------- CLI LOOP ------------------------------
# =========================================================

if __name__ == "__main__":
    print("📚 PDF RAG Ready (Smart Cache Enabled)")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Q: ").strip()
        if q.lower() in ["exit", "quit"]:
            break

        answer = setup_pipeline_and_query(PDF_PATH, q)
        print("\nA:", answer)
        print("-" * 60)
