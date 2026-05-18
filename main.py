import os
import shutil
import mlflow
import mlflow.pyfunc
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ─── Configuration ───────────────────────────────────────────────
FAISS_INDEX   = "faiss_index"
UPLOADS_DIR   = "uploads"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.1-8b-instant"
EMBEDDING_MODEL        = "all-MiniLM-L6-v2"          # default fallback
MLFLOW_TRACKING_URI    = os.environ.get("MLFLOW_TRACKING_URI", "http://3.19.56.68:5000")
MLFLOW_EXPERIMENT_NAME = "rag-embedding-experiments"
REGISTERED_MODEL_NAME  = "rag-embedding-model"


# Create uploads folder if it doesn't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)

PROMPT_TEMPLATE = PromptTemplate(
    template="""
You are an expert assistant. Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# ─── Request schema ──────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str


# ─── Lifespan — load embedding model once at startup ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Connect to MLflow
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print("MLflow connected.")
 
    # Try to load embedding model name from MLflow Production registry
    # Falls back to default if no model registered yet
    try:
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        embedding_model_name = latest[0].tags.get("embedding_model_name", EMBEDDING_MODEL)
        print(f"Embedding model loaded from MLflow registry: {embedding_model_name}")
    except Exception:
        embedding_model_name = EMBEDDING_MODEL
        print(f"No MLflow registry entry found. Using default: {embedding_model_name}")
 
    # Load embedding model
    print(f"Loading embedding model: {embedding_model_name}...")
    app.state.embedding_model_name = embedding_model_name
    app.state.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print("Embedding model loaded.")
    

    # Initialize Groq LLM
    app.state.llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0
    )
    print(f"Groq LLM initialized: {GROQ_MODEL}")
    app.state.vectorstore = None  # No index until a PDF is uploaded

    print("App ready.")

    yield

    print("Shutting down...")


# ─── App ─────────────────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="RAG Pipeline API",
    description="Upload a PDF and ask questions about it using LLaMA 3 via Groq.",
    version="1.0.0"
)


# ─── Health check ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ─── POST /upload ────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, chunks it, embeds it, and saves the FAISS index.
    Run this once per document.
    """

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded PDF to disk
    pdf_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    print(f"PDF saved to {pdf_path}")

    # Start timer for MLflow metric
    start_time = time.time()

    # Load PDF
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"Loaded {len(pages)} pages.")

    # Split into chunks
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks.")


    # Embed and save FAISS index
    print("Embedding chunks and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, app.state.embeddings)
    vectorstore.save_local(FAISS_INDEX)

    # Store in app state so /ask can use it immediately
    app.state.vectorstore = vectorstore
    print("FAISS index built and saved.")

    # Log this indexing run to MLflow
    index_time = round(time.time() - start_time, 2)
    with mlflow.start_run(run_name=f"index-{file.filename}"):
        mlflow.log_param("embedding_model", app.state.embedding_model_name)
        mlflow.log_param("filename",        file.filename)
        mlflow.log_param("chunk_size",      CHUNK_SIZE)
        mlflow.log_param("chunk_overlap",   CHUNK_OVERLAP)
        mlflow.log_metric("pages",          len(pages))
        mlflow.log_metric("chunks",         len(chunks))
        mlflow.log_metric("index_time_sec", index_time)
    print(f"MLflow run logged. Index time: {index_time}s")

    return {
        "message"   : "PDF indexed successfully.",
        "filename"  : file.filename,
        "pages"     : len(pages),
        "chunks"    : len(chunks),
        "index_time_sec" : index_time,
        "embedding_model": app.state.embedding_model_name
    }


# ─── POST /ask ───────────────────────────────────────────────────
@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Accepts a question and returns an answer grounded in the uploaded document.
    Requires a PDF to have been uploaded first via /upload.
    """

    # Check if an index exists — in memory first, then on disk
    if app.state.vectorstore is None:
        if os.path.exists(FAISS_INDEX):
            print("Loading FAISS index from disk...")
            app.state.vectorstore = FAISS.load_local(
                FAISS_INDEX,
                app.state.embeddings,
                allow_dangerous_deserialization=True
            )
            print("FAISS index loaded from disk.")
        else:
            raise HTTPException(
                status_code=400,
                detail="No document indexed yet. Please upload a PDF via POST /upload first."
            )

    # Initialize Groq LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0
    )

    # Build retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=app.state.llm,
        chain_type="stuff",
        retriever=app.state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE},
        return_source_documents=True
    )

    # Get answer
    response = qa_chain.invoke({"query": request.question})
    answer_text = response["result"].strip().replace("\n\n", " ").replace("\n", " ")
    
    # Extract source pages
    source_pages = list(set([
        doc.metadata.get("page", "unknown")
        for doc in response["source_documents"]
    ]))
    source_pages.sort()
    
    return {
        "question"    : request.question,
        "answer"      : answer_text,
        "source_pages": source_pages
    }