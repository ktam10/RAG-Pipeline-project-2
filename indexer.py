import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─── Configuration ───────────────────────────────────────────────
PDF_PATH   = "C:/Users/kaust/Downloads/Concepts_Kubernetes.pdf"   # Replace with your actual PDF filename
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FAISS_INDEX   = "faiss_index"

# ─── Component 1: Load PDF ───────────────────────────────────────
print("=" * 50)
print("COMPONENT 1: Loading PDF...")
print("=" * 50)

loader = PyPDFLoader(PDF_PATH)
pages  = loader.load()

print(f"Total pages loaded : {len(pages)}")

# ─── Component 2: Split text into chunks ─────────────────────────
print("\n" + "=" * 50)
print("COMPONENT 2: Splitting text into chunks...")
print("=" * 50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_documents(pages)
print(f"Total chunks created : {len(chunks)}")

# ─── Component 3: Embeddings + FAISS Vector Store ────────────────
print("\n" + "=" * 50)
print("COMPONENT 3: Creating embeddings and vector store and saving FAISS index...")
print("This will take a few minutes for a large document...")
print("=" * 50)
 
# Step 1 — Load the embedding model (downloads ~80MB first time only)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(FAISS_INDEX)
print(f"Vector store saved to '{FAISS_INDEX}' folder.")

print(f"\nDone. FAISS index saved to '{FAISS_INDEX}' folder.")
print("You never need to run this script again unless your document changes.")
