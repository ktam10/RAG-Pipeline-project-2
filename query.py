import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

FAISS_INDEX   = "faiss_index"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

print("\nStep 1: Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

print("Loading FAISS index from disk...")
vectorstore = FAISS.load_local(
    FAISS_INDEX,
    embeddings,
    allow_dangerous_deserialization=True
)
print("FAISS index loaded. Ready to answer questions.\n")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ─── Step 4: Build prompt template
prompt_template = """
You are an expert assistant. Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ─── Step 6: Interactive question loop ───────────────────────────
# Ask as many questions as you want without reloading anything
print("=" * 50)
print("RAG Pipeline Ready. Type 'exit' to quit.")
print("=" * 50)
 
while True:
    question = input("\nAsk a question: ").strip()
 
    if question.lower() == "exit":
        print("Exiting.")
        break
 
    if not question:
        continue
 
    response = qa_chain.invoke({"query": question})

print("\n" + "=" * 50)
print(f"Answer:\n{response['result']}")
print("=" * 50)
print("\nSource pages used:")
for doc in response['source_documents']:
    print(f"  - Page {doc.metadata['page']}")
print("=" * 50)