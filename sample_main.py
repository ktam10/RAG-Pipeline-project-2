from langchain_community.document_loaders import PyPDFLoader

#Path to K8s document
pdf_path = "C:/Users/kaust/Downloads/Concepts_Kubernetes.pdf"

#Load the PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

#Print total number of pages
print(f"Total Number of Pages loaded: {len(documents)}")

#Print First 500 character on pg 1
if documents:
    print("\nFirst 500 characters on page 1:\n")
    print(documents[4].page_content[:500])
else:
    print("No pages were loaded")