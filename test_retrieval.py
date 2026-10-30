from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).parent
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

def query_rag(query_text: str, top_k: int = 2):
    print(f"\n🔍 Query: '{query_text}'")
    print("-" * 50)
    
    # Load embeddings and FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local(
        str(VECTOR_STORE_DIR), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Retrieve top matches
    results = vector_db.similarity_search(query_text, k=top_k)
    
    for idx, doc in enumerate(results, 1):
        print(f"--- Result {idx} ---")
        print(doc.page_content)
        print()

if __name__ == "__main__":
    # Test queries based on your weather files
    query_rag("What is the weather forecast for Lahore?")
    query_rag("What are the expected temperatures in Lahore?")