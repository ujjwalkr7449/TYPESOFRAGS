"""PDF/Documents
     ↓
Chunking
     ↓
Embeddings
     ↓
Vector DB
     ↓
User Question
     ↓
Similarity Search
     ↓
Top-K Documents
     ↓
Groq LLM
     ↓
Answer"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create vector database
db = FAISS.from_documents(chunks, embeddings)

# 5. Retrieve
question = "What is the refund policy?"

docs = db.similarity_search(question, k=4)

context = "\n\n".join(doc.page_content for doc in docs)

# 6. Send context to LLM
prompt = f"""
Answer using only the following context:

{context}

Question:
{question}
"""

# Send prompt to Groq