# RAG Systems — Easy Guide

## What is RAG?

**RAG = Retrieval-Augmented Generation**

RAG allows an AI model to first **find relevant information** from your own data and then use that information to generate an answer.

### Simple idea

```text
User Question
      ↓
Search your data
      ↓
Find relevant information
      ↓
Send information + question to LLM
      ↓
Generate Answer
```

Instead of asking the LLM to answer only from its training knowledge, RAG gives it useful context from your documents, database, website, etc.

---

# 1. Basic / Naive RAG

## What is it?

Naive RAG is the simplest RAG architecture.

```text
Documents
   ↓
Split into chunks
   ↓
Create embeddings
   ↓
Store in Vector Database
   ↓
User Question
   ↓
Similarity Search
   ↓
Relevant Chunks
   ↓
LLM
   ↓
Answer
```

## How to implement

### Step 1 — Load documents

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
documents = loader.load()
```

### Step 2 — Split documents

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
```

### Step 3 — Create embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Step 4 — Store in a vector database

```python
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(chunks, embeddings)
```

### Step 5 — Retrieve information

```python
question = "What is the refund policy?"

docs = db.similarity_search(question, k=4)

context = "\n\n".join(
    doc.page_content for doc in docs
)
```

### Step 6 — Send context to an LLM

```python
prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""
```

Then send the prompt to your LLM, such as Groq.

---

# 2. Advanced RAG

## What is it?

Advanced RAG improves basic RAG.

It can use:

- Better document chunking
- Query rewriting
- Metadata filtering
- Hybrid search
- Reranking
- Context compression

### Architecture

```text
User Question
      ↓
Query Rewriting
      ↓
Retriever
      ↓
Retrieve many documents
      ↓
Reranker
      ↓
Best documents
      ↓
LLM
      ↓
Answer
```

### Example

Instead of retrieving only 4 documents:

```python
docs = retriever.invoke(question)
```

retrieve 10–20 documents:

```python
docs = retriever.invoke(question, k=20)
```

Then use a reranker to select the most relevant documents.

### Why?

The first search may return documents that are similar but not actually useful.

Reranking improves the final context.

---

# 3. Hybrid RAG

## What is it?

Hybrid RAG combines:

1. **Keyword search**
2. **Vector/semantic search**

### Architecture

```text
                 ┌── Keyword Search
Question ────────┤
                 └── Vector Search
                         ↓
                    Combine Results
                         ↓
                       Rerank
                         ↓
                         LLM
```

### Why use it?

Vector search understands meaning.

Keyword search is very good for exact terms.

For example:

```text
HR-2026-17
```

A keyword search can find this exact ID very effectively.

### Basic idea

```python
vector_results = vector_db.similarity_search(
    question,
    k=10
)

keyword_results = bm25_retriever.invoke(
    question
)

combined_results = vector_results + keyword_results
```

Then rerank the combined results.

### Good for

- Company documents
- Legal documents
- Technical documentation
- Product catalogs
- Documents containing IDs, codes, names, etc.

---

# 4. Graph RAG

## What is it?

Graph RAG stores information as **entities and relationships**.

Example:

```text
Employee
   ↓ works_for
Company
   ↓ has_department
Quality
   ↓ manages
Project
   ↓ uses
Python
```

Normal vector search mainly finds similar text.

Graph RAG can answer questions that depend on relationships.

### Example question

> Which projects are managed by the Quality department?

### Architecture

```text
Question
    ↓
Entity / Relationship Extraction
    ↓
Knowledge Graph
    ↓
Graph Query
    ↓
Relevant Information
    ↓
LLM
    ↓
Answer
```

### Neo4j example

```cypher
MATCH (d:Department)-[:MANAGES]->(p:Project)
WHERE d.name = "Quality"
RETURN p
```

### Good for

- Company knowledge
- People and organizations
- Research
- Supply chains
- Complex relationships
- Knowledge bases

---

# 5. Agentic RAG

## What is it?

In normal RAG, the pipeline is mostly fixed.

In Agentic RAG, an **AI agent decides which tool to use**.

### Architecture

```text
                     ┌── Vector Search
                     │
User → AI Agent ─────┼── SQL Database
                     │
                     ├── Web Search
                     │
                     └── Other Tools
```

### Example

User asks:

> How many defects occurred last month and what does the SOP say about them?

The agent can decide:

```text
1. Query SQL database
2. Search SOP documents
3. Compare the information
4. Generate final answer
```

### Simple implementation idea

```python
tools = [
    vector_search_tool,
    sql_query_tool,
    web_search_tool
]

agent = create_agent(
    llm=groq_llm,
    tools=tools
)

response = agent.invoke({
    "input": question
})
```

### Good for

- AI assistants
- Enterprise copilots
- Data analysis
- Customer support
- Multi-tool applications

---

# 6. Multimodal RAG

## What is it?

Multimodal RAG can retrieve information from different types of data.

For example:

```text
Text
Images
Tables
Charts
PDFs
Audio
Videos
```

### Architecture

```text
                 ┌── Text
                 ├── Images
Documents ───────┼── Tables
                 ├── Charts
                 └── Other Data
                       ↓
                Multimodal Retrieval
                       ↓
                Multimodal LLM
                       ↓
                     Answer
```

### Example

A company PDF contains:

- Sewing defect descriptions
- Garment images
- Quality tables
- Charts

A user asks:

> Show me the image related to this defect.

Multimodal RAG can retrieve the relevant image along with its text information.

### Basic implementation idea

```python
text = extract_text(pdf)
tables = extract_tables(pdf)
images = extract_images(pdf)
```

Then create searchable representations for the extracted information.

---

# 7. Corrective RAG (CRAG)

## What is it?

CRAG checks whether the retrieved information is actually useful.

Normal RAG:

```text
Question
   ↓
Retrieve
   ↓
LLM
   ↓
Answer
```

CRAG:

```text
Question
   ↓
Retrieve
   ↓
Check Retrieval Quality
   ↓
Is it relevant?
   ↓
 ┌──────────────┴──────────────┐
 YES                            NO
 ↓                               ↓
Generate                    Search Again
Answer                           ↓
                              Generate
```

### Example

```python
docs = retriever.invoke(question)

score = evaluate_retrieval(
    question,
    docs
)

if score < 0.5:
    docs = web_search(question)

answer = generate_answer(
    question,
    docs
)
```

### Main advantage

It does not blindly trust the first retrieval result.

---

# 8. Self-RAG

## What is it?

Self-RAG allows the model to evaluate its own retrieval and answer.

### Architecture

```text
Question
   ↓
Should I retrieve?
   ↓
Retrieve
   ↓
Is the information useful?
   ↓
Generate Answer
   ↓
Is the answer supported?
   ↓
Final Answer
```

### Simple idea

```python
answer = llm.generate(
    question,
    context
)

evaluation = llm.evaluate(
    question=question,
    context=context,
    answer=answer
)

if not evaluation["supported"]:
    docs = retriever.invoke(question)

    answer = llm.generate(
        question,
        docs
    )
```

### Main goal

Reduce:

- Hallucinations
- Unsupported answers
- Irrelevant context

---

# 9. Modular RAG

## What is it?

Modular RAG divides the RAG system into separate components.

```text
Question
   ↓
Query Rewriter
   ↓
Retriever
   ↓
Reranker
   ↓
Context Processor
   ↓
LLM
   ↓
Answer
```

Each component can be changed independently.

### Example

```python
class RAGSystem:

    def rewrite_query(self, question):
        pass

    def retrieve(self, query):
        pass

    def rerank(self, documents):
        pass

    def generate(self, question, documents):
        pass

    def run(self, question):

        query = self.rewrite_query(question)

        docs = self.retrieve(query)

        docs = self.rerank(docs)

        return self.generate(question, docs)
```

### Main advantage

You can replace one component without rebuilding the complete system.

---

# 10. Comparison

| RAG Type | Main Idea | Best Use |
|---|---|---|
| Naive RAG | Basic retrieval + LLM | Learning RAG |
| Advanced RAG | Better retrieval | Production RAG |
| Hybrid RAG | Keyword + vector search | Exact terms + semantic search |
| Graph RAG | Relationships | Complex connected data |
| Agentic RAG | Agent chooses tools | AI assistants |
| Multimodal RAG | Text + images + tables | PDFs, images, charts |
| CRAG | Checks retrieval quality | Reducing bad retrieval |
| Self-RAG | Self-evaluation | Reducing hallucinations |
| Modular RAG | Independent modules | Flexible production systems |

---

# 11. Recommended Learning Order

Do not try to learn everything at once.

Follow this order:

```text
1. Basic RAG
      ↓
2. Chunking + Embeddings
      ↓
3. Vector Database
      ↓
4. Advanced RAG
      ↓
5. Hybrid RAG
      ↓
6. Reranking
      ↓
7. CRAG
      ↓
8. Agentic RAG
      ↓
9. Graph RAG
      ↓
10. Multimodal RAG
```

---

# 12. Recommended Stack for a GenAI Project

A practical stack can be:

```text
Frontend
   ↓
React
   ↓
FastAPI
   ↓
RAG Pipeline
   ↓
LangChain / LlamaIndex
   ↓
Vector Database
   ↓
Groq LLM
   ↓
Response
```

Possible components:

| Component | Options |
|---|---|
| Backend | FastAPI |
| LLM | Groq |
| Embeddings | Hugging Face |
| Vector DB | FAISS / Chroma / Qdrant |
| SQL DB | MySQL / PostgreSQL |
| Graph DB | Neo4j |
| Frontend | React |
| RAG Framework | LangChain / LlamaIndex |
| Reranker | BGE / Cross-Encoder |

---

# 13. One Important Concept

Remember this difference:

### LLM

```text
Question → LLM → Answer
```

The LLM relies mainly on its learned knowledge and the prompt.

### RAG

```text
Question
    ↓
Retrieve external information
    ↓
Question + Retrieved Information
    ↓
LLM
    ↓
Answer
```

So:

> **RAG = Retrieval + Generation**

The retrieval part finds useful information.

The generation part uses an LLM to produce the answer.

---

# 14. Best Project to Practice

Build one project and gradually upgrade it.

### Project: Company Document AI Assistant

Start with:

```text
PDF Upload
    ↓
Chunk PDF
    ↓
Create Embeddings
    ↓
FAISS
    ↓
Question
    ↓
Retrieve
    ↓
Groq
    ↓
Answer
```

Then upgrade it:

```text
Basic RAG
   ↓
Advanced RAG
   ↓
Hybrid Search
   ↓
Reranking
   ↓
CRAG
   ↓
Agentic RAG
   ↓
Graph RAG
```

This single project will help you understand most important RAG concepts for a **GenAI Engineer / SDE-1 interview**.

---

# 15. Quick Interview Definitions

**What is RAG?**

> RAG is a technique where relevant external information is retrieved and provided to an LLM as context before generating an answer.

**What is a vector database?**

> A vector database stores embeddings and allows similarity search over those embeddings.

**What is chunking?**

> Chunking means splitting large documents into smaller pieces so they can be embedded and retrieved effectively.

**What is reranking?**

> Reranking takes initially retrieved documents and reorders them according to their relevance to the user's query.

**What is Hybrid RAG?**

> Hybrid RAG combines keyword-based retrieval with semantic vector retrieval.

**What is Agentic RAG?**

> Agentic RAG allows an AI agent to decide which retrieval or external tools should be used to answer a question.

**What is Graph RAG?**

> Graph RAG uses entities and relationships in a knowledge graph to retrieve connected information.

**What is CRAG?**

> Corrective RAG evaluates retrieved information and performs additional retrieval when the initial results are poor.

**What is Multimodal RAG?**

> Multimodal RAG retrieves and uses multiple data types such as text, images, tables, and charts.

---

# Final Mental Model

Remember this:

```text
                    RAG
                     │
        ┌────────────┴────────────┐
        │                         │
   RETRIEVAL                  GENERATION
        │                         │
   ┌────┼────┐                    │
   │    │    │                    │
Vector Keyword Graph              LLM
Search Search  Search              │
   │    │    │                    │
   └────┴────┘                    │
        │                         │
        └──────── Context ────────┘
                     │
                     ↓
                   ANSWER
```

Start with **Naive RAG**, understand it completely, and then add one improvement at a time. That is much easier than trying to memorize all RAG architectures together.
