question = "What is the company's leave policy?"

rewritten_query = """
company employee leave policy annual leave sick leave
"""

docs = vector_db.similarity_search(
    rewritten_query,
    k=10
)

# Then rerank the 10 results
reranked_docs = reranker(
    question,
    docs)

top_docs = reranked_docs[:4]