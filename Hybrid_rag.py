vector_results = vector_db.similarity_search(
    question,
    k=10
)
keyword_results = bm25_retriever.invoke(
    question
)
