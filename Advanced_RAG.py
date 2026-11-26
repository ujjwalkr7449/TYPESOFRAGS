question = "What is the company's leave policy?"

rewritten_query = """
company employee leave policy annual leave sick leave
"""

docs = vector_db.similarity_search(
    rewritten_query,
    k=10
)