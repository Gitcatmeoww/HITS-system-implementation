from backend.app.table_representation.openai_client import OpenAIClient
from backend.app.hyse.hypo_schema_search import cos_sim_search
import logging

# Initialize OpenAI client
openai_client = OpenAIClient()

# Evaluate semantic keywords search & semantic task search against HySE
def semantic_search(query, k):
    try:
        # Step 1: Generate embedding for the query (keyword / task)
        query_embedding = openai_client.generate_embeddings(text=query)

        # Step 2: Cosine similarity search between e(query_embed) and e(existing_scheme_embed)
        semantic_results = cos_sim_search(query_embedding, search_space=None)

        # Extract and return only the table names of the top k results
        top_k_results = [result['table_name'] for result in semantic_results[:k]]
        return top_k_results
    except Exception as e:
        logging.error(f"Error during semantic search: {e}")
        return []


if __name__ == "__main__":
    # Test for semantic search
    query = "medicine demand forecast"
    # query = "What data is needed to train a machine learning model to forecast demand for medicines?"
    top_k_results = semantic_search(query, k=10)
    print(top_k_results)