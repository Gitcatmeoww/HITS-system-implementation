from backend.app.table_representation.openai_client import OpenAIClient
from backend.app.hyse.hypo_schema_search import cos_sim_search
import logging
from dotenv import load_dotenv
# from backend.app.evals.elastic_search.es_client import es_client
from backend.app.hyse.hypo_schema_search import hyse_search
from eval_utils import get_hypo_schema_from_db, save_hypo_schema_to_db, get_hypo_schemas_from_db, save_hypo_schemas_to_db, generate_hypothetical_schemas, generate_embeddings, average_embeddings, average_embeddings_with_weights, get_query_embedding_from_db, get_keyword_embedding_from_db, save_query_embedding_to_db, save_keyword_embedding_to_db

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
openai_client = OpenAIClient()

class EvalMethods:
    def __init__(self, data_split, embed_col, k):
        self.openai_client = openai_client
        # self.es_client = es_client.client
        self.data_split = data_split
        self.embed_col = embed_col
        self.k = k

    # Evaluate semantic keywords search & semantic task search against HySE  
    def semantic_search(self, query, query_type="task"):
        try:        
            query_embedding = None

            # Step 1: Get or generate embedding for the query (keyword / task)
            if query_type == "task":
                # Check if the query embedding is cached
                query_embedding = get_query_embedding_from_db(query)
                if query_embedding is None:
                    # Generate & cache the query embedding
                    query_embedding = generate_embeddings([query])[0]
                    save_query_embedding_to_db(query, query_embedding)
            elif query_type == "keyword":
                # Check if the keyword embedding is cached
                query_embedding = get_keyword_embedding_from_db(query)
                if query_embedding is None:
                    # Generate & cache the keyword embedding
                    query_embedding = generate_embeddings([query])[0]
                    save_keyword_embedding_to_db(query, query_embedding)
            else:
                raise ValueError(f"Invalid query_type: {query_type}. Must be 'task' or 'keyword'.")

            # Step 2: Cosine similarity search between e(query_embed) and e(existing_scheme_embed)
            semantic_results = cos_sim_search(query_embedding, search_space=None, table_name=self.data_split, column_name=self.embed_col)

            # Extract and return only the table names of the top k results
            top_k_results = [result['table_name'] for result in semantic_results[:self.k]]
            return top_k_results
        except Exception as e:
            logging.error(f"Error during semantic search: {e}")
            return []
    
    # Evaluate syntactic keywords search against HySE
    # Perform syntactic keyword search against table_name & example_rows_md fields
    def syntactic_search(self, query):
        try:
            # Validate data_split
            valid_splits = ['eval_data_all', 'eval_data_test', 'eval_data_train', 'eval_data_validation']
            if self.data_split not in valid_splits:
                logging.error(f"Invalid data_split: {self.data_split}. Must be one of {valid_splits}")
                return []
            
            # Define the Elasticsearch index based on data_split
            index_name = self.data_split

            # Check if the index exists
            if not self.es_client.indices.exists(index=index_name):
                logging.error(f"Elasticsearch index '{index_name}' does not exist")
                return []

            # Define the search query
            es_query = {
                "size": self.k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "table_name": {
                                        "query": query,
                                        "fuzziness": "AUTO"
                                    }
                                }
                            },
                            {
                                "match": {
                                    "example_rows_md": {
                                        "query": query,
                                        "fuzziness": "AUTO"
                                    }
                                }
                            }
                        ]
                    }
                }
            }

            # Execute the search query
            response = self.es_client.search(index=index_name, body=es_query)

            # Extract table names from the search results
            top_k_results = [hit['_source']['table_name'] for hit in response['hits']['hits']]
            return top_k_results
        except Exception as e:
            logging.error(f"Error during syntactic search: {e}")
            return []

    def single_hyse_search(self, query, num_embed=1, include_query_embed=True, query_weight=0.5, hypo_weight=0.5):
        try:
            # Step 1: Retrieve cached query embedding if needed
            query_embedding = None
            if include_query_embed:
                query_embedding = get_query_embedding_from_db(query)
                if query_embedding is None:
                    # Generate & cache query embedding if not already cached
                    query_embedding = generate_embeddings([query])[0]
                    save_query_embedding_to_db(query, query_embedding)

            # Step 2: Retrieve hypothetical schemas & embeddings from cache
            cached_schemas, cached_embeddings = get_hypo_schemas_from_db(query, num_embed)
            num_cached = len(cached_schemas)

            # Step 3: Generate additional hypothetical schemas if needed
            if num_cached < num_embed:
                num_to_generate = num_embed - num_cached
                new_schemas = generate_hypothetical_schemas(query, num_to_generate)
                new_embeddings = generate_embeddings(new_schemas)

                # Save new schemas & embeddings to DB
                save_hypo_schemas_to_db(query, new_schemas, new_embeddings)

                # Combine cached & new embeddings
                cached_embeddings.extend(new_embeddings)

            # Step 4: Combine embeddings based on the include_query_embedding flag
            combined_embeddings = cached_embeddings
            if include_query_embed and query_embedding is not None:
                combined_embeddings = [query_embedding] + cached_embeddings

            # Step 5: Average the embeddings
            avg_embedding = average_embeddings_with_weights(
                combined_embeddings,
                query_weight=query_weight,
                hypo_weight=hypo_weight
            )

            # Step 6: Perform similarity search between the averaged embedding and e(existing_scheme_embed)
            results = cos_sim_search(avg_embedding, search_space=None, table_name=self.data_split, column_name=self.embed_col)

            # Step 7: Extract and return only the table names of the top-k results
            top_k_results = [result['table_name'] for result in results[:self.k]]
            return top_k_results
        except Exception as e:
            logging.exception(f"Error during single hyse search: {e}")
            return []

    # TODO: Multi-hyse implementation
    def multi_hyse_search(self, query):
        pass


if __name__ == "__main__":
    # Create an instance of HyseEval
    hyse_eval = EvalMethods(
        data_split="eval_data_validation",
        embed_col="example_rows_embed",
        k=10
    )

    # Test for semantic search
    query = "medicine demand forecast"
    top_k_semantic = hyse_eval.semantic_search(query)
    print("Semantic Search Results:", top_k_semantic)

    # Test for syntactic search
    syntactic_query = "demand forecast"
    top_k_syntactic = hyse_eval.syntactic_search(syntactic_query)
    print("Syntactic Search Results:", top_k_syntactic)

    # Test for hyse search
    semantic_query = "What data is needed to train a machine learning model to forecast demand for medicines?"
    top_k_hyse = hyse_eval.hyse_search(semantic_query)
    print("HySE Search Results:", top_k_hyse)