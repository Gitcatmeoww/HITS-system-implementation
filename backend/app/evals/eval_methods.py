from backend.app.table_representation.openai_client import OpenAIClient
from backend.app.hyse.hypo_schema_search import cos_sim_search
import logging
from dotenv import load_dotenv
# from backend.app.evals.elastic_search.es_client import es_client
from backend.app.hyse.hypo_schema_search import hyse_search
from eval_utils import get_hypo_schema_from_db, save_hypo_schema_to_db, get_hypo_schemas_from_db, save_hypo_schemas_to_db, generate_hypothetical_schemas, generate_embeddings, average_embeddings

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
    def semantic_search(self, query):
        try:
            # Step 1: Generate embedding for the query (keyword / task)
            query_embedding = self.openai_client.generate_embeddings(text=query)

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
    
    def hyse_search(self, query, num_schema=1):
        try:
            # Check if existing hypothetical schema & embedding available in DB
            hypo_schema, hypo_schema_embed = get_hypo_schema_from_db(query)

            # If NOT found in cache, call the original hyse_search function
            if hypo_schema is None or hypo_schema_embed is None:
                results, hypo_schema, hypo_schema_embed = hyse_search(
                    query,
                    search_space=None,
                    num_schema=num_schema,
                    k=self.k,
                    table_name=self.data_split,
                    column_name=self.embed_col
                )
                # Save the schema & embedding to DB
                save_hypo_schema_to_db(query, hypo_schema, hypo_schema_embed)
            
            # If FOUND in cache, perform cosine similarity search between cached embedding and e(existing_scheme_embed)
            else:               
                results = cos_sim_search(hypo_schema_embed, search_space=None, table_name=self.data_split, column_name=self.embed_col)

            # Extract only the table_name from each result
            return [result['table_name'] for result in results]
        except Exception as e:
            logging.error(f"Error during hyse search: {e}")
            return []
    
    def single_hyse_search(self, query, num_embed=1):
        try:
            # Retrieve up to `num_embed` hypothetical schemas & embeddings from cache
            cached_schemas, cached_embeddings = get_hypo_schemas_from_db(query, num_embed)
            num_cached = len(cached_schemas)

            if num_cached < num_embed:
                # Need to generate additional hypothetical schemas
                num_to_generate = num_embed - num_cached
                # Generate `num_to_generate` hypothetical schemas & embeddings
                new_schemas = generate_hypothetical_schemas(query, num_to_generate)
                new_embeddings = generate_embeddings(new_schemas)

                # Save new schemas & embeddings to DB
                save_hypo_schemas_to_db(query, new_schemas, new_embeddings)
                # Combine cached & new embeddings
                cached_embeddings.extend(new_embeddings)

            # Average the embeddings
            avg_embedding = average_embeddings(cached_embeddings)
            # Perform similarity search between the averaged embedding and e(existing_scheme_embed)
            results = cos_sim_search(avg_embedding, search_space=None, table_name=self.data_split, column_name=self.embed_col)
            
            # Extract only the table_name from each result
            return [result['table_name'] for result in results]
        except Exception as e:
            logging.exception(f"Error during single hyse search: {e}")
            return []


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