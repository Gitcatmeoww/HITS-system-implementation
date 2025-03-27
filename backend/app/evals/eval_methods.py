from backend.app.table_representation.openai_client import OpenAIClient
from backend.app.hyse.hypo_schema_search import cos_sim_search
import logging
from dotenv import load_dotenv
# from backend.app.evals.elastic_search.es_client import es_client
from backend.app.hyse.hypo_schema_search import hyse_search
from backend.app.actions.infer_action import infer_mentioned_metadata_fields, text_to_sql, execute_sql
from eval_utils import generate_embeddings, average_embeddings_with_weights, get_query_embedding_from_db, get_keyword_embedding_from_db, save_query_embedding_to_db, save_keyword_embedding_to_db, retrieve_or_generate_schemas

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
    def semantic_search(self, query, query_type="task", top_k=None):
        try:
            if top_k is None:
                top_k = self.k      
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
            top_k_results = [result['table_name'] for result in semantic_results[:top_k]]
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
      
    def single_hyse_search(
        self,
        query,
        num_embed=1,
        include_query_embed=True,
        query_weight=0.5,
        hypo_weight=0.5,
        return_embedding=False,
        search_space=None,
        top_k=None,
        schema_approach="relational",
    ):
        try:
            if top_k is None:
                top_k = self.k

            # Step 1: Retrieve cached query embedding if needed
            query_embedding = None
            if include_query_embed:
                query_embedding = get_query_embedding_from_db(query)
                if query_embedding is None:
                    # Generate & cache query embedding if not already cached
                    query_embedding = generate_embeddings([query])[0]
                    save_query_embedding_to_db(query, query_embedding)

            # Step 2: Retrieve or generate schema embeddings
            all_schema_embeddings = []

            if schema_approach == "relational":
                rel_embeds = retrieve_or_generate_schemas(
                    query=query,
                    num_embed=num_embed,
                    table_name="eval_hyse_schemas",
                    schema_approach="relational"
                )
                all_schema_embeddings.extend(rel_embeds)

            elif schema_approach == "non_relational":
                non_rel_embeds = retrieve_or_generate_schemas(
                    query=query,
                    num_embed=num_embed,
                    table_name="eval_hyse_schemas_non_relational",
                    schema_approach="non_relational"
                )
                all_schema_embeddings.extend(non_rel_embeds)

            elif schema_approach == "dual":
                rel_embeds = retrieve_or_generate_schemas(
                    query=query,
                    num_embed=num_embed,
                    table_name="eval_hyse_schemas",
                    schema_approach="relational"
                )
                non_rel_embeds = retrieve_or_generate_schemas(
                    query=query,
                    num_embed=num_embed,
                    table_name="eval_hyse_schemas_non_relational",
                    schema_approach="non_relational"
                )
                # Combine relational & non-relational embeddings
                all_schema_embeddings.extend(rel_embeds)
                all_schema_embeddings.extend(non_rel_embeds)

            else:
                raise ValueError(f"Unknown schema_approach: {schema_approach}")

            # Step 3: Combine embeddings based on the include_query_embedding flag
            combined_list = []
            if query_embedding is not None and include_query_embed:
                combined_list.append(query_embedding)
            combined_list.extend(all_schema_embeddings)

            # Step 4: Average the embeddings
            final_embedding = average_embeddings_with_weights(
                combined_list,
                query_weight=query_weight,
                hypo_weight=hypo_weight
            )

            if return_embedding:
                # Just return the HySE embedding, do not perform retrieval
                return final_embedding

            # Step 5: Perform similarity search between the final averaged embedding and e(existing_scheme_embed)
            results = cos_sim_search(final_embedding, search_space=search_space, table_name=self.data_split, column_name=self.embed_col)
            
            # Step 6: Extract and return only the table names of the top-k results
            top_k_results = [result['table_name'] for result in results[:top_k]]
            return top_k_results

        except Exception as e:
            logging.exception(f"Error during single_hyse_search: {e}")
            return []

    # TODO: Multi-hyse implementation
    def multi_hyse_search(self, query):
        pass

    def metadata_search(self, metadata_query, search_space):
        try:    
            # Step 1: Infer which fields the query is referencing (tags, col_num, row_num, time_granu, geo_granu)
            inferred_raw_fields = infer_mentioned_metadata_fields(
                cur_query=metadata_query,
                semantic_metadata=False
            ).get_true_fields()

            # logging.info(f"Inferred raw fields for query '{metadata_query}': {inferred_raw_fields}")

            # Step 2: Excute text to sql
            sql_clauses = text_to_sql(cur_query=metadata_query, identified_fields=inferred_raw_fields)

            # for clause in sql_clauses.sql_clauses:
            #     logging.info(f"SQL Clause => field: {clause.field}, clause: {clause.clause}")

            results = execute_sql(
                text_to_sql_instance=sql_clauses,
                search_space=search_space,
                table_name=self.data_split
            )
  
            # Step 3: Extract and return only the table names of the refined results
            return [row["table_name"] for row in results]
        except Exception as e:
            logging.exception(f"Error during metadata search: {e}")
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