from backend.app.table_representation.openai_client import OpenAIClient
from backend.app.utils import format_prompt, format_cos_sim_results
from backend.app.db.connect_db import DatabaseConnection
from pydantic import BaseModel
from typing import List
import logging

# Initialize OpenAI client
openai_client = OpenAIClient()

# Craft schema inference prompt
PROMPT_SINGLE_SCHEMA = """
Given the objective of {query}, help me generate a hypothetical table schema to support this.
Only generate one table schema, excluding any introductory phrases and focusing exclusively on the tasks themselves.
"""

PROMPT_MULTI_SCHEMA = """
Given the objective of {query}, help me generate {num_schema} normalized hypothetical table schema to support this.
Generate the schemas as a JSON array of objects, where each object represents a normalized schema.
Excluding any introductory phrases and focusing exclusively on the tasks themselves.
"""

# TODO: refactor pydantic models
# Define desired output structure
class TableSchema(BaseModel):
    column_names: list[str]
    data_types: list[str]


def hyse_search(initial_query, search_space=None, num_schema=1, k=10):
    # Step 0: Initialize the results list
    results = []
    # Determine k for single & multiple HySE evaluation
    k_single = k // num_schema + (k % num_schema > 0)
    k_multi = k // num_schema

    # Step 1: Single HySE search
    # Step 1.1: Infer a single denormalized hypothetical schema
    single_hypo_schema_json = infer_single_hypothetical_schema(initial_query).json()

    # Step 1.2: Generate embedding for the single hypothetical schema
    single_hypo_schema_embedding = openai_client.generate_embeddings(text=single_hypo_schema_json)

    # Step 1.3: Cosine similarity search between e(hypo_schema_embed) and e(existing_scheme_embed)
    single_hyse_results = cos_sim_search(single_hypo_schema_embedding, search_space, k_single)
    results.append(single_hyse_results)
    
    # Step 2: Multiple HySE search (N > 1)
    if num_schema > 1:
        # Step 2.1: Infer multiple normalized hypothetical schemas
        multi_hypo_schemas_json = infer_multiple_hypothetical_schema(initial_query, num_schema - 1).json()

        # Step 2.2: Generate embeddings for the multiple hypothetical schemas
        multi_hypo_schemas_embeddings = [openai_client.generate_embeddings(text=schema) for schema in multi_hypo_schemas_json]

        # Step 2.3: Cosine similarity search for each multiple hypothetical schema embedding
        for hypo_schema_embedding in multi_hypo_schemas_embeddings:
            results.append(cos_sim_search(hypo_schema_embedding, search_space, k_multi))

    # Step 3: Aggregate results from single & multiple HySE searches
    aggregated_results = aggregate_hyse_search_results(results)

    return aggregated_results

def infer_single_hypothetical_schema(initial_query):
    prompt = format_prompt(PROMPT_SINGLE_SCHEMA, query=initial_query)

    response_model = TableSchema

    messages = [
        {"role": "system", "content": "You are an assistant skilled in generating database schemas."},
        {"role": "user", "content": prompt}
    ]

    return openai_client.infer_metadata(messages, response_model)

def infer_multiple_hypothetical_schema(initial_query, num_schema):
    prompt = format_prompt(PROMPT_MULTI_SCHEMA, query=initial_query, num_schema=num_schema)

    response_model = List[TableSchema]

    messages = [
        {"role": "system", "content": "You are an assistant skilled in generating normalized database schemas."},
        {"role": "user", "content": prompt}
    ]

    return openai_client.infer_metadata(messages, response_model)

def cos_sim_search(input_embedding, search_space, k, column_name="comb_embed"):
    if column_name not in ["comb_embed", "query_embed"]:
        raise ValueError("Invalid embedding column")
    
    with DatabaseConnection() as db:
        if search_space:
            # Filter by specific table names
            query = f"""
                SELECT table_name, 1 - ({column_name} <=> %s::VECTOR(1536)) AS cosine_similarity
                FROM corpus_raw_metadata_with_embedding
                WHERE table_name = ANY(%s)
                ORDER BY cosine_similarity DESC
                LIMIT %s;
            """
            db.cursor.execute(query, (input_embedding, search_space, k))
        else:
            # No specific search space, search through all table names
            query = f"""
                SELECT table_name, 1 - ({column_name} <=> %s::VECTOR(1536)) AS cosine_similarity
                FROM corpus_raw_metadata_with_embedding
                ORDER BY cosine_similarity DESC
                LIMIT %s;
            """
            db.cursor.execute(query, (input_embedding, k))
        
        results = db.cursor.fetchall()
    
    # Ensure results are correctly structured
    formatted_results = [{'table_name': row['table_name'], 'cosine_similarity': float(row['cosine_similarity'])} for row in results]
    return formatted_results

def aggregate_hyse_search_results(results):
    # Flatten the list of results
    flat_results = [item for sublist in results for item in sublist]
    
    # Aggregate by table name and calculate mean cosine similarity
    aggregated_results = {}
    for result in flat_results:
        table_name = result['table_name']
        cosine_similarity = result['cosine_similarity']
        
        if not isinstance(cosine_similarity, (int, float)):
            logging.error(f"Unexpected type for cosine_similarity: {type(cosine_similarity)} with value {cosine_similarity}")
            raise ValueError(f"Unexpected type for cosine_similarity: {type(cosine_similarity)}")
        
        if table_name in aggregated_results:
            aggregated_results[table_name].append(cosine_similarity)
        else:
            aggregated_results[table_name] = [cosine_similarity]
    
    # Calculate the mean cosine similarity for each table
    final_results = [{'table_name': table_name, 'cosine_similarity': sum(similarities) / len(similarities)} for table_name, similarities in aggregated_results.items()]
    
    # Sort the final results by mean cosine similarity in descending order
    final_results.sort(key=lambda x: x['cosine_similarity'], reverse=True)
    return final_results
