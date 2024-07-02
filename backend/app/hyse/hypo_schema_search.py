from backend.app.table_representation.openai_client import OpenAIClient
from backend.app.utils import format_prompt, format_cos_sim_results
from backend.app.db.connect_db import DatabaseConnection
from pydantic import BaseModel
from typing import List, Any
import random
import logging

# Initialize OpenAI client
openai_client = OpenAIClient()

# Craft schema inference prompt
PROMPT_SINGLE_SCHEMA = """
Given the task of {query}, help me generate a database schema to to implement the task.
Only generate one table schema, excluding any introductory phrases and focusing exclusively on the tasks themselves.
Generate a JSON with keys as table names and values as column names, data types, and example rows. For example:

Task:
What data is needed to train a machine learning model to forecast demand for medicines across suppliers?

Output: 
{{
    "table_name": "Sales",
    "column_names": ["sale_id", "medicine_id", "supplier_id", "quantity_sold", "sale_date", "price", "region"],
    "data_types": ["INT", "INT", "INT", "INT", "DATE", "DECIMAL", "VARCHAR"],
    "example_row": [1, 101, 201, 50, "2024-06-01", 19.99, "North America"]
}}
"""

PROMPT_MULTI_SCHEMA = """
Given the task of {query}, generate a database schema of at least 1, and at most {num_left} normalized table headers that are needed to implement the task.
Generate a list in which each element is a JSON with keys as table names and values as column names, data types, and example rows. For example:

Task:
What data is needed to train a machine learning model to forecast demand for medicines across suppliers?

Output: 
[
  {{
    "table_name": "Sales",
    "column_names": ["sale_id", "medicine_id", "supplier_id", "quantity_sold", "sale_date", "price", "region"],
    "data_types": ["INT", "INT", "INT", "INT", "DATE", "DECIMAL", "VARCHAR"],
    "example_row": [1, 101, 201, 50, "2024-06-01", 19.99, "North America"]
  }},
  {{
    "table_name": "Medicine",
    "column_names": ["medicine_id", "name", "type", "dosage", "packaging", "shelf_life", "supplier_id"],
    "data_types": ["INT", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR", "INT", "INT"],
    "example_row": [101, "Aspirin", "Pain Reliever", "500mg", "Bottle", 24, 201]
  }}
]
"""

# TODO: refactor pydantic models
# Define desired output structure
class TableSchema(BaseModel):
    table_name: str
    column_names: List[str]
    data_types: List[str]
    example_row: List[Any]

def hyse_search(initial_query, search_space=None, num_schema=1, k=10):
    # Step 0: Initialize the results list and num_left
    results = []
    num_left = num_schema

    # Step 1: Single HySE search
    # Step 1.1: Infer a single denormalized hypothetical schema
    single_hypo_schema_json = infer_single_hypothetical_schema(initial_query).json()
    logging.info(f"Single hypothetical schema JSON: {single_hypo_schema_json}")

    # Step 1.2: Generate embedding for the single hypothetical schema
    single_hypo_schema_embedding = openai_client.generate_embeddings(text=single_hypo_schema_json)

    # Step 1.3: Cosine similarity search between e(hypo_schema_embed) and e(existing_scheme_embed)
    single_hyse_results = cos_sim_search(single_hypo_schema_embedding, search_space)
    results.append(single_hyse_results)

    # Step 1.4: Update num_left by decrementing it by 1
    num_left -= 1
    
    # Step 2: Multiple HySE search
    while num_left > 0:
        # Step 2.1: Randomly generate the number of normalized schemas m ranging from 1 to num_left
        m = random.randint(1, num_left)

        # Step 2.2: Infer multiple normalized hypothetical schemas
        multi_hypo_schemas, m = infer_multiple_hypothetical_schema(initial_query, m)
        multi_hypo_schemas_json = [schema.json() for schema in multi_hypo_schemas]
        logging.info(f"Multiple hypothetical schemas JSON: {multi_hypo_schemas_json}")

        # Step 2.3: Generate embeddings for the multiple hypothetical schemas
        multi_hypo_schemas_embeddings = [openai_client.generate_embeddings(text=schema_json) for schema_json in multi_hypo_schemas_json]

        # Step 2.4: Cosine similarity search for each multiple hypothetical schema embedding
        for hypo_schema_embedding in multi_hypo_schemas_embeddings:
            results.append(cos_sim_search(hypo_schema_embedding, search_space))
        
        # Step 2.5: Update num_left by decrementing it by m
        num_left -= m

    # Step 3: Aggregate results from single & multiple HySE searches
    aggregated_results = aggregate_hyse_search_results(results)

    # Sort aggregated results by cosine similarity and keep top k
    aggregated_results.sort(key=lambda x: x['cosine_similarity'], reverse=True)
    top_k_results = aggregated_results[:k]

    return top_k_results

def infer_single_hypothetical_schema(initial_query):
    prompt = format_prompt(PROMPT_SINGLE_SCHEMA, query=initial_query)

    response_model = TableSchema

    messages = [
        {"role": "system", "content": "You are an assistant skilled in generating database schemas."},
        {"role": "user", "content": prompt}
    ]

    return openai_client.infer_metadata(messages, response_model)

def infer_multiple_hypothetical_schema(initial_query, num_left):
    prompt = format_prompt(PROMPT_MULTI_SCHEMA, query=initial_query, num_left=num_left)

    response_model = List[TableSchema]

    messages = [
        {"role": "system", "content": "You are an assistant skilled in generating normalized database schemas."},
        {"role": "user", "content": prompt}
    ]

    response = openai_client.infer_metadata(messages, response_model)
    m = len(response)
    return response, m

def cos_sim_search(input_embedding, search_space, column_name="comb_embed"):
    if column_name not in ["comb_embed", "query_embed"]:
        raise ValueError("Invalid embedding column")
    
    with DatabaseConnection() as db:
        if search_space:
            # Filter by specific table names
            query = f"""
                SELECT table_name, 1 - ({column_name} <=> %s::VECTOR(1536)) AS cosine_similarity
                FROM corpus_raw_metadata_with_embedding
                WHERE table_name = ANY(%s)
                ORDER BY cosine_similarity DESC;
            """
            db.cursor.execute(query, (input_embedding, search_space))
        else:
            # No specific search space, search through all table names
            query = f"""
                SELECT table_name, 1 - ({column_name} <=> %s::VECTOR(1536)) AS cosine_similarity
                FROM corpus_raw_metadata_with_embedding
                ORDER BY cosine_similarity DESC;
            """
            db.cursor.execute(query, (input_embedding,))
        
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
