import logging
from backend.app.db.connect_db import DatabaseConnection
from backend.app.hyse.hypo_schema_search import infer_single_hypothetical_schema, PROMPT_SINGLE_SCHEMA_RELATIONAL, PROMPT_SINGLE_SCHEMA_NON_RELATIONAL
from backend.app.table_representation.openai_client import OpenAIClient
import numpy as np
import json
from dotenv import load_dotenv

load_dotenv()

# OpenAI client instantiation
openai_client = OpenAIClient()

def get_hypo_schema_from_db(query):
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT hypo_schema, hypo_schema_embed FROM eval_hyse_schemas_non_relational
            WHERE query = %s;
            """
            db.cursor.execute(select_query, (query,))
            result = db.cursor.fetchone()
            if result:
                hypo_schema = result['hypo_schema']
                hypo_schema_embed = result['hypo_schema_embed']
                return hypo_schema, hypo_schema_embed
            else:
                return None, None
    except Exception as e:
        logging.exception(f"Error retrieving hypothetical schema from DB: {e}")
        return None, None
    
def get_hypo_schemas_from_db(query, num_embed, table_name="eval_hyse_schemas"):
    try:
        with DatabaseConnection() as db:
            select_query = f"""
            SELECT hypo_schema, hypo_schema_embed
            FROM {table_name}
            WHERE query = %s
            ORDER BY hypo_schema_id ASC
            LIMIT %s;
            """
            db.cursor.execute(select_query, (query, num_embed))
            results = db.cursor.fetchall()

            schemas = []
            embeddings = []
            for result in results:
                schemas.append(result['hypo_schema'])

                if isinstance(result['hypo_schema_embed'], str):
                    embedding_list = json.loads(result['hypo_schema_embed'])
                elif isinstance(result['hypo_schema_embed'], list):
                    embedding_list = result['hypo_schema_embed']
                else:
                    embedding_list = result['hypo_schema_embed']

                # Convert to NumPy array
                embedding = np.array(embedding_list, dtype=float)
                embeddings.append(embedding)
            return schemas, embeddings
    except Exception as e:
        logging.exception(f"Error retrieving hypothetical schemas from DB: {e}")
        return [], []

def get_query_embedding_from_db(query):
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT query_embed FROM eval_query_embeds
            WHERE query = %s;
            """
            db.cursor.execute(select_query, (query,))
            result = db.cursor.fetchone()
            if result:
                if isinstance(result['query_embed'], str):
                    query_embedding = json.loads(result['query_embed'])
                elif isinstance(result['query_embed'], list):
                    query_embedding = result['query_embed']
                else:
                    query_embedding = result['query_embed']
                
                # Convert to NumPy array
                return np.array(query_embedding, dtype=float)
            return None
    except Exception as e:
        logging.exception(f"Error retrieving query embedding from DB: {e}")
        return None
    
def get_keyword_embedding_from_db(keyword):
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT keyword_embed FROM eval_keyword_embeds
            WHERE keyword = %s;
            """
            db.cursor.execute(select_query, (keyword,))
            result = db.cursor.fetchone()
            if result:
                if isinstance(result['keyword_embed'], str):
                    keyword_embedding = json.loads(result['keyword_embed'])
                elif isinstance(result['keyword_embed'], list):
                    keyword_embedding = result['keyword_embed']
                else:
                    keyword_embedding = result['keyword_embed']
                
                # Convert to NumPy array
                return np.array(keyword_embedding, dtype=float)
            return None
    except Exception as e:
        logging.exception(f"Error retrieving keyword embedding from DB: {e}")
        return None

def save_hypo_schema_to_db(query, hypo_schema, hypo_schema_embed):
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_hyse_schemas_non_relational (query, hypo_schema, hypo_schema_embed)
            VALUES (%s, %s, %s)
            ON CONFLICT (query) DO UPDATE SET
            hypo_schema = EXCLUDED.hypo_schema,
            hypo_schema_embed = EXCLUDED.hypo_schema_embed;
            """
            db.cursor.execute(insert_query, (query, hypo_schema, hypo_schema_embed))
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving hypothetical schema to DB: {e}")

def save_hypo_schemas_to_db(query, schemas, embeddings, table_name="eval_hyse_schemas"):
    try:
        with DatabaseConnection() as db:
            insert_query = f"""
            INSERT INTO {table_name} (query, hypo_schema, hypo_schema_embed)
            VALUES (%s, %s, %s);
            """
            data = []
            for schema, embedding in zip(schemas, embeddings):
                # Convert embedding to list
                embedding_list = embedding.tolist()
                data.append((query, schema, embedding_list))
            db.cursor.executemany(insert_query, data)
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving hypothetical schemas to DB: {e}")
    
def save_query_embedding_to_db(query, query_embedding):
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_query_embeds (query, query_embed)
            VALUES (%s, %s)
            ON CONFLICT (query) DO NOTHING;
            """
            # Ensure query_embedding is a list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            db.cursor.execute(insert_query, (query, query_embedding))
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving query embedding to DB: {e}")

def save_keyword_embedding_to_db(keyword, keyword_embedding):
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_keyword_embeds (keyword, keyword_embed)
            VALUES (%s, %s)
            ON CONFLICT (keyword) DO NOTHING;
            """
            # Ensure keyword_embedding is a list
            if isinstance(keyword_embedding, np.ndarray):
                keyword_embedding = keyword_embedding.tolist()
            db.cursor.execute(insert_query, (keyword, keyword_embedding))
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving keyword embeddings to DB: {e}")

# def get_ground_truth_header(table_name, data_split):
#     try:
#         with DatabaseConnection() as db:
#             query = f"SELECT example_rows_md FROM {data_split} WHERE table_name = %s;"
#             db.cursor.execute(query, (table_name,))
#             result = db.cursor.fetchone()
#             if result and result['example_rows_md']:
#                 return result['example_rows_md']
#             else:
#                 return ''
#     except Exception as e:
#         logging.exception(f"Error retrieving ground truth header for table '{table_name}': {e}")
#         return ''

def get_ground_truth_header(table_name, data_split):
    try:
        with DatabaseConnection() as db:
            query = f"SELECT table_header FROM {data_split} WHERE table_name = %s;"
            db.cursor.execute(query, (table_name,))
            result = db.cursor.fetchone()
            if result and result['table_header']:
                return result['table_header']
            else:
                return ''
    except Exception as e:
        logging.exception(f"Error retrieving ground truth header for table '{table_name}': {e}")
        return ''

def get_hypo_schema(query):
    try:
        with DatabaseConnection() as db:
            select_query = "SELECT hypo_schema FROM eval_hyse_schemas_non_relational WHERE query = %s;"
            db.cursor.execute(select_query, (query,))
            result = db.cursor.fetchone()
            if result and result['hypo_schema']:
                return result['hypo_schema']
            else:
                return ''
    except Exception as e:
        logging.exception(f"Error retrieving hypothetical schema for query '{query}': {e}")
        return ''

def generate_hypothetical_schemas(query, num_to_generate, schema_approach="relational"):
    schemas = []
    try:
        # Pick the template based on schema_approach
        if schema_approach == "relational":
            prompt_template = PROMPT_SINGLE_SCHEMA_RELATIONAL
        else:
            prompt_template = PROMPT_SINGLE_SCHEMA_NON_RELATIONAL

        # Generate the requested number of schemas
        for _ in range(num_to_generate):
            schema = infer_single_hypothetical_schema(initial_query=query, prompt_template=prompt_template).json()
            schemas.append(schema)
        return schemas
    except Exception as e:
        logging.exception(f"Error generating hypothetical schemas for query '{query}': {e}")
        return []

def generate_embeddings(schemas):
    embeddings = []
    try:
        for schema in schemas:
            embedding = openai_client.generate_embeddings(text=schema)

            # Convert embedding to NumPy array if not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=float)

            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        logging.exception(f"Error generating embeddings for schemas '{schemas}': {e}")
        return []

def retrieve_or_generate_schemas(query, num_embed, table_name, schema_approach): 
    # Fetch up to `num_embed` schemas + embeddings from `table_name`
    cached_schemas, cached_embeddings = get_hypo_schemas_from_db(query, num_embed, table_name=table_name)
    num_cached = len(cached_schemas)

    # If not enough cached, generate more using `generate_prompt_func`
    if num_cached < num_embed:
        num_to_generate = num_embed - num_cached
        new_schemas = generate_hypothetical_schemas(
            query,
            num_to_generate,
            schema_approach=schema_approach
        )
        new_embeddings = generate_embeddings(new_schemas)

        # Save new schemas & embeddings to DB
        save_hypo_schemas_to_db(query, new_schemas, new_embeddings, table_name=table_name)

        # Combine cached & new embeddings
        cached_embeddings.extend(new_embeddings)

    return cached_embeddings
    
def average_embeddings(embeddings):
    # Ensure all embeddings are NumPy arrays of floats
    embeddings = [np.array(embed, dtype=float) if not isinstance(embed, np.ndarray) else embed.astype(float) for embed in embeddings]
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

def average_embeddings_with_weights(embeddings, query_weight, hypo_weight):
    query_embedding = embeddings[0]
    hypo_embeddings = embeddings[1:]
    avg_embedding = (query_weight * query_embedding + hypo_weight * np.mean(hypo_embeddings, axis=0))
    return avg_embedding