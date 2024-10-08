import logging
from backend.app.db.connect_db import DatabaseConnection
from backend.app.hyse.hypo_schema_search import infer_single_hypothetical_schema
from backend.app.table_representation.openai_client import OpenAIClient
import numpy as np
import json

# OpenAI client instantiation
openai_client = OpenAIClient()

def get_hypo_schema_from_db(query):
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT hypo_schema, hypo_schema_embed FROM eval_hyse_schemas
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
    
def get_hypo_schemas_from_db(query, num_embed):
    try:
        with DatabaseConnection() as db:
            select_query = """
            SELECT hypo_schema, hypo_schema_embed FROM eval_hyse_schemas
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

def save_hypo_schema_to_db(query, hypo_schema, hypo_schema_embed):
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_hyse_schemas (query, hypo_schema, hypo_schema_embed)
            VALUES (%s, %s, %s)
            ON CONFLICT (query) DO UPDATE SET
            hypo_schema = EXCLUDED.hypo_schema,
            hypo_schema_embed = EXCLUDED.hypo_schema_embed;
            """
            db.cursor.execute(insert_query, (query, hypo_schema, hypo_schema_embed))
            db.conn.commit()
    except Exception as e:
        logging.exception(f"Error saving hypothetical schema to DB: {e}")

def save_hypo_schemas_to_db(query, schemas, embeddings):
    try:
        with DatabaseConnection() as db:
            insert_query = """
            INSERT INTO eval_hyse_schemas (query, hypo_schema, hypo_schema_embed)
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

def get_ground_truth_header(table_name, data_split):
    try:
        with DatabaseConnection() as db:
            query = f"SELECT example_rows_md FROM {data_split} WHERE table_name = %s;"
            db.cursor.execute(query, (table_name,))
            result = db.cursor.fetchone()
            if result and result['example_rows_md']:
                return result['example_rows_md']
            else:
                return ''
    except Exception as e:
        logging.exception(f"Error retrieving ground truth header for table '{table_name}': {e}")
        return ''

def get_hypo_schema(query):
    try:
        with DatabaseConnection() as db:
            select_query = "SELECT hypo_schema FROM eval_hyse_schemas WHERE query = %s;"
            db.cursor.execute(select_query, (query,))
            result = db.cursor.fetchone()
            if result and result['hypo_schema']:
                return result['hypo_schema']
            else:
                return ''
    except Exception as e:
        logging.exception(f"Error retrieving hypothetical schema for query '{query}': {e}")
        return ''

def generate_hypothetical_schemas(query, num_to_generate):
    schemas = []
    try:
        for _ in range(num_to_generate):
            schema = infer_single_hypothetical_schema(query).json()
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
    
def average_embeddings(embeddings):
    # Ensure all embeddings are NumPy arrays of floats
    embeddings = [np.array(embed, dtype=float) if not isinstance(embed, np.ndarray) else embed.astype(float) for embed in embeddings]
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding
