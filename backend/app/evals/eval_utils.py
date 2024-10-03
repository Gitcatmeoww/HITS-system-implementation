from backend.app.db.connect_db import DatabaseConnection
import logging

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