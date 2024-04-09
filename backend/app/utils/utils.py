import json
import re
from backend.app.db.connect_db import DatabaseConnection
import asyncio

def format_prompt(prompt_template, **kwargs):
    """ Formats a given prompt template with provided keyword arguments """
    return prompt_template.format(**kwargs)

def extract_time_geo_granularity(granularity_str):
    """ Extract time and geo granularity lists from a JSON string """
    granularity = json.loads(granularity_str)

    # Initialize empty sets for time and geographic granularities
    time_granu = set()
    geo_granu = set()

    # Iterate through each item in the granularity dictionary
    for value in granularity.values():
        # Check if 'temporal' key exists and has a non-null value
        if 'temporal' in value and value['temporal']:
            time_granu.add(value['temporal'])
        
        # Check if 'geographic' key exists and has a non-null value
        if 'geographic' in value and value['geographic']:
            geo_granu.add(value['geographic'])

    # Convert sets to lists for JSON serialization
    return list(time_granu), list(geo_granu)


def run_sql_file(filename):
    """ Executes the SQL statements contained within a file """
    with open(filename, 'r') as file:
        sql_script = file.read()

   # Use the DatabaseConnection context manager to get a cursor
    with DatabaseConnection() as cursor:
        try:
            # Split the script into individual statements if necessary
            sql_statements = sql_script.split(';')
            
            # Execute each statement individually
            for statement in sql_statements:
                # Strip whitespace and skip empty statements
                if statement.strip():
                    cursor.execute(statement.strip())
            # Commit the transaction
            cursor.connection.commit()
        except Exception as e:
            # Rollback the transaction on error
            cursor.connection.rollback()
            print(f"An error occurred: {e}")
            raise

def format_cos_sim_results(results):
    """ Format results returned from pgvector operations """
    formatted_results = []

    for row in results:
        formatted_result = {
            "table_name": row["table_name"],
            "cosine_similarity": row["cosine_similarity"]
        }
        formatted_results.append(formatted_result)

    return formatted_results

def clean_json_string(json_str):
    """ Clean a JSON string by fixing common formatting issues """
    # Remove special characters like newlines and tabs
    json_str = json_str.replace('\n', '\\n').replace('\t', '\\t')
    
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')
    
    # Add double quotes around any keys that are not properly quoted
    json_str = re.sub(r'(?<!")(\b[^":\n\r]+?\b)(?=\s*:)', r'"\1"', json_str)
    
    # Remove trailing commas before closing brackets or braces
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    return json_str

def validate_and_load_json(json_str):
    """ Validate and parse a JSON string into a Python dictionary using the cleaned JSON string """
    cleaned_json_str = clean_json_string(json_str)
    try:
        return json.loads(cleaned_json_str), None
    except json.JSONDecodeError as e:
        return None, str(e)

async def check_run_status(thread_id, run_id, openai_client, attempt=0):
    """ Hacky implementation to check the status of the assistant thread """
    MAX_ATTEMPTS = 30
    DELAY_BETWEEM_ATTEMPTS = 1  # seconds

    while attempt < MAX_ATTEMPTS:
        run_status = openai_client.run_status(thread_id=thread_id, run_id=run_id)
        if run_status == "completed":
            return True
        else:
            await asyncio.sleep(DELAY_BETWEEM_ATTEMPTS)
            attempt += 1
    return False