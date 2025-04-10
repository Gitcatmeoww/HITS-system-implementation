from backend.app.table_representation.openai_client import OpenAIClient
from backend.app.utils import format_prompt, get_finer_granularities
from pydantic import BaseModel
from typing import List, Dict
import logging
from backend.app.db.connect_db import DatabaseConnection
from psycopg2 import sql

# Initialize OpenAI client
openai_client = OpenAIClient()

# Craft action inference prompt
PROMPT_ACTION_INFER = """
Given two queries in a search session, decide whether the new query is a "reset" or a "refine" in relation to the previous query.
- A "reset" means the new query significantly differs from the previous query, indicating a change in the analytical task focus.
- A "refine" means the new query builds upon or slightly alters the previous query, indicating a more focused analytical task based on the earlier query.

Previous query: "{prev_query}"
Current query: "{cur_query}"

Should the search space be reset or refined?
"""

# TODO: Add examples in prompt to facilitate the metadata inference
# Craft mentioned metadata fields inference prompt
PROMPT_SEMANTIC_METADATA_INFER = """
Given the user's current query, analyze and determine which fields are being referenced. These fields include:
- Table Schema
- Example Records
- Table Description
- Table Tags

Current user query: "{cur_query}"

Identify any fields that are explicitly mentioned or strongly implied. Provide your analysis as a structured output listing only those fields that are directly related to the query.
"""

PROMPT_RAW_METADATA_INFER = """
Given the user's current query, analyze and determine which fields are being referenced. These fields include:
- Table Tags
- Number of Columns
- Number of Rows
- Temporal Granularity
- Geographic Granularity

Current user query: "{cur_query}"

Identify any fields that are explicitly mentioned or strongly implied. Provide your analysis as a structured output listing only those fields that are directly related to the query.
"""

PROMPT_SQL_TRANSLATION = """
Given these requirements, convert the identified metadata fields and the user query into precise SQL WHERE clause conditions. Follow these rules:

Identified fields: {identified_fields}
Current user query: "{cur_query}"

1. Temporal Granularity: Must be exactly one of:
   - 'Year', 'Quarter', 'Month', 'Week', 'Day', 'Hour', 'Minute', 'Second'

2. Geographic Granularity: Must be exactly one of:
   - 'Continent', 'Country', 'State/Province', 'County/District', 'City', 'Zip Code/Postal Code'

3. Numeric Columns (e.g., column_numbers, row_numbers): 
   - Use operators like >=, =, etc., based on user query context (e.g., at least 5 columns” => ">= 5")

4. Tags (table_tags): 
   - Never combine multiple tags in a single line such as "LIKE '%business%' AND table_tags LIKE '%education%'".
   - Instead, if the user wants “business and education” tags, produce two separate lines:
       {{ "field": "table_tags", "clause": "= 'business'" }},
       {{ "field": "table_tags", "clause": "= 'education'" }} 
     letting the final code treat them as AND (or OR) conditions as needed.

5. Output in JSON with a top-level key `"sql_clauses"` holding an array of objects. 
   - Each object should have `"field"` and `"clause"`. 
   - Example:
     {{
       "sql_clauses": [
         {{ "field": "table_tags", "clause": "= 'finance'" }},
         {{ "field": "temporal_granularity", "clause": "= 'day'" }},
         {{ "field": "row_numbers", "clause": ">= 3000" }}
       ]
     }}

6. No Combined Clause: 
   - Do not produce lines like "table_tags LIKE '%tag1%' AND table_tags LIKE '%tag2%'". 
   - Always split multi-tag queries into multiple items in the `sql_clauses` array.

Make sure your final output is valid JSON, with no extra keys, so each field can be parsed separately.
Now, please produce the final structured SQL clauses:
"""


# Define desired output structure
class Action(BaseModel):
    reset: bool
    refine: bool

    def get_true_fields(self):
        return [field for field, value in self.model_dump().items() if value]

class MentionedSemanticFields(BaseModel):
    table_schema: bool
    example_records: bool
    table_description: bool
    table_tags: bool

    def get_true_fields(self):
        return [field for field, value in self.model_dump().items() if value]

class MentionedRawFields(BaseModel):
    table_tags: bool
    column_numbers: bool
    row_numbers: bool
    temporal_granularity: bool
    geographic_granularity: bool

    def get_true_fields(self):
        return [field for field, value in self.model_dump().items() if value]

class SQLClause(BaseModel):
    field: str
    clause: str

class TextToSQL(BaseModel):
    sql_clauses: List[SQLClause]


def infer_action(cur_query, prev_query):
    try:
        prompt = format_prompt(PROMPT_ACTION_INFER, cur_query=cur_query, prev_query=prev_query)

        response_model = Action

        messages = [
            {"role": "system", "content": "You are an assistant skilled in search related decision making."},
            {"role": "user", "content": prompt}
        ]

        return openai_client.infer_metadata(messages, response_model)
    except Exception as e:
        logging.error(f"Failed to infer reset/refine action: {e}")
        raise RuntimeError("Failed to process the action inference.") from e

def infer_mentioned_metadata_fields(cur_query, semantic_metadata=True):
    try:
        if semantic_metadata:
            prompt = format_prompt(PROMPT_SEMANTIC_METADATA_INFER, cur_query=cur_query)
            response_model = MentionedSemanticFields
        else:
            prompt = format_prompt(PROMPT_RAW_METADATA_INFER, cur_query=cur_query)
            response_model = MentionedRawFields
        
        messages = [
            {"role": "system", "content": "You are an assistant skilled in search related decision making"},
            {"role": "user", "content": prompt}
        ]

        return openai_client.infer_metadata(messages, response_model)
    except Exception as e:
        logging.error(f"Failed to infer metadata fields: {e}")
        raise RuntimeError("Failed to process the metadata inference.") from e

def text_to_sql(cur_query, identified_fields):
    try:
        prompt = format_prompt(PROMPT_SQL_TRANSLATION, cur_query=cur_query, identified_fields=identified_fields)

        response_model = TextToSQL

        messages = [
            {"role": "system", "content": "You are an assistant skilled in text to SQL translation, and designed to output JSON."},
            {"role": "user", "content": prompt}
        ]

        return openai_client.infer_metadata(messages, response_model)
    except Exception as e:
        logging.error(f"Failed to translate text to SQL: {e}")
        raise RuntimeError("Failed to process the SQL translation.") from e

def execute_sql(text_to_sql_instance, search_space, table_name="eval_data_validation"):
    field_to_column_mapping = {
        'table_tags': 'tags',  # text[] in the DB
        'column_numbers': 'col_num',  # integer
        'row_numbers': 'row_num',  # integer
        'temporal_granularity': 'time_granu',  # a single text column
        'geographic_granularity': 'geo_granu'  # also a single text column
    }

    # Define finer granularity levels
    time_granu_order = ['second', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year']
    geo_granu_order = ['zip code/postal code', 'city', 'county/district', 'state/province', 'country', 'continent']

    with DatabaseConnection() as db:
        # Base query: either from search_space or "WHERE 1=1"
        if search_space:
            query_base = sql.SQL("SELECT DISTINCT table_name, popularity FROM {} WHERE table_name = ANY(%s)").format(
                sql.Identifier(table_name)
            )
            parameters = [search_space]
        else:
            query_base = sql.SQL("SELECT DISTINCT table_name, popularity FROM {} WHERE 1=1").format(
                sql.Identifier(table_name)
            )
            parameters = []

        where_conditions = []
        ordering = []

        for clause in text_to_sql_instance.sql_clauses:
            # Identify the DB column
            db_field = field_to_column_mapping.get(clause.field.lower())
            if not db_field:
                logging.warning(f"Unrecognized metadata field: {clause.field}")
                continue

            # Handle ORDER BY
            if 'ORDER BY' in clause.clause:
                parts = clause.clause.split()
                direction = parts[-1]  # e.g. "DESC" or "ASC"
                ordering.append(sql.SQL("{} {}").format(sql.Identifier(db_field), sql.SQL(direction)))
                continue

            # Otherwise parse operator + value
            clause_str = clause.clause.strip()  # Remove leading/trailing spaces
            operator_value = clause_str.split(None, 1)  # Split on whitespace

            if len(operator_value) == 2:
                operator, raw_value = operator_value
            else:
                # Fallback if we can't split properly
                operator = '='
                raw_value = operator_value[0] if operator_value else ''

            # Strip quotes and convert to lowercase
            raw_value = raw_value.strip("'").lower().strip()

            if db_field == 'time_granu':
                # Parse time granularity
                finer = get_finer_granularities(raw_value, time_granu_order)
                if not finer:
                    logging.info(f"No recognized time granularity for '{raw_value}'; skipping condition.")
                    continue
                # Now we check if time_granu is IN that set of granularities
                # e.g. time_granu = ANY('{day,week,month,quarter,year}')
                condition = sql.SQL("LOWER({}) = ANY(%s::text[])").format(sql.Identifier(db_field))
                where_conditions.append(condition)
                parameters.append([g.lower() for g in finer])

            elif db_field == 'geo_granu':
                # Parse geographic granularity
                finer = get_finer_granularities(raw_value, geo_granu_order)
                if not finer:
                    logging.info(f"No recognized geo granularity for '{raw_value}'; skipping condition.")
                    continue
                condition = sql.SQL("LOWER({}) = ANY(%s::text[])").format(sql.Identifier(db_field))
                where_conditions.append(condition)
                parameters.append([g.lower() for g in finer])

            elif db_field == 'tags':
                # e.g. = 'finance'
                if operator == '=':
                    condition = sql.SQL("EXISTS (SELECT 1 FROM unnest({}) AS t WHERE LOWER(t) = %s)").format(
                        sql.Identifier(db_field)
                    )
                    where_conditions.append(condition)
                    parameters.append(raw_value)
                elif operator.upper() == 'LIKE':
                    # e.g. "LIKE 'business'"
                    cond = sql.SQL("EXISTS (SELECT 1 FROM unnest({}) AS t WHERE LOWER(t) LIKE %s)").format(
                        sql.Identifier(db_field)
                    )
                    where_conditions.append(cond)
                    parameters.append(f"%{raw_value}%")
                else:
                    # Fallback: treat as =
                    cond = sql.SQL("EXISTS (SELECT 1 FROM unnest({}) AS t WHERE LOWER(t) = %s)").format(
                        sql.Identifier(db_field)
                    )
                    where_conditions.append(cond)
                    parameters.append(raw_value)

            else:
                # Basic numeric/string columns (col_num, row_num, etc.)
                # e.g. operator = '>=', raw_value='7'
                condition = sql.SQL("{} {} %s").format(sql.Identifier(db_field), sql.SQL(operator))
                where_conditions.append(condition)
                parameters.append(raw_value)

        # Combine extra WHERE conditions
        if where_conditions:
            query_base = query_base + sql.SQL(" AND ") + sql.SQL(" AND ").join(where_conditions)
        # Add any ordering
        if ordering:
            query_base = query_base + sql.SQL(" ORDER BY ") + sql.Composed(ordering)

        # Execute the query
        try:
            db.cursor.execute(query_base, parameters)
            results = db.cursor.fetchall()
            return results
        except Exception as e:
            logging.error(f"SQL execution failed: {e}")
            return []