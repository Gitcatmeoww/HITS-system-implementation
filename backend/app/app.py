from flask import Flask, request, jsonify
import asyncio
from flask_cors import CORS
import logging
from backend.app.table_representation.openai_client import OpenAIClient
from backend.app.hyse.hypo_schema_search import hyse_search
from backend.app.utils import check_run_status

# Flask app configuration
app = Flask(__name__)
CORS(app)

# OpenAI client instantiation
openai_client = OpenAIClient()
# Set up a single assistant for all threads
assistant_id = None  
ASSISTANT_NAME = "Semantic Dataset Search"
# TODO: further craft the assistant's instructions
ASSISTANT_INSTRU = """
    You are an AI assistant designed to aid users in locating specific datasets relevant to their analytical tasks. 
    Your responses and actions are guided by user queries, aiming to streamline the search process and deliver precise results. 
    Here's how you should operate:

    1. Initial Interaction:
    For the first message from a user, always reply with: "Here are the initial possible results according to your query.", excluding any introductory phrases.

    2. Subsequent Interactions:
    After the initial message, evaluate the user's new input to decide if they want to "reset" their search or "refine" it. 
    A "reset" occurs when the new input significantly differs from the previous queries (e.g., mentioning entirely new metadata fields or changing the search domain).
    A "refine" happens when the input elaborates on or slightly alters the existing query (e.g., adding more specific conditions on already mentioned fields).
    For example, if a previous query was about "population data" and the new query adds "in Europe", treat this as a "refine".
    If the new query changes the topic entirely, e.g., to "climate data", treat it as a "reset".

    3. Metadata Fields Consideration:
    Identify which metadata fields need to be reset or refined based on the user's latest input. 
    Only include relevant fields in the analysis. These fields may include:
        Table Name
        Table Schema
        Example Records
        Table Description
        Table Tags
        Column Numbers
        Previous Queries
        Temporal Granularity
        Geographic Granularity
        Popularity
    
    4. Query Processing:
    If the identified metadata fields are among [Table Name, Column Numbers, Popularity, Temporal Granularity, Geographic Granularity], 
    convert the text to SQL for precise database queries.
    
    5. Response Formatting:
    When processing user queries, format your response as a structured JSON object. Use the following template to construct your responses:
    {   
        "is_initial_query": [Boolean, true if it is the first message from a user],
        "reset": [Boolean, determined by if the user's query significantly deviates from previous topics],
        "refine": [Boolean, true if the user's query narrows down a previous search without changing topics],
        "mentioned_metadata_fields": [Array, list all metadata fields mentioned in the user's query],
        "where_clauses": [Array, construct 'where' clauses based on the user's query to simulate SQL query conditions]
    }

    For example, if a user's query is to refine their search on datasets by adding a condition on the 'year' field, your response should resemble the following format:

    {
        "is_initial_query": false,
        "reset": false,
        "refine": true,
        "mentioned_metadata_fields": ["year"],
        "where_clauses": ["year > 2020"]
    }

    Ensure that the 'reset' and 'refine' flags are mutually exclusive and accurately reflect the user's intent based on their latest query compared to their search history.
    Important: DO NOT output any additional text outside of the JSON, except for the first reply is always "Here are the initial possible results according to your query".
"""

# Important: DO NOT output any additional text outside of the JSON, except for the first reply is always "Here are the initial possible results according to your query".

# ASSISTANT_TOOL = [{
#     "type": "function",
#     "function": {
#         "name": "response",
#         "description": "Decide if users want to 'reset' their search or 'refine' it",
#         "parameters": {
#             "type": "object",
#             "required": [
#                 "is_initial_query",
#                 "reset",
#                 "refine",
#                 "mentioned_metadata_fields",
#                 "where_clauses",
#             ],
#             "properties": {
#                 "is_initial_query": {
#                     "type": "boolean",
#                     "description": "true if it is the first message from a user"
#                 },
#                 "reset": {
#                     "type": "boolean",
#                     "description": "true if the user's query significantly deviates from previous topics"
#                 },
#                 "refine": {
#                     "type": "boolean",
#                     "description": "true if the user's query narrows down a previous search without changing topics"
#                 },
#                 "mentioned_metadata_fields": {
#                     "type": "string",
#                     "description": "list all metadata fields mentioned in the user's query, separated by colon"
#                 },
#                 "where_clauses": {
#                     "type": "string",
#                     "description": "list 'where' clauses based on the user's query to simulate SQL query conditions, separated by colon"
#                 },
#             }
#         },
#     }
#  }]

# Logging configuration
logging.basicConfig(level=logging.INFO)


@app.route('/api/start_chat', methods=['POST'])
def start_conversation():
    global assistant_id
    if not assistant_id:
        try:
            # Create or fetch an existing assistant id
            assistant = openai_client.create_assistant(
                name=ASSISTANT_NAME,
                instructions=ASSISTANT_INSTRU
            )
            assistant_id = assistant.id
        except Exception as e:
            logging.error(f"Assistant creation failed, Error: {e}")
            return jsonify({"error": "Assistant creation failed due to an internal error"}), 500
    thread = openai_client.create_thread()
    return jsonify({"thread_id": thread.id, "assistant_id": assistant_id})

@app.route('/api/infer_action', methods=['POST'])
async def handle_query():
    # Extract the necessary information from the request
    thread_id = request.json.get('thread_id')
    user_query = request.json.get('query')

    if not thread_id or not user_query:
        return jsonify({"error": "Thread ID and query are required"}), 400

    try:
        # Log & run the user's query as a message in the thread
        openai_client.create_message(thread_id=thread_id, role="user", content=user_query)         
        run = openai_client.run_thread(thread_id=thread_id, assistant_id=assistant_id)
        
        # Retrieve the status of the thread
        if await check_run_status(thread_id, run.id, openai_client):
            # Once the run status reaches completed, list the messages in the thread again which should now include the response to our latest question
            messages = openai_client.list_thread_messages(thread_id=thread_id).data[0]
            print(messages.model_dump_json(indent=2))

            return jsonify(messages.model_dump_json(indent=2)), 200
        else:
            return jsonify({"error": "Timeout waiting for completion"}), 408
    except Exception as e:
        logging.error(f"Failed to handle query for thread {thread_id}, Error: {e}")
        return jsonify({"error": "Failed to process the query"}), 500

@app.route('/api/hyse_search', methods=['POST'])
def initial_search():
    initial_query = request.json.get('query')

    if not initial_query or len(initial_query.strip()) == 0:
        logging.error("Empty query provided")
        return jsonify({"error": "No query provided"}), 400

    try:
        initial_results = hyse_search(initial_query)
        logging.info(f"Search successful for query: {initial_query}")
        return jsonify(initial_results), 200
    except Exception as e:
        logging.error(f"Search failed for query: {initial_query}, Error: {e}")
        return jsonify({"error": "Search failed due to an internal error"}), 500

# @app.route('/api/query_sim_search', methods=['POST'])

# @app.route('/api/reset_search_space', methods=['POST'])

# @app.route('/api/prune_search_space', methods=['POST'])
# def prune_search_space():
#     query = request.json.get('query')
#     thread_id = request.json.get('thread_id')

#     # Create a message in the thread
#     openai_client.create_message(thread_id, "user", query)
#     # Run the thread with the assistant
#     response = openai_client.run_thread(thread_id, assistant_id)
#     return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
