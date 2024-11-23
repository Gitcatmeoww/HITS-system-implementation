import logging
import os
import csv
from tqdm import tqdm
from dotenv import load_dotenv
from backend.app.db.connect_db import DatabaseConnection
from backend.app.evals.eval_methods import EvalMethods
from eval_utils import get_ground_truth_header, get_hypo_schema

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluator:
    def __init__(self, data_split="eval_data_validation", embed_col="example_rows_embed", k=10, limit=None, num_embed=1):
        self.data_split = data_split
        self.embed_col = embed_col
        self.k = k
        self.limit = limit
        self.num_embed = num_embed
        self.eval_methods = EvalMethods(data_split=data_split, embed_col=embed_col, k=k)
        self.db_connection = DatabaseConnection()
        self.ground_truths = []
        self.task_queries = []
        self.keywords = []
        self.results_dir = "eval/results"
        self.results_file = os.path.join(self.results_dir, "per_row_results.csv")
        self.failed_rows_file = os.path.join(self.results_dir, "failed_rows.csv")
        self.failed_queries_file = os.path.join(self.results_dir, "failed_queries.csv")
        self.load_data()
        self.initialize_results_files()

    def load_data(self):
        try:
            with self.db_connection as db:
                query = f"SELECT table_name, task_queries, keywords FROM {self.data_split}"
                if self.limit:
                    query += f" LIMIT {self.limit};"
                else:
                    query += ";"
                db.cursor.execute(query)
                rows = db.cursor.fetchall()

                for row in rows:
                    self.ground_truths.append(row['table_name'])
                    self.task_queries.append(row['task_queries'])
                    self.keywords.append(row['keywords'])
        except Exception as e:
            logging.exception(f"Error loading data from the database: {e}")

    # Initialize the results files by writing the headers if they don't exist
    def initialize_results_files(self):
        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize per_row_results.csv
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'Index', 'Table Name', 'Method', 'Query Type', 'Query',
                    'Precision', 'Ground Truth Header', 'Hypothetical Schema'
                ]
                writer.writerow(header)
            logging.info(f"Initialized results file '{self.results_file}'")

        # Initialize failed_rows.csv
        if not os.path.exists(self.failed_rows_file):
            with open(self.failed_rows_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['Index', 'Table Name', 'Error Message']
                writer.writerow(header)
            logging.info(f"Initialized failed rows file '{self.failed_rows_file}'")

        # Initialize failed_queries.csv
        if not os.path.exists(self.failed_queries_file):
            with open(self.failed_queries_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['Index', 'Table Name', 'Method', 'Query Type', 'Query', 'Error Message']
                writer.writerow(header)
            logging.info(f"Initialized failed queries file '{self.failed_queries_file}'")

    # Compute precision @k for a single query
    def compute_precision_at_k(self, results_list, ground_truth):
        return 1 if ground_truth in results_list else 0

    # Run the evaluation across all methods & compute average precision @k
    def evaluate(self):
        # Initialize performance metrics
        methods = [
            {'name': 'HySE Search', 'function': self.eval_methods.single_hyse_search, 'query_type': 'task'},
            {'name': 'Semantic Task Search', 'function': self.eval_methods.semantic_search, 'query_type': 'task'},
            {'name': 'Semantic Keyword Search', 'function': self.eval_methods.semantic_search, 'query_type': 'keyword'},
            # {'name': 'Syntactic Keyword Search', 'function': self.eval_methods.syntactic_search, 'query_type': 'keyword'}
        ]
        precisions = {method['name']: [] for method in methods}

        # Iterate over each entry
        for idx, ground_truth_table in tqdm(enumerate(self.ground_truths), total=len(self.ground_truths), desc="Evaluating", unit="entry"):
            try:
                task_queries = self.task_queries[idx]
                keywords = self.keywords[idx]

                # Combine queries and methods
                for method in methods:
                    method_name = method['name']
                    search_function = method['function']
                    query_type = method['query_type']
                    queries = task_queries if query_type == 'task' else keywords

                    for query in queries:
                        try:
                            if method_name == 'HySE Search':
                                # Pass `num_embed` parameter
                                results = search_function(query=query, num_embed=self.num_embed)
                            else:        
                                results = search_function(query=query)
                            precision = self.compute_precision_at_k(results, ground_truth_table)
                            precisions[method_name].append(precision)

                            # Retrieve ground truth header & hypothetical schema for HySE search
                            ground_truth_header = ''
                            hypo_schema = ''

                            if method_name == 'HySE Search':
                                # Retrieve ground truth header from example_rows_md
                                ground_truth_header = get_ground_truth_header(ground_truth_table, self.data_split)
                                # Retrieve hypothetical schema from eval_hyse_schemas
                                hypo_schema = get_hypo_schema(query)

                            self.save_row_result(
                                idx, ground_truth_table, method_name, query_type, query, precision,
                                ground_truth_header, hypo_schema
                            )
                        except Exception as e:
                            logging.exception(f"Error in {method_name} with query '{query}' at index {idx}: {e}")
                            precisions[method_name].append(0)
                            self.save_failed_query(idx, ground_truth_table, method_name, query_type, query, str(e))
            except Exception as e:
                logging.exception(f"Error processing row {idx} (table: {ground_truth_table}): {e}")
                self.save_failed_row(idx, ground_truth_table, str(e))

        # Compute average precision @k for each method
        avg_precisions = {
            method_name: sum(scores) / len(scores) if scores else 0
            for method_name, scores in precisions.items()
        }

        # Report the results
        for method_name, avg_precision in avg_precisions.items():
            logging.info(f"Average Precision @{self.k} for {method_name}: {avg_precision}")

        # Return the results
        return avg_precisions
    
    def save_row_result(self, idx, table_name, method_name, query_type, query, precision, ground_truth_header='', hypothetical_schema=''):
        try:
            with open(self.results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    idx, table_name, method_name, query_type, query, precision,
                    ground_truth_header, hypothetical_schema
                ])
        except Exception as e:
            logging.exception(f"Error saving result for index {idx}: {e}")

    def save_failed_row(self, idx, table_name, error_message):
        try:
            with open(self.failed_rows_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([idx, table_name, error_message])
        except Exception as e:
            logging.exception(f"Error saving failed row for index {idx}: {e}")

    def save_failed_query(self, idx, table_name, method_name, query_type, query, error_message):
        try:
            with open(self.failed_queries_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([idx, table_name, method_name, query_type, query, error_message])
        except Exception as e:
            logging.exception(f"Error saving failed query for index {idx}: {e}")


if __name__ == "__main__":
    evaluator = Evaluator(
        data_split="eval_data_validation",
        embed_col="example_rows_embed",
        k=10,
        limit=150,
        num_embed=2
    )

    results = evaluator.evaluate()
    print("Evaluation Results:")
    for method, precision in results.items():
        print(f"{method}: {precision}")
