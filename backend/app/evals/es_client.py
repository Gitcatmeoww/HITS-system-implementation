import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

class ElasticSearchClient:
    def __init__(self):
        self.client = Elasticsearch(
            os.environ['AZURE_ES_ENDPOINT'],
            api_key=os.environ['AZURE_ES_API_KEY'],
        )
    
    def test_connection(self):
        try:
            health = self.client.cluster.health()
            print("Cluster health:", health)
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {e}")


# Initialize the Elasticsearch client
es_client = ElasticSearchClient()

# Test the connection
es_client.test_connection()