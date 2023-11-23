import os
import boto3
import json
from tenacity import retry, stop_after_attempt
from typing import List

from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection

from langchain.docstore.document import Document
from langchain.vectorstores import OpenSearchVectorSearch

from sm_utils import create_sagemaker_embeddings_from_js_model

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

# Get params from Lambda environment variables
aosEndpoint = os.environ.get('opensearch_cluster_domain')
embeddingModelEndpoint = os.environ.get('embedding_endpoint')

class OpenSearchClient:
    def __init__(self, _opensearch_cluster_domain: str):
        """
        Initialize OpenSearch client using OpenSearch Endpoint
        """
        self.client = OpenSearch(
            hosts = [{'host': _opensearch_cluster_domain.replace("https://", ""), 'port': 443}],
            http_auth = awsauth,
            use_ssl = True,
            verify_certs = True,
            connection_class = RequestsHttpConnection,
            region=region
        )

    def validation(self, index: str, _body: str):
        # avoid NotFoundError: NotFoundError(404, 'index_not_found_exception'...
        if not self.client.indices.exists(index=index):
            # hint to the caller that the index does not exist
            return {
                'statusCode': 404,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': f'index {index} does not exist'})
            }

    def create_index(self, index: str, body: str, _kwargs: dict):
        """
        Create an index in OpenSearch.

        Args:
            index (str): The name of the index to create.
            body (dict): A dictionary containing the settings and mappings for the index.

        Sample body:
        {
            "aos_index": "chatbot-index",
            "operation": "create",
            "body": {
                "settings": {
                "index": {
                    "number_of_shards": 2,
                    "number_of_replicas": 1
                }
                },
                "mappings": {
                "properties": {
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 1024
                    }
                }
                }
            }
        }
        """
        body_dict = json.loads(body)
        # Extract the settings and mappings from the body
        settings = body_dict.get('body', {}).get('settings', {})
        mappings = body_dict.get('body', {}).get('mappings', {})

        # Create the index with the specified settings and mappings
        self.client.indices.create(
            index=index,
            body={
                'settings': settings,
                'mappings': mappings
            }
        )

    def delete_index(self, index: str, _body: str, _kwargs: dict):
        """
        Delete an index in OpenSearch.

        Args:
            index (str): The name of the index to delete.
        """
        # break if the validation fails
        self.validation(index, _body)
        # delete the index
        self.client.indices.delete(index=index)

    def delete_document(self, index: str, body: str, _kwargs: dict):
        """
        Delete a document in a specific index.

        Args:
            index (str): The name of the index to delete.
            document_id (str): The id of the document to delete.

        Sample body:
        {
            "aos_index": "chatbot-index",
            "operation": "delete_document",
            "body": {
                "document_id": "1"
            }
        }
        """
        self.validation(index, body)
        body_dict = json.loads(body)
        document_id = body_dict.get('body', {}).get('document_id', {})
        # delete the document
        self.client.delete(index=index, id=document_id)

    def bulk(self, index: str, body: str, _kwargs: dict):
        """
        Bulk index documents in a specific index.

        Args:
            index (str): The name of the index to delete.
            document (List[Document]): A list of documents to index.

        Sample body:
        {
            "aos_index": "chatbot-index",
            "operation": "bulk",
            "body": {
                List[Document]
            }
        }
        """
        self.validation(index, body)
        body_dict = json.loads(body)
        document = body_dict.get('body', {})
        # bulk index the documents
        self.client.bulk(index=index, body=document)

    def index(self, index: str, body: str, _kwargs: dict):
        """
        Index a document in a specific index.

        Args:
            index (str): The name of the index to delete.
            document (List[Document]): A list of documents to index.

        Sample body:
        {
            "aos_index": "chatbot-index",
            "operation": "index",
            "body": {
                List[Document]
            }
        }
        """
        self.validation(index, body)
        body_dict = json.loads(body)
        document = body_dict.get('body', {})
        # iterate through the documents and index them
        for doc in document:
            try:
                response = self.client.index(index=index, body=doc)
                logger.info(f"response: {response}")
            except Exception as e:
                logger.error(f"Error indexing document: {e}")

    def query(self, index: str, body: str, _kwargs: dict):
        """
        Basic query with fixed result size

        Args:
            index (str): The name of the index to delete.
            body (str): The query body.

        Sample body:
        {
            "aos_index": "chatbot-index",
            "operation": "query",
            "body": {
                "field": field,
                "value": value,
                "size": size
            }
        }
        """
        self.validation(index, body)
        body_dict = json.loads(body)
        field = str(body_dict.get('field'))
        value = str(body_dict.get('value'))
        # optional size with default value 100
        size = str(body_dict.get('size', 100))
        logger.info(f"field: {field}, value: {value}, size: {size}")
        body = {
            "query": {
                "match": {
                    field: value
                }
            },
            "size": size,
            "sort": [
                {
                    "_score": {
                        "order": "desc"
                    }
                }
            ]
        }
        response = self.client.search(index=index, body=body)
        return response

    def query_all(self, index: str, _body: str, _kwargs: dict):
        self.validation(index, _body)
        body = {
            "query": {
                "match_all": {}
            }
        }
        response = self.client.search(index=index, body=body)
        return response

    def query_with_must_and_filter(self, index: str, body: str, _kwargs: dict):
        """
        Execute a search query using the query DSL, using bool query to filter on metadata.

        Args:
            index (str): The name of the index to delete.
            body (str): The query body.

        Sample body:
        {
            "aos_index": "chatbot-index",
            "operation": "query_with_must_and_filter",
            "body": {
                "quey": query,
                "filter": filter
                "size": size
            }
        }
        """
        self.validation(index, body)
        body_dict = json.loads(body)
        query = str(body_dict.get('query'))
        filter = str(body_dict.get('filter'))
        # optional size with default value 100
        size = str(body_dict.get('size', 100))
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"text": query}},
                    ],
                    # looking for documents where the metadata field exactly matches the value of filter
                    "filter": [
                        {"term": {"metadata": filter}}
                    ]
                }
            }
        }
        response = self.client.search(index=index, body=body)
        return response

    def query_knn(self, index: str, body: str, _kwargs: dict):
        """
        Execute a search query using knn.

        Args:
            index (str): The name of the index to delete.
            body (str): The query body.

        Sample body:
        {
            "aos_index": "chatbot-index",
            "operation": "query_knn",
            "body": {
                "query": query,
                "field": field,
                "size": size
            }
        }
        """
        self.validation(index, body)
        body_dict = json.loads(body)
        query = str(body_dict.get('query'))
        field = str(body_dict.get('field'))
        # optional size with default value 100
        size = str(body_dict.get('size', 100))
        body = {
            "size": size,
            "query": {
                "knn": {
                    "vector_field": {
                        "vector": query,
                        "k": size
                    }
                }
            }
        }
        response = self.client.search(index=index, body=body)
        return response

    def query_exact(self, index: str, body: str, _kwargs: dict):
        """
        Execute a search query using exact match.

        Args:
            index (str): The name of the index to delete.
            body (str): The query body.

        Sample body:
        {
            "aos_index": "chatbot-index",
            "operation": "query_exact",
            "body": {
                "query": query,
                "field": field,
                "size": size
            }
        }
        """
        self.validation(index, body)
        body_dict = json.loads(body)
        query = str(body_dict.get('query'))
        field = str(body_dict.get('field'))
        # optional size with default value 100
        size = str(body_dict.get('size', 100))
        body =  {
            "query" : {
                "match_phrase":{
                    field : query
                }
            }
        }
        response = self.client.search(index=index, body=body)
        return response

    @retry(stop=stop_after_attempt(3))
    def embed_document(self, index: str, body: str, _kwargs: dict):
        """
        Returns:
            List of ids from adding the texts into the vectorstore.

        Sample body:
        {
            "index": "chatbot-index",
            "operation": "embed_document",
            "body": {
                "documents": {
                    "page_content": "This is a test",
                    "metadata": {
                        "heading_hierarchy": "Title 1"
                        ...
                    }
                }
            }
        }
        """
        body_dict = json.loads(body)
        # assemble the document according to the Document class below:      
        document = Document(
            page_content=body_dict['documents']['page_content'],
            metadata=body_dict['documents']['metadata']
        )
        logger.info("embeddingModelEndpoint: {}, region: {}, aosEndpoint: {}".format(embeddingModelEndpoint, region, aosEndpoint))
        embeddings = create_sagemaker_embeddings_from_js_model(embeddingModelEndpoint, region)
        docsearch = OpenSearchVectorSearch(
            index_name=index,
            embedding_function=embeddings,
            opensearch_url="https://{}".format(aosEndpoint),
            http_auth = awsauth,
            use_ssl = True,
            verify_certs = True,
            connection_class = RequestsHttpConnection
        )
        logger.info("Adding documents %s to OpenSearch with index %s", document, index)
        res = docsearch.add_documents(documents=[document])
        logger.info("Retry statistics: %s and response: %s", embed_document.retry.statistics, res)
        return res