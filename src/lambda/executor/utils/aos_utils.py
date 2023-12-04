import json
import boto3
import requests
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection

credentials = boto3.Session().get_credentials()

region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

IMPORT_OPENSEARCH_PY_ERROR = (
    "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
)
def _import_not_found_error():
    """Import not found error if available, otherwise raise error."""
    try:
        from opensearchpy.exceptions import NotFoundError
    except ImportError:
        raise ImportError(IMPORT_OPENSEARCH_PY_ERROR)
    return NotFoundError

class LLMBotOpenSearchClient:
    def __init__(self, host):
        """
        Initialize OpenSearch client using OpenSearch Endpoint
        """
        self.client = OpenSearch(
            hosts = [{'host': host.replace("https://", ""), 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        self.query_match = {"knn": self._build_knn_search_query,
                            "exact": self._build_exactly_match_query,
                            "basic": self._build_basic_search_query}
    
    def _build_basic_search_query(self, index_name, query_term, field, size, filter=None):
        """
        Build basic search query

        :param index_name: Target Index Name
        :param query_term: query term
        :param field: search field
        :param size: number of results to return from aos
        
        :return: aos response json
        """
        query = {
                "size": size,
                "query": {
                    "bool":{
                        "should": [ {"match": { field : query_term }} ],
                    }
                },
                "sort": [
                    {
                        "_score": {
                            "order": "desc"
                        }
                    }
                ]
            }
        if filter:
            query["query"]["bool"]["filter"] = filter

        return query
    
    def _build_knn_search_query(self, index_name, query_term, field, size, filter=None):
        """
        Build knn search query

        :param index_name: Target Index Name
        :param query_term: query term
        :param field: search field
        :param size: number of results to return from aos
        
        :return: aos response json
        """
        if filter:
            query = {
                "size": size,
                "query" : { 
                    "bool": {
                        "filter": {
                            "bool": {
                                "must": filter
                            }
                        },
                        "must": [
                            {
                                "knn": {
                                    field: {
                                        "vector": query_term,
                                        "k": size
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        else:
            query = {
                "size": size,
                "query": {
                    "knn": {
                        field: {
                            "vector": query_term,
                            "k": size
                        }
                    }
                }
            }
        return query
    
    def _build_exactly_match_query(self, index_name, query_term, field, size):
        """
        Build exactly match query

        :param index_name: Target Index Name
        :param query_term: query term
        :param field: search field
        :param size: number of results to return from aos
        
        :return: aos response json
        """
        query =  {
            "query" : {
                "match_phrase":{
                    field : query_term
                }
            }
        }
        return query

    def organize_results(self, query_type, response, field):
        """
        Organize results from aos response

        :param query_type: query type
        :param response: aos response json
        """
        results = []
        aos_hits = response["hits"]["hits"]
        if query_type == "exact":
            for aos_hit in aos_hits:
                doc = aos_hit['_source'][field]
                source = aos_hit['_source']['metadata']['source']
                score = aos_hit["_score"]
                results.append({'doc': doc, 'score': score, 'source': source})
        else:
            for aos_hit in aos_hits:
                doc = f"{aos_hit['_source'][field]}"
                source = aos_hit['_source']['metadata']['source']
                score = aos_hit["_score"]
                results.append({'doc': doc, 'score': score, 'source': source})
        return results

    def search(self, index_name, query_type, query_term, field: str = "text", size: int = 10, filter=None):
        """
        Perform search on aos
        
        :param index_name: Target Index Name
        :param query_type: query type
        :param query_term: query term
        :param field: search field
        :param size: number of results to return from aos
        
        :return: aos response json
        """
        not_found_error = _import_not_found_error()
        try:
            self.client.indices.get(index=index_name)
        except not_found_error:
            return []
        query = self.query_match[query_type](index_name, query_term, field, size, filter)
        response = self.client.search(
            body=query,
            index=index_name
        )
        return response 