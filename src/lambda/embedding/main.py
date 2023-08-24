import os
import sys
import time
import json
import boto3
import datetime
import hashlib
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers

logger = logging.getLogger()

# fetch all the environment variables
_document_bucket = os.environ.get('document_bucket')
_opensearch_cluster_domain = os.environ.get('opensearch_cluster_domain')

s3 = boto3.resource('s3')
aws_region = boto3.Session().region_name
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, aws_region)

index_body = {
    "settings" : {
        "index":{
            "number_of_shards" : 1,
            "number_of_replicas" : 0,
            "knn": "true",
            "knn.algo_param.ef_search": 32
        }
    },
    "mappings": {
        "properties": {
            "publish_date" : {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss"
            },
            "idx" : {
                "type": "integer"
            },
            "doc_type" : {
                "type" : "Sentence"
            },
            "doc": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "doc_title": {
                "type": "keyword"
            },
            "doc_category": {
                "type": "keyword"
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 32
                    }
                }            
            }
        }
    }
}

# AOS_ENDPOINT = "vpc-domain66ac69e0-s8fhzjmrdeuy-ylxlocb22mslvy4ko2owavcwli.us-west-2.es.amazonaws.com"
# REGION = "us-west-2"
# INDEX_NAME = 'chatbot-index'

def load_content_json_from_s3(bucket, object_key):
    """
    Load json content from s3
    """

    obj = s3.Object(bucket,object_key)
    file_content = obj.get()['Body'].read()
    
    json_content = json.loads(file_content)
    
    return json_content

def iterate_paragraph(json_content, object_key, index_name):
    """
    Iterate over the paragraphs in the json content and yield a document for each paragraph
    """
    doc_title = object_key

    publish_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for sent_id, sent in enumerate(json_content):
        paragraph_content = sent[0]
        embedding = sent[1]
        document = { "publish_date": publish_date, "idx":sent_id, "doc" : paragraph_content, "doc_type" : "Sentence", "content" : paragraph_content, "doc_title": doc_title, "doc_category": "", "embedding" : embedding}
        yield {"_index": index_name, "_source": document, "_id": hashlib.md5(str(document).encode('utf-8')).hexdigest()}


def WriteVecIndexToAOS(bucket, object_key, aos_endpoint, index_name):
    """
    Write the vector index to aos cluster
    """

    try:
        file_content = load_content_json_from_s3(bucket, object_key)

        client = OpenSearch(
            hosts = [{'host': aos_endpoint, 'port': 443}],
            http_auth = auth,
            use_ssl = True,
            verify_certs = True,
            connection_class = RequestsHttpConnection
        )
    except Exception as e:
        print(f"There was an error when connecting to aos cluster, Exception: {str(e)}")
        return '' 

    try:
        response = client.indices.create(index_name)
        print('\nCreating index:')
        print(response)
    except Exception as e:
        # index_name already exists
        client.indices.delete(index = index_name)
        response = client.indices.create(index_name)
        print('\nCreating index:')
        print(response)

    try:
        gen_aos_record_func = iterate_paragraph(file_content, object_key, index_name)

        response = helpers.bulk(client, gen_aos_record_func)
        return response
    
    except Exception as e:
        print(f"There was an error when ingest:{object_key} to aos cluster, Exception: {str(e)}")
        return '' 
    
def lambda_handler(event, context):
    # TODO implement
    request_timestamp = time.time()
    logger.info(f'request_timestamp :{request_timestamp}')
    logger.info(f"event:{event}")
    logger.info(f"context:{context}")

    # parse aos endpoint from event
    index_name = event['aos_index']
    object_key = event['object_key']

    WriteVecIndexToAOS(_document_bucket, object_key, aos_endpoint=_opensearch_cluster_domain, index_name=index_name)
    return {
        'statusCode': 200,
        'body': json.dumps('Finished Writing to AOS!')
    }
