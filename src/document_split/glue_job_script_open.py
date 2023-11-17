import os
import boto3
import sys
import re
import logging
import json
import itertools
import numpy as np

from typing import Generator, Any, Dict, Iterable, List, Optional, Tuple
from bs4 import BeautifulSoup
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from langchain.docstore.document import Document
# from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection

from llm_bot_dep import sm_utils, aos_utils, enhance_utils, loader_utils
from llm_bot_dep.opensearch_vector_search import OpenSearchVectorSearch
from llm_bot_dep.build_index import process_shard
from requests_aws4auth import AWS4Auth

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')

def process_json(jsonstr: str, max_os_docs_per_put, **kwargs):
    logger.info("Processing JSON file...")
    chunks = json.loads(jsonstr)

    db_shards = (len(chunks) // max_os_docs_per_put) + 1
    shards = np.array_split(chunks, db_shards)
    return shards

def cb_process_object(file_type: str, file_content, aos_endpoint, index_name, embeddings_model_info_list, region, awsauth, content_type, max_os_docs_per_put):
    res = None
    if file_type == 'json':
        shards = process_json(file_content, max_os_docs_per_put)
        # process_shard(shards, embeddingModelEndpoint, region, 'chatbot-index-9', aosEndpoint, awsauth, "ug", start_id)
        for shard_id, shard in enumerate(shards):
            process_shard(shard, embeddings_model_info_list, region, index_name, aos_endpoint, awsauth, 1, content_type, max_os_docs_per_put)
        # process_shard(shards, embeddingModelEndpoint, region, 'chatbot-faq-index-1', aosEndpoint, awsauth, "faq", start_id)
        
    return res

def iterate_s3_files(bucket: str, prefix: str) -> Generator:    
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            # skip the prefix with slash, which is the folder name
            if obj['Key'].endswith('/'):
                continue
            key = obj['Key']
            file_type = key.split('.')[-1]  # Extract file extension

            response = s3.get_object(Bucket=bucket, Key=key)
            file_content = response['Body'].read()
            # assemble bucket and key as args for the callback function
            kwargs = {'bucket': bucket, 'key': key}

            if file_type in ['txt', 'csv']:
                yield 'text', file_content.decode('utf-8'), kwargs
            elif file_type in ['html']:
                yield 'html', file_content.decode('utf-8'), kwargs
            elif file_type in ['pdf']:
                yield 'pdf', file_content, kwargs
            elif file_type in ['jpg', 'png']:
                yield 'image', file_content, kwargs
            elif file_type in ['json']:
                yield 'json', file_content.decode('utf-8'), kwargs
            else:
                logger.info(f"Unknown file type: {file_type}")

def batch_generator(generator, batch_size):
    while True:
        batch = list(itertools.islice(generator, batch_size))
        if not batch:
            break
        yield batch

def split_chunk(content: List[Document], embeddingModelEndpoint: str, aosEndpoint: str, index_name: str, chunk_size: int = 1000) -> List[Document]:
    embeddings = sm_utils.create_sagemaker_embeddings_from_js_model(embeddingModelEndpoint, region)

    def chunk_generator(content: List[Document], chunk_size: int = 1000):
        # iterate documents list and split per document with chunk size
        for i in range(0, len(content)):
            # TODO, split the document into chunks, will be deprecated and replaced by the ASK model directly
            chunks = [content[i].page_content[j:j+chunk_size] for j in range(0, len(content[i].page_content), chunk_size)]
            # create a new document for each chunk
            for chunk in chunks:
                metadata = content[i].metadata
                doc = Document(page_content=chunk, metadata=metadata)
                yield doc

    generator = chunk_generator(content, )
    batches = batch_generator(generator, batch_size=10)

    master_user_username = "icyxu"
    master_user_password = "OpenSearch1!"

    for batch in batches:
        if len(batch) == 0:
            continue
        logger.info("Adding documents %s to OpenSearch index...", batch)
        docsearch = OpenSearchVectorSearch(
            index_name=index_name,
            embedding_function=embeddings,
            opensearch_url="https://{}".format(aosEndpoint),
            http_auth = (master_user_username, master_user_password),
            use_ssl = True,
            verify_certs = True,
            connection_class = RequestsHttpConnection
        )
        docsearch.add_documents(documents=batch)

# main function to be called by Glue job script
def main(s3_bucket, s3_prefix, aos_endpoint, index_name, embedding_model_endpoint, region, awsauth, content_type, max_os_docs_per_put):
    # check if offline mode
    logger.info("Running in offline mode with consideration for large file size...")
    for file_type, file_content, kwargs in iterate_s3_files(s3_bucket, s3_prefix):
        start_id = int(s3_prefix.split(".")[0].split("_")[1])*100000
        # try:
        res = cb_process_object(file_type, file_content, aos_endpoint, index_name, embedding_model_endpoint, region, awsauth, content_type, max_os_docs_per_put)
        if res:
            logger.info("Result: %s", res)
        # except Exception as e:
        #     logger.error("Error processing object %s: %s", kwargs['bucket'] + '/' + kwargs['key'], e)

if __name__ == '__main__':
    # Parse arguments
    from awsglue.utils import getResolvedOptions
    args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_BUCKET', 'S3_PREFIX', 'AOS_ENDPOINT',
                                        'EMBEDDING_MODEL_ENDPOINT', 'REGION', 'OFFLINE', 'EMBEDDING_LANG',
                                        'EMBEDDING_TYPE', 'INDEX_NAME', 'CONTENT_TYPE'])
    logger.info("Starting Glue job with passing arguments: %s", args)
    _embedding_endpoint_name_list = args['EMBEDDING_MODEL_ENDPOINT'].split(',')
    _embedding_lang_list = args['EMBEDDING_LANG'].split(',')
    _embedding_type_list = args['EMBEDDING_TYPE'].split(',')

    embeddings_model_info_list = []
    for endpoint_name, lang, type in zip(_embedding_endpoint_name_list, _embedding_lang_list, _embedding_type_list):
        embeddings_model_info_list.append({
            "endpoint_name": endpoint_name,
            "lang": lang,
            "type": type})

    s3_bucket = args['S3_BUCKET']
    s3_prefix = args['S3_PREFIX']
    aos_endpoint = args['AOS_ENDPOINT']
    region = args['REGION']
    offline = args['OFFLINE']
    index_name = args['INDEX_NAME']
    content_type = args['CONTENT_TYPE']

    credentials = boto3.Session().get_credentials()
    # awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
    awsauth = AWS4Auth(region=region, service='es', refreshable_credentials=credentials)
    MAX_OS_DOCS_PER_PUT = 8

    logger.info("boto3 version: %s", boto3.__version__)
    main(s3_bucket, s3_prefix, aos_endpoint, index_name, embeddings_model_info_list, region, awsauth, content_type, MAX_OS_DOCS_PER_PUT)