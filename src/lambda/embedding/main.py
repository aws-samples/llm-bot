import os
import time
import json
import logging
import numpy as np
import boto3, json
import tempfile
import nltk

from langchain.document_loaders import S3DirectoryLoader
# from langchain.vectorstores import OpenSearchVectorSearch
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from opensearch_vector_search import OpenSearchVectorSearch
from sm_utils import create_sagemaker_embeddings_from_js_model
from requests_aws4auth import AWS4Auth
from aos_utils import OpenSearchClient

from opensearchpy import OpenSearch, RequestsHttpConnection

# global constants
MAX_FILE_SIZE = 1024*1024*1024 # 1GB
MAX_OS_DOCS_PER_PUT = 2
CHUNK_SIZE_FOR_DOC_SPLIT = 600
CHUNK_OVERLAP_FOR_DOC_SPLIT = 20

logger = logging.getLogger()
# logging.basicConfig(format='%(asctime)s,%(module)s,%(processName)s,%(levelname)s,%(message)s', level=logging.INFO, stream=sys.stderr)
logger.setLevel(logging.INFO)

# fetch all the environment variables
_document_bucket = os.environ.get('document_bucket')
_embeddings_model_endpoint_name = os.environ.get('embedding_endpoint')
_opensearch_cluster_domain = os.environ.get('opensearch_cluster_domain')

s3 = boto3.resource('s3')
aws_region = boto3.Session().region_name
document_bucket = s3.Bucket(_document_bucket)
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, aws_region, 'es', session_token=credentials.token)

def load_documents(prefix=""):
    docs = []
    for obj in document_bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):   # bypass the prefix directory
            continue
        else:
            # loader = S3FileLoader(bucket, obj.key)
            with tempfile.TemporaryDirectory(dir='/tmp') as temp_dir:
                file_path = f"{temp_dir}/{obj.key}"
                logging.info(f"_document_bucket={_document_bucket}, obj.key={obj.key}, file_path={file_path}")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                s3.meta.client.download_file(_document_bucket, obj.key, file_path)

                loader = UnstructuredFileLoader(file_path)
                # return loader.load()
                docs.extend(loader.load())
    return docs

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = CHUNK_SIZE_FOR_DOC_SPLIT,
        chunk_overlap = CHUNK_OVERLAP_FOR_DOC_SPLIT,
        length_function = len,
    )

    # add a custom metadata field, timestamp and embeddings_model
    for doc in docs:
        doc.metadata['timestamp'] = time.time()
        doc.metadata['embeddings_model'] = _embeddings_model_endpoint_name
    chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
    return chunks

def load_processed_documents(prefix=""):
    chunks = []
    for obj in document_bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):   # bypass the prefix directory
            continue
        else:
            # loader = S3FileLoader(bucket, obj.key)
            with tempfile.TemporaryDirectory(dir='/tmp') as temp_dir:
                file_path = f"{temp_dir}/{obj.key}"
                logging.info(f"_document_bucket={_document_bucket}, obj.key={obj.key}, file_path={file_path}")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                s3.meta.client.download_file(_document_bucket, obj.key, file_path)

                file_content = json.load(open(file_path, 'r'))
                return file_content
                # for raw_chunk in file_content:
                #     chunk_source = raw_chunk.get('source') if isinstance(raw_chunk.get('source'), str) else "CSDC & DGR Data 20230830"
                #     chunk = Document(page_content=raw_chunk['content'], metadata={"source": chunk_source})
                #     chunk = Document(page_content=raw_chunk['content'], metadata={"source": chunk_source})
                #     chunks.append(chunk)
    
    for chunk in chunks:
        chunk.metadata['timestamp'] = time.time()
        chunk.metadata['embeddings_model'] = _embeddings_model_endpoint_name

    return chunks

def process_shard(shard, embeddings_model_endpoint_name, aws_region, os_index_name, os_domain_ep, os_http_auth, shard_id, doc_type="repost") -> int: 
    # logger.info(f'Starting process_shard with content: {shard}')
    st = time.time()
    embeddings = create_sagemaker_embeddings_from_js_model(embeddings_model_endpoint_name, aws_region)
    docsearch = OpenSearchVectorSearch(
        index_name=os_index_name,
        embedding_function=embeddings,
        opensearch_url="https://{}".format(os_domain_ep),
        http_auth = os_http_auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    # docsearch.add_documents(documents=shard)
    if doc_type == "faq":
        docsearch.add_documents(documents=shard, ids=range(shard_id*len(shard), (shard_id+1)*len(shard)))
    elif doc_type == "ug":
        docsearch.add_ug_documents(documents=shard, ids=range(shard_id*len(shard), (shard_id+1)*len(shard)))
    et = time.time() - st
    logger.info(f'Shard completed in {et} seconds.')
    return 0

def lambda_handler(event, context):
    request_timestamp = time.time()
    logger.info(f'request_timestamp :{request_timestamp}')
    logger.info(f"event:{event}")
    logger.info(f"context:{context}")

    # parse arguments from event
    index_name = json.loads(event['body'])['aos_index']

    # re-route GET request to seperate processing branch
    if event['httpMethod'] == 'GET':
        query = json.loads(event['body'])['query']
        aos_client = OpenSearchClient(_opensearch_cluster_domain)
        # check if the operation is query of search for OpenSearch
        if query['operation'] == 'query':
            response = aos_client.query(index_name, query['field'], query['value'])
        elif query['operation'] == 'match_all':
            response = aos_client.match_all(index_name)
        else:
            raise Exception(f'Invalid query operation: {query["operation"]}')

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(response)
        }

    # parse arguments from event
    prefix = json.loads(event['body'])['document_prefix']
    doc_type = json.loads(event['body'])['doc_type']
    file_processed = json.loads(event['body']).get('file_processed', False)

    # Set the NLTK data path to the /tmp directory (writable in AWS Lambda)
    nltk.data.path.append("/tmp")
    # List of NLTK packages to download
    nltk_packages = ['punkt', 'averaged_perceptron_tagger']
    # Download the required NLTK packages to /tmp
    for package in nltk_packages:
        nltk.download(package, download_dir='/tmp')

    aos_client = OpenSearch(
        hosts = [{'host': _opensearch_cluster_domain.replace("https://", ""), 'port': 443}],
        http_auth = awsauth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        region=aws_region
    )

    # iterate all files within specific s3 prefix in bucket llm-bot-documents and print out file number and total size
    total_size = 0
    total_files = 0
    for obj in document_bucket.objects.filter(Prefix=prefix):
        total_files += 1
        total_size += obj.size
    logger.info(f'total_files:{total_files}, total_size:{total_size}')
    # raise error and return if the total size is larger than 100MB
    if total_size > MAX_FILE_SIZE:
        raise Exception(f'total_size:{total_size} is larger than {MAX_FILE_SIZE}')
    
    # split all docs into chunks
    st = time.time()
    logger.info('Loading documents ...')
    if file_processed:
        chunks = load_processed_documents(prefix=prefix)
    else:
        docs = load_documents(prefix=prefix)
        chunks = split_documents(docs)

    et = time.time() - st
    # [Document(page_content = 'xx', metadata = { 'source': '/tmp/xx/xx.pdf', 'timestamp': 123.456, 'embeddings_model': 'embedding-endpoint'})],
    logger.info(f'Time taken: {et} seconds. {len(chunks)} chunks generated')

    st = time.time()
    db_shards = (len(chunks) // MAX_OS_DOCS_PER_PUT) + 1
    shards = np.array_split(chunks, db_shards)
    # logger.info(f'Loading chunks into vector store ... using {db_shards} shards, shards content: {shards}')

    # TBD, create index if not exists instead of using API in AOS console manually
    # Reply: Langchain has already implemented the code to create index if not exists
    # Refer Link: https://github.com/langchain-ai/langchain/blob/eb3d1fa93caa26d497e5b5bdf6134d266f6a6990/libs/langchain/langchain/vectorstores/opensearch_vector_search.py#L120
    exists = aos_client.indices.exists(index_name)
    logger.info(f"index_name={index_name}, exists={exists}")

    # shard_start_index = 1
    for shard_id, shard in enumerate(shards):
        process_shard(shards[shard_id].tolist(), _embeddings_model_endpoint_name, aws_region, index_name, _opensearch_cluster_domain, awsauth, shard_id, doc_type)

    et = time.time() - st
    logger.info(f'Time taken: {et} seconds. all shards processed')

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
            "created": request_timestamp,
            "model": _embeddings_model_endpoint_name,            
        })
    }
