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

from .opensearch_vector_search import OpenSearchVectorSearch
from .sm_utils import create_sagemaker_embeddings_from_js_model

from opensearchpy import OpenSearch, RequestsHttpConnection

logger = logging.getLogger()

def load_processed_documents(document_bucket_name, prefix=""):
    file_content = []
    s3 = boto3.resource('s3')
    document_bucket = s3.Bucket(document_bucket_name)
    for obj in document_bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/"):   # bypass the prefix directory
            continue
        else:
            # loader = S3FileLoader(bucket, obj.key)
            with tempfile.TemporaryDirectory(dir='/tmp') as temp_dir:
                file_path = f"{temp_dir}/{obj.key}"
                logging.info(f"_document_bucket={document_bucket_name}, obj.key={obj.key}, file_path={file_path}")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                s3.meta.client.download_file(document_bucket_name, obj.key, file_path)

                file_content.extend(json.load(open(file_path, 'r')))
                # for raw_chunk in file_content:
                #     chunk_source = raw_chunk.get('source') if isinstance(raw_chunk.get('source'), str) else "CSDC & DGR Data 20230830"
                #     chunk = Document(page_content=raw_chunk['content'], metadata={"source": chunk_source})
                #     chunk = Document(page_content=raw_chunk['content'], metadata={"source": chunk_source})
                #     chunks.append(chunk)
    # for doc in file_content:
    #     doc["metadata"]['timestamp'] = time.time()
    #     doc["metadata"]['embeddings_model'] = _embeddings_model_endpoint_name
    return file_content

def process_shard(shard, embedding_model_info_list, aws_region, os_index_name, os_domain_ep, os_http_auth, shard_id, doc_type="faq", max_os_docs_per_put=2) -> int: 
    # logger.info(f'Starting process_shard with content: {shard}')
    st = time.time()
    for embedding_model_info in embedding_model_info_list:
        embedding_function = create_sagemaker_embeddings_from_js_model(
                embedding_model_info["endpoint_name"],
                aws_region,
                embedding_model_info["lang"],
                embedding_model_info["type"])
        docsearch = OpenSearchVectorSearch(
            index_name=os_index_name,
            embedding_function=embedding_function,
            opensearch_url="https://{}".format(os_domain_ep),
            http_auth = os_http_auth,
            use_ssl = True,
            verify_certs = True,
            connection_class = RequestsHttpConnection
        )
        # docsearch.add_documents(documents=shard)
        if doc_type == "faq":
            docsearch.add_faq_documents_v2(documents=shard,
                                    ids=range(max_os_docs_per_put*shard_id, max_os_docs_per_put*shard_id+len(shard)),
                                    embedding_lang=embedding_model_info["lang"],
                                    embedding_type=embedding_model_info["type"])
        elif doc_type == "ug":
            docsearch.add_ug_documents_v2(documents=shard,
                                       ids=range(max_os_docs_per_put*shard_id, max_os_docs_per_put*shard_id+len(shard)),
                                       embedding_lang=embedding_model_info["lang"],
                                       embedding_type=embedding_model_info["type"])
    et = time.time() - st
    logger.info(f'Shard completed in {et} seconds.')
    return 0