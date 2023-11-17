"""
Helper functions for using Samgemaker Endpoint via langchain
"""
import sys
import time
import json
import logging
import traceback
from typing import List
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

logger = logging.getLogger()
# logging.basicConfig(format='%(asctime)s,%(module)s,%(processName)s,%(levelname)s,%(message)s', level=logging.INFO, stream=sys.stderr)
logger.setLevel(logging.INFO)

# extend the SagemakerEndpointEmbeddings class from langchain to provide a custom embedding function
class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(
        self, texts: List[str], chunk_size: int = 5
    ) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size
        st = time.time()
        for i in range(0, len(texts), _chunk_size):
            embedding_texts = [text[:(512-56)] for text in texts[i:i + _chunk_size]]
            try:
                response = self._embedding_func(embedding_texts)
            except Exception as error:
                traceback.print_exc()
                print(f"embedding endpoint error: {texts}", error)
            results.extend(response)
        time_taken = time.time() - st
        logger.info(f"got results for {len(texts)} in {time_taken}s, length of embeddings list is {len(results)}")
        return results


# class for serializing/deserializing requests/responses to/from the embeddings model
class SimilarityZhContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs={}) -> bytes:
        # add bge_prompt to each element in prompt
        new_prompt = [p for p in prompt]
        input_str = json.dumps({"inputs": new_prompt, **model_kwargs})
        return input_str.encode('utf-8') 

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["sentence_embeddings"]
        if len(embeddings) == 1:
            return [embeddings[0]]
        return embeddings

class RelevanceZhContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs={}) -> bytes:
        # add bge_prompt to each element in prompt
        new_prompt = ["为这个句子生成表示以用于检索相关文章：" + p for p in prompt]
        input_str = json.dumps({"inputs": new_prompt, **model_kwargs})
        return input_str.encode('utf-8') 

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["sentence_embeddings"]
        if len(embeddings) == 1:
            return [embeddings[0]]
        return embeddings

class SimilarityEnContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs={}) -> bytes:
        # add bge_prompt to each element in prompt
        new_prompt = [p for p in prompt]
        input_str = json.dumps({"inputs": new_prompt, **model_kwargs})
        return input_str.encode('utf-8') 

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["sentence_embeddings"]
        if len(embeddings) == 1:
            return [embeddings[0]]
        return embeddings

class RelevanceEnContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs={}) -> bytes:
        # add bge_prompt to each element in prompt
        new_prompt = ["Represent this sentence for searching relevant passages:" + p for p in prompt]
        input_str = json.dumps({"inputs": new_prompt, **model_kwargs})
        return input_str.encode('utf-8') 

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        embeddings = response_json["sentence_embeddings"]
        if len(embeddings) == 1:
            return [embeddings[0]]
        return embeddings


def create_sagemaker_embeddings_from_js_model(embeddings_model_endpoint_name: str, aws_region: str, lang: str = "zh", type: str = "similarity") -> SagemakerEndpointEmbeddingsJumpStart:
    # all set to create the objects for the ContentHandler and 
    # SagemakerEndpointEmbeddingsJumpStart classes
    if lang == "zh":
        if type == "similarity":
            content_handler = SimilarityZhContentHandler()
        elif type == "relevance":
            content_handler = RelevanceZhContentHandler()
    elif lang == "en":
        if type == "similarity":
            content_handler = SimilarityEnContentHandler()
        elif type == "relevance":
            content_handler = RelevanceEnContentHandler()
    logger.info(f'content_handler: {content_handler}, embeddings_model_endpoint_name: {embeddings_model_endpoint_name}, aws_region: {aws_region}')
    # note the name of the LLM Sagemaker endpoint, this is the model that we would
    # be using for generating the embeddings
    embeddings = SagemakerEndpointEmbeddingsJumpStart( 
        endpoint_name = embeddings_model_endpoint_name,
        region_name = aws_region, 
        content_handler = content_handler
    )
    return embeddings