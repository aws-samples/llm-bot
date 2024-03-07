# embeddings
import os
import json 
import boto3
from typing import List,Dict
from langchain_community.embeddings.sagemaker_endpoint import (
    SagemakerEndpointEmbeddings,
)
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler

class BGEEmbeddingSagemakerEndpoint:
    class vectorContentHandler(EmbeddingsContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
            input_str = json.dumps({"inputs": inputs, **model_kwargs})
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> List[List[float]]:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json["sentence_embeddings"]

    def __new__(cls,endpoint_name,region_name=os.environ['AWS_REGION']):
        client = boto3.client(
            "sagemaker-runtime",
            region_name=region_name
        )
        content_handler = cls.vectorContentHandler()
        embedding  = SagemakerEndpointEmbeddings(
            client=client,
            endpoint_name=endpoint_name,
            content_handler=content_handler
        )
        return embedding
