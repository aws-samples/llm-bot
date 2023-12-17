import boto3 
import json 
import requests 
import re
import traceback

from sm_utils import SagemakerEndpointVectorOrCross
from llmbot_utils import concat_recall_knowledge
from prompt_template import claude2_rag_template_render

class StreamResponse:
    def __init__(self,raw_stream) -> None:
        self.raw_stream = raw_stream

    
    @staticmethod
    def postprocess(ret):
        return ret
    
    def __iter__(self):
        if self.raw_stream:
            for event in self.raw_stream:
                chunk = event.get("chunk")
                if chunk:
                    completion = json.loads(chunk.get("bytes").decode())['completion']
                    yield self.postprocess(completion)

    
    def close(self):
        self.raw_stream.close()


class ModelMeta(type):
    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        if name == 'Model':
            return new_cls
        new_cls.model_map[new_cls.modelId] = new_cls
        return new_cls
    
class Model(metaclass=ModelMeta):
    model_map = {}
    @classmethod
    def get_model(cls,model_id):
        return cls.model_map[model_id]

    @classmethod
    def generate(cls,*args,**kwargs):
        raise NotImplementedError


class Claude2(Model):
    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'
    client = None  

    default_generate_kwargs = {
        "max_tokens_to_sample": 2000,
        "temperature": 0.7,
        "top_p": 0.9
    }

    @classmethod
    def create_client(cls):
        bedrock = boto3.client(
            service_name='bedrock-runtime'
            )
        return bedrock

    @classmethod
    def generate_stream(cls,body):
        response = cls.client.invoke_model_with_response_stream(
            modelId=cls.modelId, body=body
        )
        stream = response.get("body")
        return StreamResponse(stream)
        

    @classmethod
    def _generate(cls,prompt,use_default_prompt_template=True,stream=False,**generate_kwargs):
        if cls.client is None:
            cls.client = cls.create_client()
        generate_kwargs = dict(cls.default_generate_kwargs.copy(),**generate_kwargs)
        
        if use_default_prompt_template:
            prompt=f"\n\nHuman:{prompt}\n\nAssistant:"
            
        body = json.dumps(dict(generate_kwargs,prompt=prompt))
       
        if stream:
            return cls.generate_stream(body)

        response = cls.client.invoke_model(body=body, modelId=cls.modelId, accept=cls.accept, contentType=cls.contentType)

        response_body = json.loads(response.get('body').read())
        # text
        return response_body.get('completion')
    
    @classmethod
    def generate_rag(cls,**kwargs):
        query = kwargs['query']
        contexts = kwargs['contexts']
        context_num = kwargs.get('context_num',2)
        stream = kwargs.get('stream',False)
        
        prompt = claude2_rag_template_render(
            query,
            [context['doc'] for context in contexts[:context_num]]
        )
        extracted_generate_kwargs = {k:kwargs[k] for k in cls.default_generate_kwargs if k in kwargs}
    
        answer = cls._generate(
            prompt,
            stream=stream,
            use_default_prompt_template=False,
            **extracted_generate_kwargs)
        if not stream:
            answer = cls.postprocess(answer)
        else:
            answer.postprocess = lambda x:x.rstrip('</result>')
        return {
            "answer":answer,
            "prompt":prompt
        }
    
    @classmethod 
    def postprocess(cls,answer):
        rets = re.findall('<result>(.*?)</result>','<result>'+ answer,re.S)
        rets = [ret.strip() for ret in rets]
        rets = [ret for ret in rets if ret]
        if not rets:
            return answer  
        return rets[0]
        
    generate = generate_rag


class ClaudeInstance(Claude2):
    modelId = 'anthropic.claude-instant-v1'


class Claude21(Claude2):
    modelId = 'anthropic.claude-v2:1'
    region_name = 'us-west-2'

class CSDCDGRModel(Model):
    modelId = 'csdc-internlm-7b'
    @classmethod
    def generate(
        cls,
        query:str,
        contexts:list,
        history,
        llm_model_endpoint,
        region_name,
        parameters,
        context_trunc_length:int=2560,
        model_type="answer",
        context_num=2,
        **kwargs
        ):
    
        # generate_answer
        recall_knowledge_str = concat_recall_knowledge(contexts[:context_num])
        answer = SagemakerEndpointVectorOrCross(prompt=query,
                                                endpoint_name=llm_model_endpoint,
                                                region_name=region_name,
                                                model_type=model_type,
                                                stop=None,
                                                history=history,
                                                parameters=parameters,
                                                context=recall_knowledge_str[:context_trunc_length])
        ret = {"prompt": query, "context": recall_knowledge_str, "answer": answer}
    
        return ret


def generate(model_id,**kwargs):
    if model_id is None:
        model_id = 'anthropic.claude-v2:1'
    model_cls = Model.get_model(model_id)
    ret = model_cls.generate(**kwargs)
    return ret 

from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
class CustomLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        model_cls = Model.get_model("anthropic.claude-v2")
        return model_cls._generate(prompt)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}