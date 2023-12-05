import boto3 
import json 
import requests 
import re
import traceback



from ..sm_utils import SagemakerEndpointVectorOrCross
from ..llmbot_utils import concat_recall_knowledge
from .prompt_template import claude2_rag_template_render


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
    region_name = 'us-east-1'

    default_generate_kwargs = {
        "max_tokens_to_sample": 2000,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    @classmethod
    def create_client(cls):
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=cls.region_name
            )
        return bedrock

    @classmethod
    def _generate(cls,prompt,**generate_kwargs):
        if cls.client is None:
            cls.client = cls.create_client()

        generate_kwargs = dict(cls.default_generate_kwargs.copy(),**generate_kwargs)
        
        body = json.dumps(dict(generate_kwargs,prompt=f"\n\nHuman:{prompt}\n\nAssistant:"))

        response = cls.client.invoke_model(body=body, modelId=cls.modelId, accept=cls.accept, contentType=cls.contentType)

        response_body = json.loads(response.get('body').read())
        # text
        return response_body.get('completion')
    
    @classmethod
    def generate_rag(cls,**kwargs):
        query = kwargs['query']
        contexts = kwargs['contexts']
        context_num = kwargs.get('context_num',2)
        prompt = claude2_rag_template_render(
            query,
            [context['doc'] for context in contexts[:context_num]]
        )
        extracted_generate_kwargs = {k:kwargs[k] for k in cls.default_generate_kwargs if k in kwargs}
        answer = cls._generate(prompt,**extracted_generate_kwargs)
        
        answer = cls.postprocess(answer)
        return {
            "answer":answer,
            "prompt":prompt
        }
    
    @classmethod 
    def postprocess(cls,answer):
        rets = re.findall('<result>(.*?)</result>',answer,re.S)
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