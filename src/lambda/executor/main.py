import json
import logging
import os
import boto3
import time
from aos_utils import LLMBotOpenSearchClient
from llmbot_utils import QueryType, combine_recalls, concat_recall_knowledge, process_input_messages
from ddb_utils import get_session, update_session
from sm_utils import SagemakerEndpointVectorOrCross

logger = logging.getLogger()
handler = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


region = os.environ['AWS_REGION']
embedding_endpoint = os.environ.get("embedding_endpoint", "")
cross_endpoint = os.environ.get("cross_endpoint", "")
aos_endpoint = os.environ.get("aos_endpoint", "")
aos_index = os.environ.get("aos_index", "")
llm_endpoint = os.environ.get('llm_endpoint', "")
chat_session_table = os.environ.get('chat_session_table', "")

sm_client = boto3.client("sagemaker-runtime")
aos_client = LLMBotOpenSearchClient(aos_endpoint)

class APIException(Exception):
    def __init__(self, message, code: str = None):
        if code:
            super().__init__("[{}] {}".format(code, message))
        else:
            super().__init__(message)

def handle_error(func):
    """Decorator for exception handling"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIException as e:
            logger.exception(e)
            raise e
        except Exception as e:
            logger.exception(e)
            raise RuntimeError(
                "Unknown exception, please check Lambda log for more details"
            )

    return wrapper


def organize_results(response):
    """
    Organize results from aos response

    :param query_type: query type
    :param response: aos response json
    """
    results = []
    aos_hits = response["hits"]["hits"]
    for aos_hit in aos_hits:
        result = {}
        result["doc"] = aos_hit['_source']['text']
        result["source"] = aos_hit['_source']['metadata']['source']
        result["score"] = aos_hit["_score"]
        result["title"] = aos_hit['_source']['title']
        result["content"] = aos_hit['_source']['content']
        result["answer"] = aos_hit['_source']['answer']
        # result.update(aos_hit["_source"])
        results.append(result)
    return results

def main_entry(session_id:str, query_input:str, history:list, embedding_model_endpoint:str, cross_model_endpoint:str, 
               llm_model_endpoint:str, aos_index:str, enable_knowledge_qa:bool, temperature: float, enable_q_q_match:bool):
    """
    Entry point for the Lambda function.

    :param session_id: The ID of the session.
    :param query_input: The query input.
    :param history: The history of the conversation.
    :param embedding_model_endpoint: The endpoint of the embedding model.
    :param cross_model_endpoint: The endpoint of the cross model.
    :param llm_model_endpoint: The endpoint of the language model.
    :param llm_model_name: The name of the language model.
    :param aos_index: The index of the AOS engine.
    :param enable_knowledge_qa: Whether to enable knowledge QA.
    :param temperature: The temperature of the language model.

    return: answer(str)
    """
    # 1. concatenate query_input and history to unified prompt
    query_knowledge = ''.join([query_input] + [row[0] for row in history][::-1])
    debug_info = {}

    # 2. get AOS q-q-knn recall 
    start = time.time()
    query_embedding = SagemakerEndpointVectorOrCross(prompt="为这个句子生成表示以用于检索相关文章：" + query_knowledge, endpoint_name=embedding_model_endpoint, region_name=region, model_type="vector", stop=None)
    opensearch_knn_response = aos_client.search(index_name=aos_index, query_type="knn", query_term=query_embedding, field="title_vector")
    opensearch_knn_response = organize_results(opensearch_knn_response)
    # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
    elpase_time = time.time() - start
    logger.info(f'runing time of opensearch_knn : {elpase_time}s seconds')
    if enable_q_q_match and len(opensearch_knn_response) > 0 and opensearch_knn_response[0]["score"] > 0.9:
        answer = opensearch_knn_response[0]["answer"]
        sources = [opensearch_knn_response[0]["source"]]
        recall_knowledge_str = ""
        query_type = QueryType.KnowledgeQuery
        debug_info["q_q_match_info"] = opensearch_knn_response[:1]
    else:
        if enable_knowledge_qa:
            # 2. get AOS knn recall 
            start = time.time()
            query_embedding = SagemakerEndpointVectorOrCross(prompt="为这个句子生成表示以用于检索相关文章：" + query_knowledge, endpoint_name=embedding_model_endpoint, region_name=region, model_type="vector", stop=None)
            opensearch_knn_response = aos_client.search(index_name=aos_index, query_type="knn", query_term=query_embedding, field="text_vector")
            opensearch_knn_response = organize_results(opensearch_knn_response)
            # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
            elpase_time = time.time() - start
            logger.info(f'runing time of opensearch_knn : {elpase_time}s seconds')
            debug_info["knowledge_qa_knn_recall"] = opensearch_knn_response[:10]
            
            # 3. get AOS invertedIndex recall
            start = time.time()
            opensearch_query_response = aos_client.search(index_name=aos_index, query_type="basic", query_term=query_knowledge)
            opensearch_query_response = organize_results(opensearch_query_response)
            # logger.info(json.dumps(opensearch_query_response, ensure_ascii=False))
            elpase_time = time.time() - start
            logger.info(f'runing time of opensearch_query : {elpase_time}s seconds')
            debug_info["knowledge_qa_boolean_recall"] = opensearch_query_response[:10]

            # 4. combine these two opensearch_knn_response and opensearch_query_response
            recall_knowledge = combine_recalls(opensearch_knn_response, opensearch_query_response)
            debug_info["knowledge_qa_combined_recall"] = recall_knowledge[:10]
            
            # 5. Predict correlation score using cross model
            recall_knowledge_cross = []
            for knowledge in recall_knowledge:
                # get score using cross model
                score = float(SagemakerEndpointVectorOrCross(prompt=query_knowledge, endpoint_name=cross_model_endpoint, region_name=region, model_type="cross", stop=None, context=knowledge['doc']))
                # logger.info(json.dumps({'doc': knowledge['doc'], 'score': score, 'source': knowledge['source']}, ensure_ascii=False))
                if score > 0.8:
                    recall_knowledge_cross.append({'doc': knowledge['doc'], 'score': score, 'source': knowledge['source']})

            recall_knowledge_cross.sort(key=lambda x: x["score"], reverse=True)
            debug_info["knowledge_qa_cross_model_sort"] = recall_knowledge_cross[:10]

            recall_knowledge_str = concat_recall_knowledge(recall_knowledge_cross[:2])
            sources = list(set([item["source"] for item in recall_knowledge_cross[:2]]))
            query_type = QueryType.KnowledgeQuery
            elpase_time = time.time() - start
            logger.info(f'runing time of recall knowledge : {elpase_time}s seconds')
        else:
            recall_knowledge_str = ""
            query_type = QueryType.Conversation

        # 6. generate answer using question and recall_knowledge
        parameters = {'temperature': temperature}
        try:
            # generate_answer
            answer = SagemakerEndpointVectorOrCross(prompt=query_input, endpoint_name=llm_model_endpoint, region_name=region, model_type="answer", stop=None, history=history, parameters=parameters, context=recall_knowledge_str)
        except Exception as e:
            logger.info(f'Exceptions: str({e})')
            answer = ""
    
    # 7. update_session
    start = time.time()
    update_session(session_id=session_id, chat_session_table=chat_session_table, 
                   question=query_input, answer=answer, knowledge_sources=sources)
    elpase_time = time.time() - start
    logger.info(f'runing time of update_session : {elpase_time}s seconds')

    # 8. log results
    json_obj = {
        "session_id": session_id,
        "query": query_input,
        "recall_knowledge_cross_str": recall_knowledge_str,
        "detect_query_type": str(query_type),
        "history": history,
        "chatbot_answer": answer,
        "sources": sources,
        "timestamp": int(time.time()),
        "debug_info": debug_info 
    }

    json_obj_str = json.dumps(json_obj, ensure_ascii=False)
    # logger.info(json_obj_str)

    return answer, sources, debug_info 

@handle_error
def lambda_handler(event, context):
    request_timestamp = time.time()
    logger.info(f'request_timestamp :{request_timestamp}')
    logger.info(f"event:{event}")
    logger.info(f"context:{context}")

    # Get request body
    event_body = json.loads(event['body'])
    model = event_body['model']
    messages = event_body['messages']
    temperature = event_body['temperature']
    enable_q_q_match = event_body['enable_q_q_match']

    history, question = process_input_messages(messages)
    role = "user"
    session_id = f"{role}_{int(request_timestamp)}"
    knowledge_qa_flag = True if model == 'knowledge_qa' else False
    
    main_entry_start = time.time() 
    answer, sources, debug_info = main_entry(session_id, question, history, embedding_endpoint, cross_endpoint, llm_endpoint, aos_index, knowledge_qa_flag, temperature, enable_q_q_match)
    main_entry_elpase = time.time() - main_entry_start  
    logger.info(f'runing time of main_entry : {main_entry_elpase}s seconds')

    llmbot_response = {
        "id": session_id,
        "object": "chat.completion",
        "created": int(request_timestamp),
        "model": model,
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": answer,
                    "knowledge_sources": sources
                },
                "finish_reason": "stop",
                "index": 0
            }
        ]
    }

    # 2. return rusult
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': llmbot_response,
        'debug_info': debug_info
    }
