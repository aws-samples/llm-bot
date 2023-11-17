import json
import logging
import os
import boto3
import time
import copy
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
# aos_index = os.environ.get("aos_index", "")
aos_faq_index = os.environ.get("aos_faq_index", "")
aos_ug_index = os.environ.get("aos_ug_index", "")

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

def organize_faq_results(response):
    """
    Organize results from aos response

    :param query_type: query type
    :param response: aos response json
    """
    results = []
    if not response:
        return results
    aos_hits = response["hits"]["hits"]
    for aos_hit in aos_hits:
        result = {}
        try:
            result["doc"] = aos_hit['_source']['text']
            result["source"] = aos_hit['_source']['metadata']['source']
            result["score"] = aos_hit["_score"]
            result["detail"] = aos_hit['_source']
            result["answer"] = aos_hit['_source']['answer']
        except:
            print("index_error")
            print(aos_hit['_source'])
            continue
        # result.update(aos_hit["_source"])
        results.append(result)
    return results

def organize_ug_results(response):
    """
    Organize results from aos response

    :param query_type: query type
    :param response: aos response json
    """
    results = []
    aos_hits = response["hits"]["hits"]
    for aos_hit in aos_hits:
        result = {}
        result["doc"] = f"{aos_hit['_source']['title']} {aos_hit['_source']['content']}"
        result["source"] = aos_hit['_source']['metadata']['source']
        result["score"] = aos_hit["_score"]
        result["detail"] = aos_hit['_source']
        # result.update(aos_hit["_source"])
        results.append(result)
    return results

def remove_redundancy_debug_info(results):
    filtered_results = copy.deepcopy(results)
    for result in filtered_results:
        for field in list(result["detail"].keys()):
            if field.endswith("_vector"):
                del result["detail"][field]
    return filtered_results

def get_answer(query_input:str, history:list, embedding_model_endpoint:str, cross_model_endpoint:str, 
               llm_model_endpoint:str, aos_faq_index:str, aos_ug_index:str, enable_knowledge_qa:bool, temperature: float, enable_q_q_match:bool):
    # 1. concatenate query_input and history to unified prompt
    query_knowledge = ''.join([query_input] + [row[0] for row in history][::-1])
    debug_info = {
        "query": query_input,
        "q_q_match_info": {},
        "knowledge_qa_knn_recall": {},
        "knowledge_qa_boolean_recall": {},
        "knowledge_qa_combined_recall": {},
        "knowledge_qa_cross_model_sort": {},
        "knowledge_qa_llm": {},
    }

    # 2. get AOS q-q-knn recall 
    start = time.time()
    # query_embedding = SagemakerEndpointVectorOrCross(prompt="为这个句子生成表示以用于检索相关文章：" + query_knowledge, endpoint_name=embedding_model_endpoint, region_name=region, model_type="vector", stop=None)
    query_embedding = SagemakerEndpointVectorOrCross(query_knowledge, endpoint_name=embedding_model_endpoint, region_name=region, model_type="vector", stop=None)
    if enable_q_q_match:
        opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn", query_term=query_embedding, field="title_vector")
        opensearch_knn_results = organize_faq_results(opensearch_knn_response)
        # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_knn : {elpase_time}s seconds')
        if len(opensearch_knn_results) > 0:
            debug_info["q_q_match_info"] = remove_redundancy_debug_info(opensearch_knn_results[:3])
            if opensearch_knn_results[0]["score"] >= 0.9:
                answer = opensearch_knn_results[0]["answer"]
                sources = [opensearch_knn_results[0]["source"]]
                recall_knowledge_str = ""
                query_type = QueryType.KnowledgeQuery
                return answer, query_type, sources, recall_knowledge_str, debug_info
    if enable_knowledge_qa:
        # 2. get AOS knn recall 
        faq_result_num = 3
        ug_result_num = 3
        start = time.time()
        opensearch_knn_results = []
        opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn", query_term=query_embedding, field="text_vector")
        opensearch_knn_results.extend(organize_faq_results(opensearch_knn_response)[:faq_result_num])
        # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
        opensearch_knn_response = aos_client.search(index_name=aos_ug_index, query_type="knn", query_term=query_embedding, field="title_vector")
        opensearch_knn_results.extend(organize_ug_results(opensearch_knn_response)[:ug_result_num])
        debug_info["knowledge_qa_knn_recall"] = remove_redundancy_debug_info(opensearch_knn_results)
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_knn : {elpase_time}s seconds')
        
        # 3. get AOS invertedIndex recall
        start = time.time()
        opensearch_query_results = []
        # opensearch_query_response = aos_client.search(index_name=aos_faq_index, query_type="basic", query_term=query_knowledge, field="text")
        # opensearch_query_results.extend(organize_faq_results(opensearch_query_response))
        # opensearch_query_response = aos_client.search(index_name=aos_ug_index, query_type="basic", query_term=query_knowledge, field="title")
        # opensearch_query_results.extend(organize_ug_results(opensearch_query_response))
        # logger.info(json.dumps(opensearch_query_response, ensure_ascii=False))
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_query : {elpase_time}s seconds')
        debug_info["knowledge_qa_boolean_recall"] = remove_redundancy_debug_info(opensearch_query_results[:20])

        # 4. combine these two opensearch_knn_response and opensearch_query_response
        recall_knowledge = combine_recalls(opensearch_knn_results, opensearch_query_results)
        recall_knowledge.sort(key=lambda x: x["score"], reverse=True)
        debug_info["knowledge_qa_combined_recall"] = recall_knowledge[:40]
        
        # 5. Predict correlation score using cross model
        # recall_knowledge_cross = []
        # for knowledge in recall_knowledge:
        #     # get score using cross model
        #     score = float(SagemakerEndpointVectorOrCross(prompt=query_knowledge, endpoint_name=cross_model_endpoint, region_name=region, model_type="cross", stop=None, context=knowledge['doc']))
        #     # logger.info(json.dumps({'doc': knowledge['doc'], 'score': score, 'source': knowledge['source']}, ensure_ascii=False))
        #     if score > 0.8:
        #         recall_knowledge_cross.append({'doc': knowledge['doc'], 'score': score, 'source': knowledge['source']})

        # recall_knowledge_cross.sort(key=lambda x: x["score"], reverse=True)
        # debug_info["knowledge_qa_cross_model_sort"] = recall_knowledge_cross[:10]

        # recall_knowledge_str = concat_recall_knowledge(recall_knowledge_cross[:2])
        recall_knowledge_str = concat_recall_knowledge(recall_knowledge[:2])
        # sources = list(set([item["source"] for item in recall_knowledge_cross[:2]]))
        sources = list(set([item["source"] for item in recall_knowledge[:2]]))
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
        answer = SagemakerEndpointVectorOrCross(prompt=query_input,
                                                endpoint_name=llm_model_endpoint,
                                                region_name=region,
                                                model_type="answer",
                                                stop=None,
                                                history=history,
                                                parameters=parameters,
                                                context=recall_knowledge_str[:2560])
        debug_info["knowledge_qa_llm"] = {"prompt": query_input, "context": recall_knowledge_str, "answer": answer}
    except Exception as e:
        logger.info(f'Exceptions: str({e})')
        answer = ""
    return answer, query_type, sources, recall_knowledge_str, debug_info

def main_entry(session_id:str, query_input:str, history:list, embedding_model_endpoint:str, cross_model_endpoint:str, 
               llm_model_endpoint:str, aos_faq_index:str, aos_ug_index:str, enable_knowledge_qa:bool, temperature: float, enable_q_q_match:bool):
    """
    Entry point for the Lambda function.

    :param session_id: The ID of the session.
    :param query_input: The query input.
    :param history: The history of the conversation.
    :param embedding_model_endpoint: The endpoint of the embedding model.
    :param cross_model_endpoint: The endpoint of the cross model.
    :param llm_model_endpoint: The endpoint of the language model.
    :param llm_model_name: The name of the language model.
    :param aos_faq_index: The faq index of the AOS engine.
    :param aos_ug_index: The ug index of the AOS engine.
    :param enable_knowledge_qa: Whether to enable knowledge QA.
    :param temperature: The temperature of the language model.

    return: answer(str)
    """
    answer, query_type, sources, recall_knowledge_str, debug_info = get_answer(query_input,
               history,
               embedding_model_endpoint,
               cross_model_endpoint,
               llm_model_endpoint,
               aos_faq_index,
               aos_ug_index,
               enable_knowledge_qa,
               temperature,
               enable_q_q_match)
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
    # aos_faq_index = event_body['aos_faq_index']
    # aos_ug_index = event_body['aos_ug_index']
    messages = event_body['messages']
    temperature = event_body['temperature']
    enable_q_q_match = event_body['enable_q_q_match']

    history, question = process_input_messages(messages)
    role = "user"
    session_id = f"{role}_{int(request_timestamp)}"
    knowledge_qa_flag = True if model == 'knowledge_qa' else False
    
    main_entry_start = time.time() 
    answer, sources, debug_info = main_entry(session_id, question, history, embedding_endpoint, cross_endpoint, llm_endpoint, aos_faq_index, aos_ug_index, knowledge_qa_flag, temperature, enable_q_q_match)
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
        'body': json.dumps(llmbot_response),
        # 'body': llmbot_response,
        'debug_info': json.dumps(debug_info)
        # 'debug_info': debug_info
    }
