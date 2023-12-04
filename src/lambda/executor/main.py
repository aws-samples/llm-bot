import json
import logging
import os
import boto3
import time
import copy
import traceback
from preprocess import run_preprocess  
from aos_utils import LLMBotOpenSearchClient
from llmbot_utils import QueryType, combine_recalls, concat_recall_knowledge, process_input_messages
from ddb_utils import get_session, update_session
from sm_utils import SagemakerEndpointVectorOrCross
from llm import generate as llm_generate

logger = logging.getLogger()
handler = logging.StreamHandler()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


region = os.environ['AWS_REGION']
zh_embedding_endpoint = os.environ.get("zh_embedding_endpoint", "")
en_embedding_endpoint = os.environ.get("en_embedding_endpoint", "")
cross_endpoint = os.environ.get("cross_endpoint", "")
rerank_endpoint = os.environ.get("rerank_endpoint", "")
aos_endpoint = os.environ.get("aos_endpoint", "")
# aos_index = os.environ.get("aos_index", "")
aos_faq_index = os.environ.get("aos_faq_index", "")
aos_ug_index = os.environ.get("aos_ug_index", "")

llm_endpoint = os.environ.get('llm_endpoint', "")
chat_session_table = os.environ.get('chat_session_table', "")

sm_client = boto3.client("sagemaker-runtime")

# print(aos_endpoint)
# print(sfg)
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

def get_faq_answer(source, index_name):
    opensearch_query_response = aos_client.search(index_name=index_name,
                                                  query_type="basic", query_term=source,
                                                  field="metadata.source")
    for r in opensearch_query_response["hits"]["hits"]:
        if r["_source"]["metadata"]["field"] == "answer":
            return r["_source"]["content"]
    return ""

def get_faq_content(source, index_name):
    opensearch_query_response = aos_client.search(index_name=index_name,
                                                  query_type="basic", query_term=source,
                                                  field="metadata.source")
    for r in opensearch_query_response["hits"]["hits"]:
        if r["_source"]["metadata"]["field"] == "all_text":
            return r["_source"]["content"]
    return ""

def organize_faq_results(response, index_name):
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
            result["source"] = aos_hit['_source']['metadata']['source']
            result["score"] = aos_hit["_score"]
            result["detail"] = aos_hit['_source']
            result["content"] = aos_hit['_source']['content']
            result["answer"] = get_faq_answer(result["source"], index_name)
            result["doc"] = get_faq_content(result["source"], index_name)
        except:
            print("index_error")
            print(aos_hit['_source'])
            continue
        # result.update(aos_hit["_source"])
        results.append(result)
    return results

def get_ug_content(source, index_name):
    opensearch_query_response = aos_client.search(index_name=index_name,
                                                  query_type="basic", query_term=source,
                                                  field="metadata.source", size=100)
    for r in opensearch_query_response["hits"]["hits"]:
        if r["_source"]["metadata"]["field"] == "all_text":
            return r["_source"]["content"]
    return ""

def organize_ug_results(response, index_name):
    """
    Organize results from aos response

    :param query_type: query type
    :param response: aos response json
    """
    results = []
    aos_hits = response["hits"]["hits"]
    for aos_hit in aos_hits:
        result = {}
        result["source"] = aos_hit['_source']['metadata']['source']
        result["score"] = aos_hit["_score"]
        result["detail"] = aos_hit['_source']
        result["content"] = aos_hit['_source']['content']
        result["doc"] = get_ug_content(result["source"], index_name)
        # result.update(aos_hit["_source"])
        results.append(result)
    return results

def remove_redundancy_debug_info(results):
    filtered_results = copy.deepcopy(results)
    for result in filtered_results:
        for field in list(result["detail"].keys()):
            if field.endswith("embedding"):
                del result["detail"][field]
    return filtered_results

def get_answer(query_input:str, history:list, zh_embedding_model_endpoint:str, en_embedding_model_endpoint:str,
               cross_model_endpoint:str, rerank_model_endpoint:str, llm_model_endpoint:str,
               aos_faq_index:str, aos_ug_index:str, enable_knowledge_qa:bool, temperature: float, enable_q_q_match:bool,
               llm_model_id=None
            ):
    # 1. concatenate query_input and history to unified prompt
    query_knowledge = ''.join([query_input] + [row[0] for row in history][::-1])
    debug_info = {
        "query": query_input,
        "query_parser_info": {},
        "q_q_match_info": {},
        "knowledge_qa_knn_recall": {},
        "knowledge_qa_boolean_recall": {},
        "knowledge_qa_combined_recall": {},
        "knowledge_qa_cross_model_sort": {},
        "knowledge_qa_llm": {},
        "knowledge_qa_rerank": {},
    }

    # 2. get AOS q-q-knn recall 
    start = time.time()
    parsed_query = run_preprocess(query_knowledge) 
    debug_info["query_parser_info"] = parsed_query
    if parsed_query["query_lang"] == "zh":
        zh_query_similarity_embedding_prompt = query_knowledge
        zh_query_relevance_embedding_prompt = "为这个句子生成表示以用于检索相关文章：" + query_knowledge
        en_query_similarity_embedding_prompt = parsed_query["translated_text"]
        en_query_relevance_embedding_prompt = "Represent this sentence for searching relevant passages: " + parsed_query["translated_text"]
    elif parsed_query["query_lang"] == "en":
        zh_query_similarity_embedding_prompt = parsed_query["translated_text"]
        zh_query_relevance_embedding_prompt = "为这个句子生成表示以用于检索相关文章：" + parsed_query["translated_text"]
        en_query_similarity_embedding_prompt = query_knowledge
        en_query_relevance_embedding_prompt = "Represent this sentence for searching relevant passages: " + query_knowledge
    zh_query_similarity_embedding = SagemakerEndpointVectorOrCross(prompt=zh_query_similarity_embedding_prompt,
                                                                endpoint_name=zh_embedding_model_endpoint, region_name=region,
                                                                model_type="vector", stop=None)
    zh_query_relevance_embedding = SagemakerEndpointVectorOrCross(prompt=zh_query_relevance_embedding_prompt,
                                                                endpoint_name=zh_embedding_model_endpoint, region_name=region,
                                                                model_type="vector", stop=None)
    en_query_similarity_embedding = SagemakerEndpointVectorOrCross(prompt=en_query_similarity_embedding_prompt,
                                                                endpoint_name=en_embedding_model_endpoint, region_name=region,
                                                                model_type="vector", stop=None)
    en_query_relevance_embedding = SagemakerEndpointVectorOrCross(prompt=en_query_relevance_embedding_prompt,
                                                                endpoint_name=en_embedding_model_endpoint, region_name=region,
                                                                model_type="vector", stop=None)
    if enable_q_q_match:
        opensearch_knn_results = []
        opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn",
                                                    query_term=zh_query_similarity_embedding, field="embedding", size=2)
        opensearch_knn_results.extend(organize_faq_results(opensearch_knn_response, aos_faq_index))
        opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn",
                                                    query_term=en_query_similarity_embedding, field="embedding", size=2)
        opensearch_knn_results.extend(organize_faq_results(opensearch_knn_response, aos_faq_index))
        # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
        elpase_time = time.time() - start
        logger.info(f'runing time of opensearch_knn : {elpase_time}s seconds')
        if len(opensearch_knn_results) > 0:
            debug_info["q_q_match_info"] = remove_redundancy_debug_info(opensearch_knn_results[:3])
            if opensearch_knn_results[0]["score"] >= 0.9:
                source = opensearch_knn_results[0]["source"]
                answer = opensearch_knn_results[0]["answer"]
                sources = [source]
                recall_knowledge_str = ""
                query_type = QueryType.KnowledgeQuery
                return answer, query_type, sources, recall_knowledge_str, debug_info
    if enable_knowledge_qa:
        # 2. get AOS knn recall 
        faq_result_num = 2
        ug_result_num = 100
        start = time.time()
        opensearch_knn_results = []
        opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn",
                                                    query_term=zh_query_relevance_embedding, field="embedding", size=faq_result_num)
        opensearch_knn_results.extend(organize_faq_results(opensearch_knn_response, aos_faq_index)[:faq_result_num])
        opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn",
                                                    query_term=en_query_relevance_embedding, field="embedding", size=faq_result_num)
        opensearch_knn_results.extend(organize_faq_results(opensearch_knn_response, aos_faq_index)[:faq_result_num])
        # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
        filter = None
        if parsed_query["is_api_query"]:
            filter = [{"term": {"metadata.is_api": True}}]

        opensearch_knn_response = aos_client.search(index_name=aos_ug_index, query_type="knn",
                                                    query_term=zh_query_relevance_embedding, field="embedding", filter=filter, size=ug_result_num)
        opensearch_knn_results.extend(organize_ug_results(opensearch_knn_response, aos_ug_index)[:ug_result_num])
        opensearch_knn_response = aos_client.search(index_name=aos_ug_index, query_type="knn",
                                                    query_term=en_query_relevance_embedding, field="embedding", filter=filter, size=ug_result_num)
        opensearch_knn_results.extend(organize_ug_results(opensearch_knn_response, aos_ug_index)[:ug_result_num])

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
        # recall_knowledge.sort(key=lambda x: x["score"], reverse=True)
        # debug_info["knowledge_qa_combined_recall"] = recall_knowledge
        
        # 5. Predict correlation score using cross model
        # recall_knowledge_cross = []
        # for knowledge in recall_knowledge:
        #     # get score using cross model
        #     score = float(SagemakerEndpointVectorOrCross(prompt=query_knowledge, endpoint_name=cross_model_endpoint, region_name=region, model_type="cross", stop=None, context=knowledge['doc']))
        #     # logger.info(json.dumps({'doc': knowledge['doc'], 'score': score, 'source': knowledge['source']}, ensure_ascii=False))
        #     if score > 0.8:
        #         recall_knowledge_cross.append({'doc': knowledge['doc'], 'score': score, 'source': knowledge['source']})
        rerank_pair = []
        for knowledge in recall_knowledge:
            rerank_pair.append([query_knowledge, knowledge["content"]])
        score_list = json.loads(SagemakerEndpointVectorOrCross(prompt=json.dumps(rerank_pair), endpoint_name=rerank_model_endpoint,
                                                          region_name=region, model_type="rerank", stop=None))
        rerank_knowledge = []
        for knowledge, score in zip(recall_knowledge, score_list):
            # if score > 0:
            knowledge["rerank_score"] = score
            rerank_knowledge.append(knowledge)
        rerank_knowledge.sort(key=lambda x:x["rerank_score"], reverse=True)
        debug_info["knowledge_qa_rerank"] = rerank_knowledge

        # recall_knowledge_cross.sort(key=lambda x: x["score"], reverse=True)
        # debug_info["knowledge_qa_cross_model_sort"] = recall_knowledge_cross[:10]

        # recall_knowledge_str = concat_recall_knowledge(recall_knowledge_cross[:2])
        recall_knowledge_str = concat_recall_knowledge(rerank_knowledge[:2])
        # sources = list(set([item["source"] for item in recall_knowledge_cross[:2]]))
        sources = list(set([item["source"] for item in rerank_knowledge[:2]]))
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
        # answer = SagemakerEndpointVectorOrCross(prompt=query_input,
        #                                         endpoint_name=llm_model_endpoint,
        #                                         region_name=region,
        #                                         model_type="answer",
        #                                         stop=None,
        #                                         history=history,
        #                                         parameters=parameters,
        #                                         context=recall_knowledge_str[:2560])
        generate_input = dict(
            model_id = llm_model_id,
            query = query_input,
            contexts = rerank_knowledge,
            history=history,
            region_name=region,
            parameters=parameters,
            context_num = 2,
            model_type="answer",
            llm_model_endpoint=llm_model_endpoint
        )
        ret = llm_generate(**generate_input)
        answer = ret['answer']
        debug_info["knowledge_qa_llm"] = ret
    except Exception as e:
        logger.info(f'{traceback.format_exc()}')
        answer = ""
    return answer, query_type, sources, recall_knowledge_str, debug_info

def main_entry(session_id:str, query_input:str, history:list, zh_embedding_model_endpoint:str, en_embedding_model_endpoint:str,
               cross_model_endpoint:str, rerank_model_endpoint:str, llm_model_endpoint:str,
               aos_faq_index:str, aos_ug_index:str, enable_knowledge_qa:bool, temperature: float, enable_q_q_match:bool,
               llm_model_id=None
               ):
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
               zh_embedding_model_endpoint,
               en_embedding_model_endpoint,
               cross_model_endpoint,
               rerank_model_endpoint,
               llm_model_endpoint,
               aos_faq_index,
               aos_ug_index,
               enable_knowledge_qa,
               temperature,
               enable_q_q_match,
               llm_model_id=llm_model_id
               )
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
    llm_model_id = event_body.get('llm_model_id',None)
    # aos_faq_index = event_body['aos_faq_index']
    # aos_ug_index = event_body['aos_ug_index']
    messages = event_body['messages']
    temperature = event_body['temperature']
    if "enable_q_q_match" in event_body:
        enable_q_q_match = event_body['enable_q_q_match']
    else:
        enable_q_q_match = False
    if "enable_debug" in event_body:
        enable_debug = event_body['enable_debug']
    else:
        enable_debug = False

    history, question = process_input_messages(messages)
    role = "user"
    session_id = f"{role}_{int(request_timestamp)}"
    knowledge_qa_flag = True if model == 'knowledge_qa' else False
    
    main_entry_start = time.time() 
    answer, sources, debug_info = main_entry(session_id, question, history, zh_embedding_endpoint, en_embedding_endpoint,
                                             cross_endpoint, rerank_endpoint, llm_endpoint,
                                             aos_faq_index, aos_ug_index, knowledge_qa_flag,
                                             temperature, enable_q_q_match,
                                             llm_model_id=llm_model_id
                                             )
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
    if enable_debug:
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(llmbot_response),
            'debug_info': json.dumps(debug_info)
        }
    else:
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(llmbot_response),
        }