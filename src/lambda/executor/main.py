import json
import logging
import os
import boto3
import time
import copy
import traceback
from utils.preprocess import run_preprocess  
from utils.aos_utils import LLMBotOpenSearchClient
from utils.llmbot_utils import QueryType, combine_recalls, concat_recall_knowledge, process_input_messages
from utils.ddb_utils import get_session, update_session
from utils.sm_utils import SagemakerEndpointVectorOrCross
from utils.llm import generate as llm_generate

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

def parse_query(query_input:str, history:list,
                zh_embedding_model_endpoint:str, en_embedding_model_endpoint:str,
                debug_info:dict):
    start = time.time()
    # concatenate query_input and history to unified prompt
    query_knowledge = ''.join([query_input] + [row[0] for row in history][::-1])

    # get query embedding
    parsed_query = run_preprocess(query_knowledge) 
    debug_info["query_parser_info"] = parsed_query
    if parsed_query["query_lang"] == "zh":
        parsed_query["zh_query"] = query_knowledge
        parsed_query["en_query"] = parsed_query["translated_text"]
    elif parsed_query["query_lang"] == "en":
        parsed_query["zh_query"] = parsed_query["translated_text"]
        parsed_query["en_query"] = query_knowledge
    zh_query_similarity_embedding_prompt = parsed_query["zh_query"]
    en_query_similarity_embedding_prompt = parsed_query["en_query"]
    zh_query_relevance_embedding_prompt = "为这个句子生成表示以用于检索相关文章：" + parsed_query["zh_query"]
    en_query_relevance_embedding_prompt = "Represent this sentence for searching relevant passages: " + parsed_query["en_query"]
    parsed_query["zh_query_similarity_embedding"] = SagemakerEndpointVectorOrCross(prompt=zh_query_similarity_embedding_prompt,
                                                                endpoint_name=zh_embedding_model_endpoint, region_name=region,
                                                                model_type="vector", stop=None)
    parsed_query["zh_query_relevance_embedding"] = SagemakerEndpointVectorOrCross(prompt=zh_query_relevance_embedding_prompt,
                                                                endpoint_name=zh_embedding_model_endpoint, region_name=region,
                                                                model_type="vector", stop=None)
    parsed_query["en_query_similarity_embedding"] = SagemakerEndpointVectorOrCross(prompt=en_query_similarity_embedding_prompt,
                                                                endpoint_name=en_embedding_model_endpoint, region_name=region,
                                                                model_type="vector", stop=None)
    parsed_query["en_query_relevance_embedding"] = SagemakerEndpointVectorOrCross(prompt=en_query_relevance_embedding_prompt,
                                                                endpoint_name=en_embedding_model_endpoint, region_name=region,
                                                                model_type="vector", stop=None)
    elpase_time = time.time() - start
    logger.info(f'runing time of parse query: {elpase_time}s seconds')
    return parsed_query

def q_q_match(parsed_query, debug_info):
    start = time.time()
    opensearch_knn_results = []
    opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn",
                                                query_term=parsed_query["zh_query_similarity_embedding"], field="embedding", size=2)
    opensearch_knn_results.extend(organize_faq_results(opensearch_knn_response, aos_faq_index))
    opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn",
                                                query_term=parsed_query["en_query_similarity_embedding"], field="embedding", size=2)
    opensearch_knn_results.extend(organize_faq_results(opensearch_knn_response, aos_faq_index))
    # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
    elpase_time = time.time() - start
    logger.info(f'runing time of opensearch_knn : {elpase_time}s seconds')
    answer = None
    sources = None
    if len(opensearch_knn_results) > 0:
        debug_info["q_q_match_info"] = remove_redundancy_debug_info(opensearch_knn_results[:3])
        if opensearch_knn_results[0]["score"] >= 0.9:
            source = opensearch_knn_results[0]["source"]
            answer = opensearch_knn_results[0]["answer"]
            sources = [source]
            return answer, sources
    return answer, sources

def get_relevant_documents(parsed_query, rerank_model_endpoint:str, aos_faq_index:str, aos_ug_index:str, debug_info):
    # 1. get AOS knn recall 
    faq_result_num = 2
    ug_result_num = 20
    start = time.time()
    opensearch_knn_results = []
    opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn",
                                                query_term=parsed_query["zh_query_relevance_embedding"], field="embedding", size=faq_result_num)
    opensearch_knn_results.extend(organize_faq_results(opensearch_knn_response, aos_faq_index)[:faq_result_num])
    opensearch_knn_response = aos_client.search(index_name=aos_faq_index, query_type="knn",
                                                query_term=parsed_query["en_query_relevance_embedding"], field="embedding", size=faq_result_num)
    opensearch_knn_results.extend(organize_faq_results(opensearch_knn_response, aos_faq_index)[:faq_result_num])
    # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
    faq_recall_end_time = time.time()
    elpase_time = faq_recall_end_time  - start
    logger.info(f'runing time of faq recall : {elpase_time}s seconds')
    filter = None
    if parsed_query["is_api_query"]:
        filter = [{"term": {"metadata.is_api": True}}]

    opensearch_knn_response = aos_client.search(index_name=aos_ug_index, query_type="knn",
                                                query_term=parsed_query["zh_query_relevance_embedding"], field="embedding", filter=filter, size=ug_result_num)
    opensearch_knn_results.extend(organize_ug_results(opensearch_knn_response, aos_ug_index)[:ug_result_num])
    opensearch_knn_response = aos_client.search(index_name=aos_ug_index, query_type="knn",
                                                query_term=parsed_query["en_query_relevance_embedding"], field="embedding", filter=filter, size=ug_result_num)
    opensearch_knn_results.extend(organize_ug_results(opensearch_knn_response, aos_ug_index)[:ug_result_num])

    debug_info["knowledge_qa_knn_recall"] = remove_redundancy_debug_info(opensearch_knn_results)
    ug_recall_end_time = time.time()
    elpase_time = ug_recall_end_time  - faq_recall_end_time
    logger.info(f'runing time of ug recall: {elpase_time}s seconds')
    
    # 2. get AOS invertedIndex recall
    opensearch_query_results = []

    # 3. combine these two opensearch_knn_response and opensearch_query_response
    recall_knowledge = combine_recalls(opensearch_knn_results, opensearch_query_results)
    
    rerank_pair = []
    for knowledge in recall_knowledge:
        # rerank_pair.append([parsed_query["query"], knowledge["content"]][:1024])
        rerank_pair.append([parsed_query["en_query"], knowledge["content"]][:1024*10])
    en_score_list = json.loads(SagemakerEndpointVectorOrCross(prompt=json.dumps(rerank_pair), endpoint_name=rerank_model_endpoint,
                                                        region_name=region, model_type="rerank", stop=None))
    rerank_pair = []
    for knowledge in recall_knowledge:
        # rerank_pair.append([parsed_query["query"], knowledge["content"]][:1024])
        rerank_pair.append([parsed_query["zh_query"], knowledge["content"]][:1024*10])
    zh_score_list = json.loads(SagemakerEndpointVectorOrCross(prompt=json.dumps(rerank_pair), endpoint_name=rerank_model_endpoint,
                                                        region_name=region, model_type="rerank", stop=None))
    rerank_knowledge = []
    for knowledge, score in zip(recall_knowledge, zh_score_list):
        # if score > 0:
        new_knowledge = knowledge.copy()
        new_knowledge["rerank_score"] = score
        rerank_knowledge.append(new_knowledge)
    for knowledge, score in zip(recall_knowledge, en_score_list):
        # if score > 0:
        new_knowledge = knowledge.copy()
        new_knowledge["rerank_score"] = score
        rerank_knowledge.append(new_knowledge)
    rerank_knowledge.sort(key=lambda x:x["rerank_score"], reverse=True)
    debug_info["knowledge_qa_rerank"] = rerank_knowledge

    rerank_end_time = time.time()
    elpase_time = rerank_end_time  - ug_recall_end_time
    logger.info(f'runing time of rerank: {elpase_time}s seconds')

    return rerank_knowledge

def main_entry(session_id:str, query_input:str, history:list, zh_embedding_model_endpoint:str, en_embedding_model_endpoint:str,
               rerank_model_endpoint:str, llm_model_endpoint:str, aos_faq_index:str, aos_ug_index:str,
               enable_knowledge_qa:bool, temperature: float, enable_q_q_match:bool,
               llm_model_id=None
               ):
    """
    Entry point for the Lambda function.

    :param session_id: The ID of the session.
    :param query_input: The query input.
    :param history: The history of the conversation.
    :param embedding_model_endpoint: The endpoint of the embedding model.
    :param rerank_model_endpoint: The endpoint of the rerank model.
    :param llm_model_endpoint: The endpoint of the language model.
    :param llm_model_name: The name of the language model.
    :param aos_faq_index: The faq index of the AOS engine.
    :param aos_ug_index: The ug index of the AOS engine.
    :param enable_knowledge_qa: Whether to enable knowledge QA.
    :param temperature: The temperature of the language model.

    return: answer(str)
    """
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
    contexts = []
    if enable_knowledge_qa:
        try:
            # 1. parse query 
            parsed_query = parse_query(query_input, history, zh_embedding_model_endpoint, en_embedding_model_endpoint, debug_info)
            # 2. query question match
            if enable_q_q_match:
                answer, sources = q_q_match(parsed_query, debug_info)
                if answer and sources:
                    return answer, sources, contexts, debug_info
            # 3. recall and rerank
            knowledges = get_relevant_documents(parsed_query, rerank_model_endpoint, aos_faq_index, aos_ug_index, debug_info)
            context_num = 2
            sources = list(set([item["source"] for item in knowledges[:context_num]]))
            contexts = knowledges[:context_num]
            # 4. generate answer using question and recall_knowledge
            parameters = {'temperature': temperature}
            generate_input = dict(
                model_id = llm_model_id,
                query = query_input,
                contexts = knowledges[:context_num],
                history=history,
                region_name=region,
                parameters=parameters,
                context_num = context_num,
                model_type="answer",
                llm_model_endpoint=llm_model_endpoint
            )

            llm_start_time = time.time()
            ret = llm_generate(**generate_input)
            llm_end_time = time.time()
            elpase_time = llm_end_time  - llm_start_time
            logger.info(f'runing time of llm: {elpase_time}s seconds')
            answer = ret['answer']
            debug_info["knowledge_qa_llm"] = ret
        except Exception as e:
            logger.info(f'Exception Query: {query_input}')
            logger.info(f'{traceback.format_exc()}')
            answer = ""
        query_type = QueryType.KnowledgeQuery
    else:
        query_type = QueryType.Conversation

    # 5. update_session
    start = time.time()
    update_session(session_id=session_id, chat_session_table=chat_session_table, 
                   question=query_input, answer=answer, knowledge_sources=sources)
    elpase_time = time.time() - start
    logger.info(f'runing time of update_session : {elpase_time}s seconds')

    # 6. log results
    json_obj = {
        "session_id": session_id,
        "query": query_input,
        "detect_query_type": str(query_type),
        "history": history,
        "chatbot_answer": answer,
        "sources": sources,
        "timestamp": int(time.time()),
        "debug_info": debug_info 
    }
    json_obj_str = json.dumps(json_obj, ensure_ascii=False)
    # logger.info(json_obj_str)
    return answer, sources, contexts, debug_info

def retriever_entry(query_input:str, history:list, zh_embedding_model_endpoint:str, en_embedding_model_endpoint:str,
               rerank_model_endpoint:str, aos_faq_index:str, aos_ug_index:str):
    """
    Entry point for the Lambda function.

    :param session_id: The ID of the session.
    :param query_input: The query input.
    :param history: The history of the conversation.
    :param embedding_model_endpoint: The endpoint of the embedding model.
    :param rerank_model_endpoint: The endpoint of the rerank model.
    :param aos_faq_index: The faq index of the AOS engine.
    :param aos_ug_index: The ug index of the AOS engine.

    return: doc(list)
    """
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
    # 1. parse query 
    parsed_query = parse_query(query_input, history, zh_embedding_model_endpoint, en_embedding_model_endpoint, debug_info)
    # 2. recall and rerank
    knowledges = get_relevant_documents(parsed_query, rerank_model_endpoint, aos_faq_index, aos_ug_index, debug_info)
    return knowledges, debug_info

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
    if "retrieval_only" in event_body:
        retrieval_only = event_body['retrieval_only']
    else:
        retrieval_only = False
    if "get_contexts" in event_body:
        get_contexts = event_body['get_contexts']
    else:
        get_contexts = False

    history, question = process_input_messages(messages)
    role = "user"
    session_id = f"{role}_{int(request_timestamp)}"
    knowledge_qa_flag = True if model == 'knowledge_qa' else False

    response = {'statusCode': 200, 'headers': {'Content-Type': 'application/json'}}
    
    main_entry_start = time.time() 
    if retrieval_only:
        knowledges, debug_info = retriever_entry(question, history,
                                     zh_embedding_endpoint, en_embedding_endpoint, rerank_endpoint, aos_faq_index, aos_ug_index)
        retrieval_response = {
            "knowledges": knowledges
        }
        if enable_debug:
            retrieval_response["debug_info"] = debug_info
        response["body"] = json.dumps(retrieval_response, ensure_ascii=False)
        return response
    answer, sources, contexts, debug_info = main_entry(session_id, question, history, zh_embedding_endpoint, en_embedding_endpoint,
                                             rerank_endpoint, llm_endpoint, aos_faq_index, aos_ug_index, knowledge_qa_flag,
                                             temperature, enable_q_q_match, llm_model_id)
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

    # 2. return result
    if get_contexts:
        llmbot_response["contexts"] = contexts
    if enable_debug:
        llmbot_response["debug_info"] = debug_info
    response["body"] = json.dumps(llmbot_response)
    return response