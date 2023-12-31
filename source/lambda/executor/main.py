import json
import logging
import os
import sys

import boto3
import sys 
import time
import copy
import traceback


import retriever as retriever
# from llm_utils import CustomLLM
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda


from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.globals import set_verbose
from langchain.llms import OpenAI


logger = logging.getLogger()
handler = logging.StreamHandler()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


from preprocess_utils import run_preprocess
from aos_utils import LLMBotOpenSearchClient
from llmbot_utils import (
    QueryType,
    combine_recalls,
    concat_recall_knowledge,
    process_input_messages,
)
from ddb_utils import get_session, update_session
from sm_utils import SagemakerEndpointVectorOrCross
# from llm_utils import generate as llm_generate
from llm_utils import get_rag_llm_chain
from response_utils import process_response
from constant import Type,IntentType

region = os.environ["AWS_REGION"]
embedding_endpoint = os.environ.get("embedding_endpoint", "")
zh_embedding_endpoint = os.environ.get("zh_embedding_endpoint", "")
en_embedding_endpoint = os.environ.get("en_embedding_endpoint", "")
cross_endpoint = os.environ.get("rerank_endpoint", "")
rerank_endpoint = os.environ.get("rerank_endpoint", "")
aos_endpoint = os.environ.get("aos_endpoint", "")
aos_index = os.environ.get("aos_index", "")
aos_faq_index = os.environ.get("aos_faq_index", "")
aos_ug_index = os.environ.get("aos_ug_index", "")
llm_endpoint = os.environ.get("llm_endpoint", "")
chat_session_table = os.environ.get("chat_session_table", "")
websocket_url = os.environ.get("websocket_url", "")
sm_client = boto3.client("sagemaker-runtime")
aos_client = LLMBotOpenSearchClient(aos_endpoint)
ws_client = None 

# get aos_index_dict

class APIException(Exception):
    def __init__(self, message, code: str = None):
        if code:
            super().__init__("[{}] {}".format(code, message))
        else:
            super().__init__(message)

def load_ws_client():
    global ws_client
    if ws_client is None:
        ws_client = boto3.client(
        "apigatewaymanagementapi", endpoint_url=websocket_url
        )
    return ws_client


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
    opensearch_query_response = aos_client.search(
        index_name=index_name,
        query_type="basic",
        query_term=source,
        field="metadata.source",
    )
    for r in opensearch_query_response["hits"]["hits"]:
        if r["_source"]["metadata"]["field"] == "answer":
            return r["_source"]["content"]
    return ""


def get_faq_content(source, index_name):
    opensearch_query_response = aos_client.search(
        index_name=index_name,
        query_type="basic",
        query_term=source,
        field="metadata.source",
    )
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
            result["source"] = aos_hit["_source"]["metadata"]["source"]
            result["score"] = aos_hit["_score"]
            result["detail"] = aos_hit["_source"]
            result["content"] = aos_hit["_source"]["content"]
            result["answer"] = get_faq_answer(result["source"], index_name)
            result["doc"] = get_faq_content(result["source"], index_name)
        except:
            print("index_error")
            print(aos_hit["_source"])
            continue
        # result.update(aos_hit["_source"])
        results.append(result)
    return results


def get_ug_content(source, index_name):
    opensearch_query_response = aos_client.search(
        index_name=index_name,
        query_type="basic",
        query_term=source,
        field="metadata.source",
        size=100,
    )
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
        result["source"] = aos_hit["_source"]["metadata"]["source"]
        result["score"] = aos_hit["_score"]
        result["detail"] = aos_hit["_source"]
        result["content"] = aos_hit["_source"]["content"]
        result["doc"] = get_ug_content(result["source"], index_name)
        # result.update(aos_hit["_source"])
        results.append(result)
    return results

def organize_results(response, aos_index=None):
    """
    Organize results from aos response

    :param query_type: query type
    :param response: aos response json
    """
    results = []
    aos_hits = response["hits"]["hits"]
    for aos_hit in aos_hits:
        result = {}
        result["source"] = aos_hit['_source']['metadata']['file_path']
        result["score"] = aos_hit["_score"]
        result["detail"] = aos_hit['_source']
        result["content"] = aos_hit['_source']['text']
        result["doc"] = aos_hit['_source']['text']
        results.append(result)
    return results

def remove_redundancy_debug_info(results):
    filtered_results = copy.deepcopy(results)
    for result in filtered_results:
        for field in list(result["detail"].keys()):
            if field.endswith("embedding") or field.startswith("vector"):
                del result["detail"][field]
    return filtered_results


def main_entry(
    session_id: str,
    query_input: str,
    history: list,
    embedding_model_endpoint: str,
    cross_model_endpoint: str,
    llm_model_endpoint: str,
    aos_index: str,
    enable_knowledge_qa: bool,
    temperature: float,
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
    :param aos_index: The index of the AOS engine.
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

    if enable_knowledge_qa:
        # 1. concatenate query_input and history to unified prompt
        query_knowledge = "".join([query_input] + [row[0] for row in history][::-1])
        logger.info(f"1. query knowledge: {query_knowledge}")

        # 2. get AOS knn recall
        knn_retrieval_size = 10
        start = time.time()
        query_embedding = SagemakerEndpointVectorOrCross(
            prompt="为这个句子生成表示以用于检索相关文章：" + query_knowledge,
            endpoint_name=embedding_model_endpoint,
            region_name=region,
            model_type="vector",
            stop=None,
        )
        opensearch_knn_respose = aos_client.search(
            index_name=aos_index, query_type="knn", query_term=query_embedding,
            field="vector_field", size=knn_retrieval_size
        )
        logger.info(json.dumps(opensearch_knn_respose, ensure_ascii=False))
        elpase_time = time.time() - start
        logger.info(f"runing time of opensearch_knn : {elpase_time}s seconds")

        # 3. get AOS invertedIndex recall
        start = time.time()
        opensearch_query_response = aos_client.search(
            index_name=aos_index, query_type="basic", query_term=query_knowledge
        )
        logger.info(json.dumps(opensearch_query_response, ensure_ascii=False))
        elpase_time = time.time() - start
        logger.info(f"runing time of opensearch_query : {elpase_time}s seconds")

        # 4. combine these two opensearch_knn_respose and opensearch_query_response
        opensearch_knn_results = organize_results(opensearch_knn_respose)
        debug_info["knowledge_qa_knn_recall"] = remove_redundancy_debug_info(opensearch_knn_results)
        opensearch_query_results = organize_results(opensearch_query_response)
        debug_info["knowledge_qa_boolean_recall"] = remove_redundancy_debug_info(opensearch_query_results)
        recall_knowledge = combine_recalls(
            opensearch_knn_results, opensearch_query_results
        )
        logger.info(f"4. recall_knowledge: {recall_knowledge}")

        # 5. Predict correlation score using cross model
        recall_knowledge_cross = []
        rerank_score_threshold = 0
        for knowledge in recall_knowledge:
            # get score using cross model
            score = float(
                SagemakerEndpointVectorOrCross(
                    prompt=query_knowledge,
                    endpoint_name=cross_model_endpoint,
                    region_name=region,
                    model_type="cross",
                    stop=None,
                    context=knowledge["doc"],
                )
            )
            logger.info(
                json.dumps(
                    {
                        "doc": knowledge["doc"],
                        "score": score,
                        "source": knowledge["source"],
                    },
                    ensure_ascii=False,
                )
            )
            if score > rerank_score_threshold:
                recall_knowledge_cross.append(
                    {
                        "doc": knowledge["doc"],
                        "score": score,
                        "source": knowledge["source"],
                    }
                )
        debug_info["knowledge_qa_rerank"] = recall_knowledge_cross

        # recall_knowledge_cross.sort(key=lambda x: x["score"], reverse=True)

        final_retrieval_size = 5
        recall_knowledge_str = concat_recall_knowledge(recall_knowledge_cross[:final_retrieval_size])
        logger.info(recall_knowledge_str)
        sources = list(set([item["source"] for item in recall_knowledge_cross[:final_retrieval_size]]))
        query_type = QueryType.KnowledgeQuery
        elpase_time = time.time() - start
        logger.info(f"runing time of recall knowledge : {elpase_time}s seconds")
    else:
        recall_knowledge_str = ""
        query_type = QueryType.Conversation

    # 6. generate answer using question and recall_knowledge
    parameters = {"temperature": temperature}
    try:
        # generate_answer
        answer = SagemakerEndpointVectorOrCross(
            prompt=query_input,
            endpoint_name=llm_model_endpoint,
            region_name=region,
            model_type="answer",
            stop=None,
            history=history,
            parameters=parameters,
            context=recall_knowledge_str,
        )
    except Exception as e:
        logger.info(f"Exceptions: str({e})")
        answer = ""

    debug_info["knowledge_qa_llm"] = {
        "context": recall_knowledge_str,
        "answer": answer 
    }

    # 7. update_session
    # start = time.time()
    # update_session(
    #     session_id=session_id,
    #     chat_session_table=chat_session_table,
    #     question=query_input,
    #     answer=answer,
    #     knowledge_sources=sources,
    # )
    # elpase_time = time.time() - start
    # logger.info(f"runing time of update_session : {elpase_time}s seconds")

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
    }

    json_obj_str = json.dumps(json_obj, ensure_ascii=False)
    logger.info(json_obj_str)

    return answer, sources, debug_info


def parse_query(
    query_input: str,
    history: list,
    zh_embedding_model_endpoint: str,
    en_embedding_model_endpoint: str,
    debug_info: dict,
):
    start = time.time()
    # concatenate query_input and history to unified prompt
    query_knowledge = "".join([query_input] + [row[0] for row in history][::-1])

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
    zh_query_relevance_embedding_prompt = (
        "为这个句子生成表示以用于检索相关文章：" + parsed_query["zh_query"]
    )
    en_query_relevance_embedding_prompt = (
        "Represent this sentence for searching relevant passages: "
        + parsed_query["en_query"]
    )
    parsed_query["zh_query_similarity_embedding"] = SagemakerEndpointVectorOrCross(
        prompt=zh_query_similarity_embedding_prompt,
        endpoint_name=zh_embedding_model_endpoint,
        region_name=region,
        model_type="vector",
        stop=None,
    )
    parsed_query["zh_query_relevance_embedding"] = SagemakerEndpointVectorOrCross(
        prompt=zh_query_relevance_embedding_prompt,
        endpoint_name=zh_embedding_model_endpoint,
        region_name=region,
        model_type="vector",
        stop=None,
    )
    parsed_query["en_query_similarity_embedding"] = SagemakerEndpointVectorOrCross(
        prompt=en_query_similarity_embedding_prompt,
        endpoint_name=en_embedding_model_endpoint,
        region_name=region,
        model_type="vector",
        stop=None,
    )
    parsed_query["en_query_relevance_embedding"] = SagemakerEndpointVectorOrCross(
        prompt=en_query_relevance_embedding_prompt,
        endpoint_name=en_embedding_model_endpoint,
        region_name=region,
        model_type="vector",
        stop=None,
    )
    elpase_time = time.time() - start
    logger.info(f"runing time of parse query: {elpase_time}s seconds")
    return parsed_query


def q_q_match(parsed_query, debug_info):
    start = time.time()
    opensearch_knn_results = []
    opensearch_knn_response = aos_client.search(
        index_name=aos_faq_index,
        query_type="knn",
        query_term=parsed_query["zh_query_similarity_embedding"],
        field="embedding",
        size=2,
    )
    opensearch_knn_results.extend(
        organize_faq_results(opensearch_knn_response, aos_faq_index)
    )
    opensearch_knn_response = aos_client.search(
        index_name=aos_faq_index,
        query_type="knn",
        query_term=parsed_query["en_query_similarity_embedding"],
        field="embedding",
        size=2,
    )
    opensearch_knn_results.extend(
        organize_faq_results(opensearch_knn_response, aos_faq_index)
    )
    # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
    elpase_time = time.time() - start
    logger.info(f"runing time of opensearch_knn : {elpase_time}s seconds")
    answer = None
    sources = None
    if len(opensearch_knn_results) > 0:
        debug_info["q_q_match_info"] = remove_redundancy_debug_info(
            opensearch_knn_results[:3]
        )
        if opensearch_knn_results[0]["score"] >= 0.9:
            source = opensearch_knn_results[0]["source"]
            answer = opensearch_knn_results[0]["answer"]
            sources = [source]
            return answer, sources
    return answer, sources


def get_relevant_documents(
    parsed_query,
    rerank_model_endpoint: str,
    aos_index: str,
    debug_info,
):
    # 1. get AOS knn recall
    result_num = 20
    start = time.time()
    opensearch_knn_results = []
    opensearch_knn_response = aos_client.search(
        index_name=aos_index,
        query_type="knn",
        query_term=parsed_query["zh_query_relevance_embedding"],
        field="vector_field",
        size=result_num,
    )
    opensearch_knn_results.extend(
        organize_results(opensearch_knn_response, aos_index)[:result_num]
    )
    recall_end_time = time.time()
    elpase_time = recall_end_time - start
    logger.info(f"runing time of recall : {elpase_time}s seconds")

    # 2. get AOS invertedIndex recall
    opensearch_query_results = []

    # 3. combine these two opensearch_knn_response and opensearch_query_response
    recall_knowledge = combine_recalls(opensearch_knn_results, opensearch_query_results)

    rerank_pair = []
    rerank_text_length = 1024 * 10
    for knowledge in recall_knowledge:
        # rerank_pair.append([parsed_query["query"], knowledge["content"]][:1024])
        rerank_pair.append(
            [parsed_query["zh_query"], knowledge["content"]][: rerank_text_length]
        )
    zh_score_list = json.loads(
        SagemakerEndpointVectorOrCross(
            prompt=json.dumps(rerank_pair),
            endpoint_name=rerank_model_endpoint,
            region_name=region,
            model_type="rerank",
            stop=None,
        )
    )
    rerank_knowledge = []
    for knowledge, score in zip(recall_knowledge, zh_score_list):
        # if score > 0:
        new_knowledge = knowledge.copy()
        new_knowledge["rerank_score"] = score
        rerank_knowledge.append(new_knowledge)
    debug_info["knowledge_qa_rerank"] = rerank_knowledge

    rerank_end_time = time.time()
    elpase_time = rerank_end_time - recall_end_time
    logger.info(f"runing time of rerank: {elpase_time}s seconds")

    return rerank_knowledge

def get_relevant_documents_dgr(
    parsed_query,
    rerank_model_endpoint: str,
    aos_faq_index: str,
    aos_ug_index: str,
    debug_info,
):
    # 1. get AOS knn recall
    faq_result_num = 2
    ug_result_num = 20
    start = time.time()
    opensearch_knn_results = []
    opensearch_knn_response = aos_client.search(
        index_name=aos_faq_index,
        query_type="knn",
        query_term=parsed_query["zh_query_relevance_embedding"],
        field="embedding",
        size=faq_result_num,
    )
    opensearch_knn_results.extend(
        organize_faq_results(opensearch_knn_response, aos_faq_index)[:faq_result_num]
    )
    opensearch_knn_response = aos_client.search(
        index_name=aos_faq_index,
        query_type="knn",
        query_term=parsed_query["en_query_relevance_embedding"],
        field="embedding",
        size=faq_result_num,
    )
    opensearch_knn_results.extend(
        organize_faq_results(opensearch_knn_response, aos_faq_index)[:faq_result_num]
    )
    # logger.info(json.dumps(opensearch_knn_response, ensure_ascii=False))
    faq_recall_end_time = time.time()
    elpase_time = faq_recall_end_time - start
    logger.info(f"runing time of faq recall : {elpase_time}s seconds")
    filter = None
    if parsed_query["is_api_query"]:
        filter = [{"term": {"metadata.is_api": True}}]

    opensearch_knn_response = aos_client.search(
        index_name=aos_ug_index,
        query_type="knn",
        query_term=parsed_query["zh_query_relevance_embedding"],
        field="embedding",
        filter=filter,
        size=ug_result_num,
    )
    opensearch_knn_results.extend(
        organize_ug_results(opensearch_knn_response, aos_ug_index)[:ug_result_num]
    )
    opensearch_knn_response = aos_client.search(
        index_name=aos_ug_index,
        query_type="knn",
        query_term=parsed_query["en_query_relevance_embedding"],
        field="embedding",
        filter=filter,
        size=ug_result_num,
    )
    opensearch_knn_results.extend(
        organize_ug_results(opensearch_knn_response, aos_ug_index)[:ug_result_num]
    )

    debug_info["knowledge_qa_knn_recall"] = remove_redundancy_debug_info(
        opensearch_knn_results
    )
    ug_recall_end_time = time.time()
    elpase_time = ug_recall_end_time - faq_recall_end_time
    logger.info(f"runing time of ug recall: {elpase_time}s seconds")

    # 2. get AOS invertedIndex recall
    opensearch_query_results = []

    # 3. combine these two opensearch_knn_response and opensearch_query_response
    recall_knowledge = combine_recalls(opensearch_knn_results, opensearch_query_results)

    rerank_pair = []
    for knowledge in recall_knowledge:
        # rerank_pair.append([parsed_query["query"], knowledge["content"]][:1024])
        rerank_pair.append(
            [parsed_query["en_query"], knowledge["content"]][: 1024 * 10]
        )
    en_score_list = json.loads(
        SagemakerEndpointVectorOrCross(
            prompt=json.dumps(rerank_pair),
            endpoint_name=rerank_model_endpoint,
            region_name=region,
            model_type="rerank",
            stop=None,
        )
    )
    rerank_pair = []
    for knowledge in recall_knowledge:
        # rerank_pair.append([parsed_query["query"], knowledge["content"]][:1024])
        rerank_pair.append(
            [parsed_query["zh_query"], knowledge["content"]][: 1024 * 10]
        )
    zh_score_list = json.loads(
        SagemakerEndpointVectorOrCross(
            prompt=json.dumps(rerank_pair),
            endpoint_name=rerank_model_endpoint,
            region_name=region,
            model_type="rerank",
            stop=None,
        )
    )
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
    rerank_knowledge.sort(key=lambda x: x["rerank_score"], reverse=True)
    debug_info["knowledge_qa_rerank"] = rerank_knowledge

    rerank_end_time = time.time()
    elpase_time = rerank_end_time - ug_recall_end_time
    logger.info(f"runing time of rerank: {elpase_time}s seconds")

    return rerank_knowledge


def dgr_entry(
    session_id: str,
    query_input: str,
    history: list,
    zh_embedding_model_endpoint: str,
    en_embedding_model_endpoint: str,
    cross_model_endpoint: str,
    rerank_model_endpoint: str,
    llm_model_endpoint: str,
    aos_faq_index: str,
    aos_ug_index: str,
    enable_knowledge_qa: bool,
    temperature: float,
    enable_q_q_match: bool,
    llm_model_id=None,
    stream=False
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
    :param stream(Bool): Whether to use llm stream decoding output.

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
    sources = []
    answer = ""
    try:
        # 1. parse query
        parsed_query = parse_query(
            query_input,
            history,
            zh_embedding_model_endpoint,
            en_embedding_model_endpoint,
            debug_info,
        )
        # 2. query question match
        if enable_q_q_match:
            answer, sources = q_q_match(parsed_query, debug_info)
            if answer and sources:
                return answer, sources, contexts, debug_info
        # 3. recall and rerank
        knowledges = get_relevant_documents_dgr(
            parsed_query,
            rerank_model_endpoint,
            aos_faq_index,
            aos_ug_index,
            debug_info,
        )
        context_num = 6
        sources = list(set([item["source"] for item in knowledges[:context_num]]))
        contexts = knowledges[:context_num]
        # 4. generate answer using question and recall_knowledge
        parameters = {"temperature": temperature}
        generate_input = dict(
            model_id=llm_model_id,
            query=query_input,
            contexts=knowledges[:context_num],
            history=history,
            region_name=region,
            model_kwargs=parameters,
            context_num=context_num,
            model_type="answer",
            llm_model_endpoint=llm_model_endpoint,
            stream=stream
        )

        llm_start_time = time.time()
        llm_chain = get_rag_llm_chain(
            **generate_input
        )
        llm_chain.invoke()

        answer = llm_generate(**generate_input)
        llm_end_time = time.time()
        elpase_time = llm_end_time - llm_start_time
        logger.info(f"runing time of llm: {elpase_time}s seconds")
        # answer = ret["answer"]
        debug_info["knowledge_qa_llm"] = answer
    except Exception as e:
        logger.info(f"Exception Query: {query_input}")
        logger.info(f"{traceback.format_exc()}")
        answer = ""

    # 5. update_session
    # start = time.time()
    # update_session(session_id=session_id, chat_session_table=chat_session_table, 
    #                question=query_input, answer=answer, knowledge_sources=sources)
    # elpase_time = time.time() - start
    # logger.info(f'runing time of update_session : {elpase_time}s seconds')

    return answer, sources, contexts, debug_info

def market_chain_entry(
    session_id: str,
    query_input: str,
    history: list,
    zh_embedding_model_endpoint: str,
    en_embedding_model_endpoint: str,
    cross_model_endpoint: str,
    rerank_model_endpoint: str,
    llm_model_endpoint: str,
    aos_index: str,
    enable_knowledge_qa: bool,
    temperature: float,
    enable_q_q_match: bool,
    llm_model_id=None,
    stream=False,
    intent_type=IntentType.KNOWLEDGE_QA
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
    :param aos_index: The index of the AOS engine.
    :param enable_knowledge_qa: Whether to enable knowledge QA.
    :param temperature: The temperature of the language model.
    :param stream(Bool): Whether to use llm stream decoding output.

    return: answer(str)
    """
    aos_index_dict = json.loads(os.environ.get("aos_index_dict", '{"aos_index_mkt_qd":"aws-cn-mkt-knowledge","aos_index_mkt_qq":"gcr-mkt-qq","aos_index_dgr_qd":"ug-index-3","aos_index_dgr_qq":"faq-index-2"}'))
    aos_index_mkt_qd = aos_index_dict['aos_index_mkt_qd']
    aos_index_mkt_qq = aos_index_dict['aos_index_mkt_qq']
    aos_index_dgr_qd = aos_index_dict['aos_index_dgr_qd']
    aos_index_dgr_qq = aos_index_dict['aos_index_dgr_qq']

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
    sources = []
    answer = ""

    dgr_q_d_retriever = retriever.QueryDocumentRetriever(aos_index_dgr_qd, "embedding", "content", "source")
    dgr_q_q_retriever = retriever.QueryQuestionRetriever(
        index=aos_index_dgr_qq, vector_field="embedding", source_field="source", size=5)
    mkt_q_d_retriever = retriever.QueryDocumentRetriever(aos_index_mkt_qd, "vector_field", "text", "file_path")
    mkt_q_q_retriever = retriever.QueryQuestionRetriever(
        index=aos_index_mkt_qq, vector_field="vector_field", source_field="file_path", size=5)

    def format_docs(docs, top_k=1):
        # return "\n\n".join(doc.page_content for doc in docs["docs"][:top_k])
        contexts = []
        contexts.extend([{"doc": doc.page_content} for doc in docs["dgr_docs"][:top_k]])
        contexts.extend([{"doc": doc.page_content} for doc in docs["mkt_docs"][:top_k]])
        return contexts

    def format_sources(docs, top_k=2):
        # return [doc.metadata["source"] for doc in docs["docs"][:top_k]]
        sources = []
        sources.extend([{"doc": doc.metadata["source"]} for doc in docs["dgr_docs"][:top_k]])
        sources.extend([{"doc": doc.metadata["source"]} for doc in docs["mkt_docs"][:top_k]])
        return sources 

    def get_qq_result(docs, threshold=0.7):
        if len(docs) > 0 and docs[0]["score"]:
            source = docs[0]["source"]
            answer = docs[0]["answer"]
            sources = [source]
            return answer, sources
        else:
            return None, []

    def get_strict_qq_result(docs, threshold=0.7):
        results = []
        for doc in docs:
            results.append({"score": doc.metadata["score"], 
                            "source": doc.metadata["source"],
                            "answer": doc.metadata["answer"],
                            "question": doc.metadata["question"]})
        output = {"answer": json.dumps(results, ensure_ascii=False), "sources": [], "contexts": []}
        return output

    def output_postprocess(raw_output):
        output = {"answer": "", "sources": [], "contexts": []}
        if raw_output is not None:
            output["answer"] = raw_output.get("answer", "")
            output["sources"] = raw_output.get("sources", [])
            output["contexts"] = raw_output.get("contexts", [])
        return output
 
    def contexts_trunc(contexts:list,context_num=2):
        return [context['doc'] for context in contexts[:context_num]]
 
    contexts_trunc_stage = RunnableLambda(
        lambda x: {"query": x["query"], "contexts": contexts_trunc(x["contexts"], context_num=2)}
        )

    llm_chain = get_rag_llm_chain(
        model_id=llm_model_id, 
        model_kwargs=None,  # TODO 
        stream=stream
        )
    llm_chain = contexts_trunc_stage | llm_chain
    qd_llm_chain = RunnableParallel({
                "dgr_docs": dgr_q_d_retriever,
                "mkt_docs": mkt_q_d_retriever,
                "query": lambda x:x["query"],
                "debug_info": lambda x:x["debug_info"]}
            ) | RunnableParallel({
                "contexts": format_docs,
                "sources": format_sources,
                "query": lambda x:x["query"],
                "debug_info": lambda x:x["debug_info"]}
            ) | RunnableParallel({
                "answer": llm_chain,
                "sources": lambda x:x["sources"],
                "contexts": lambda x:x["contexts"],
                "debug_info": lambda x:x["debug_info"]})
    qq_chain = dgr_q_q_retriever
    def qq_route(info, threshold=0.9):
        for doc in info["docs"]:
            if doc.metadata["score"] > threshold:
                output = {"answer": doc.metadata["answer"], "sources": doc.metadata["source"],
                          "contexts": [], "debug_info": lambda x:x["debug_info"]}
                return output
        return qd_llm_chain

    qq_qd_llm_chain = RunnableParallel({
                "docs": qq_chain,
                "query": lambda x:x["query"],
                "debug_info": lambda x:x["debug_info"]}
            ) | RunnableLambda(qq_route)

    strict_q_q_chain = mkt_q_q_retriever | RunnableLambda(get_strict_qq_result)

    def route(info):
        if info["intent_type"] == IntentType.AUTO.value:
            return qq_qd_llm_chain
        elif info["intent_type"] == IntentType.KNOWLEDGE_QA.value:
            return qq_qd_llm_chain
        elif info["intent_type"] == IntentType.CHAT.value:
            return llm_chain
        elif info["intent_type"] == IntentType.STRICT_QQ.value:
            return strict_q_q_chain
        else:
            return qq_qd_llm_chain

    full_chain = RunnableLambda(route)
    response = full_chain.invoke({"query": query_input, "debug_info": debug_info, "intent_type": intent_type})
    answer = response["answer"]
    sources = response["sources"]
    contexts = response["contexts"]
    return answer, sources, contexts, debug_info

def main_chain_entry(
    session_id: str,
    query_input: str,
    history: list,
    zh_embedding_model_endpoint: str,
    en_embedding_model_endpoint: str,
    cross_model_endpoint: str,
    rerank_model_endpoint: str,
    llm_model_endpoint: str,
    aos_index: str,
    enable_knowledge_qa: bool,
    temperature: float,
    enable_q_q_match: bool,
    llm_model_id=None,
    stream=False,
    intent_type=IntentType.KNOWLEDGE_QA
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
    :param aos_index: The index of the AOS engine.
    :param enable_knowledge_qa: Whether to enable knowledge QA.
    :param temperature: The temperature of the language model.
    :param stream(Bool): Whether to use llm stream decoding output.

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
    sources = []
    answer = ""

    set_verbose(True)
    q_d_retriever = retriever.QueryDocumentRetriever(aos_index, "vector_field", "text", "file_path")

    def format_docs(docs, top_k=6):
        # return "\n\n".join(doc.page_content for doc in docs["docs"][:top_k])
        contexts = []
        contexts.extend([{"doc": doc.page_content} for doc in docs["docs"][:top_k]])
        return contexts

    def format_sources(docs, top_k=1):
        # return [doc.metadata["source"] for doc in docs["docs"][:top_k]]
        sources = []
        sources.extend([{"doc": doc.metadata["source"]} for doc in docs["docs"][:top_k]])
        return sources 

    # llm = CustomLLM(model_id=llm_model_id, stream=stream)

    # llm_chain = RunnableLambda(lambda x:json.dumps(x)) | CustomLLM(model_id="anthropic.claude-v2", stream=stream)
    # llm_generate_for_chain = CustomLLM(model_id=llm_model_id, stream=stream)
    # llm_chain = RunnableLambda(lambda x:json.dumps(x)) | RunnableLambda(llm_generate_for_chain)

    def contexts_trunc(contexts:list,context_num=2):
        return [context['doc'] for context in contexts[:context_num]]
 
    contexts_trunc_stage = RunnableLambda(
        lambda x: {"query": x["query"], "contexts": contexts_trunc(x["contexts"], context_num=2)}
        )
    
    llm_chain = get_rag_llm_chain(
        model_id=llm_model_id, 
        model_kwargs=None,  # TODO 
        stream=stream
        )
    llm_chain = contexts_trunc_stage | llm_chain

    rag_chain = RunnableParallel({
                "docs": q_d_retriever,
                "query": lambda x:x["query"],
                "debug_info": lambda x:x["debug_info"]}
            ) | RunnableParallel({
                "contexts": format_docs,
                "sources": format_sources,
                "query": lambda x:x["query"],
                "debug_info": lambda x:x["debug_info"]}
            ) | RunnableParallel({
                "answer": llm_chain,
                "sources": lambda x:x["sources"],
                "contexts": lambda x:x["contexts"],
                "debug_info": lambda x:x["debug_info"]})
    response = rag_chain.invoke({"query": query_input, "debug_info": debug_info})
    answer = response["answer"]
    sources = response["sources"]
    contexts = response["contexts"]
    return answer, sources, contexts, debug_info

def _is_websocket_request(event):
    """Check if the request is WebSocket or Restful

    Args:
        event: lambda request event
    """
    if (
        "requestContext" in event
        and "eventType" in event["requestContext"]
        and event["requestContext"]["eventType"] == "MESSAGE"
    ):
        return True
    else:
        return False


# @handle_error
def lambda_handler(event, context):
    request_timestamp = time.time()
    logger.info(f"request_timestamp :{request_timestamp}")
    logger.info(f"event:{event}")
    logger.info(f"context:{context}")

    # Get request body
    event_body = json.loads(event["body"])
    model = event_body["model"]
    messages = event_body["messages"]
    temperature = event_body["temperature"]
    stream = _is_websocket_request(event)
    if stream:
        load_ws_client()

    logger.info(f'stream decode: {stream}')
    type = event_body.get("type", Type.COMMON.value)
    enable_q_q_match = event_body.get("enable_q_q_match", False)
    enable_debug = event_body.get("enable_debug", False)

    retrieval_only = event_body.get("enable_debug", False)
    get_contexts = event_body.get("get_contexts", False)
    intent_type = event_body.get("model", IntentType.KNOWLEDGE_QA.value)
    llm_model_id = event_body.get("llm_model_id", "anthropic.claude-v2")
    # stream = event_body.get("stream", False)

    history, question = process_input_messages(messages)
    role = "user"
    
    if stream:
        session_id = event['requestContext']['connectionId']
    else:
        session_id = f"{role}_{int(request_timestamp)}"

    knowledge_qa_flag = True if model == "knowledge_qa" else False

    main_entry_start = time.time()
    contexts = []
    if type.lower() == Type.COMMON.value:
        answer, sources, context, debug_info = main_chain_entry(
            session_id,
            question,
            history,
            zh_embedding_endpoint,
            en_embedding_endpoint,
            cross_endpoint,
            rerank_endpoint,
            llm_endpoint,
            aos_index,
            knowledge_qa_flag,
            temperature,
            enable_q_q_match,
            stream=stream,
            llm_model_id=llm_model_id,
            intent_type=intent_type
        )
    elif type.lower() == Type.DGR.value:
        answer, sources, context, debug_info = dgr_entry(
            session_id,
            question,
            history,
            zh_embedding_endpoint,
            en_embedding_endpoint,
            cross_endpoint,
            rerank_endpoint,
            llm_endpoint,
            aos_faq_index,
            aos_ug_index,
            knowledge_qa_flag,
            temperature,
            enable_q_q_match,
            stream=stream
        )
    elif type.lower() == Type.MARKET_CHAIN.value:
        answer, sources, context, debug_info = market_chain_entry(
            session_id,
            question,
            history,
            zh_embedding_endpoint,
            en_embedding_endpoint,
            cross_endpoint,
            rerank_endpoint,
            llm_endpoint,
            aos_index,
            knowledge_qa_flag,
            temperature,
            enable_q_q_match,
            stream=stream,
            llm_model_id=llm_model_id,
            intent_type=intent_type
        )

    main_entry_elpase = time.time() - main_entry_start
    logger.info(f"runing time of {type} entry : {main_entry_elpase}s seconds")

    return process_response(**dict(
        stream=stream,
        session_id=session_id,
        model=model,
        request_timestamp=request_timestamp,
        answer=answer,
        sources=sources,
        get_contexts=get_contexts,
        contexts=contexts,
        enable_debug=enable_debug,
        debug_info=debug_info,
        ws_client=ws_client
    ))