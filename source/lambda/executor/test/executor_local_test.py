import json
import sys
import csv

import logging
log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

sys.path.append("utils")
sys.path.append(".")
import aos_utils
from requests_aws4auth import AWS4Auth
import boto3
# region = "us-east-1"
# credentials = boto3.Session().get_credentials()
# aos_utils.awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

from dotenv import load_dotenv
load_dotenv()
# import os
# region = os.environ["AWS_REGION"]
# print(region)
import main

class DummyWebSocket:
    def post_to_connection(self,ConnectionId,Data):
        data = json.loads(Data)
        message = data['choices'][0].get('message',None)
        ret = data
        if message is not None:
            message_type = ret['choices'][0]['message_type']
            if message_type == "START":
                pass
            elif message_type == "CHUNK":
                print(ret['choices'][0]['message']['content'],end="",flush=True)
            elif message_type == "END":
                return 
            elif message_type == "ERROR":
                print(ret['choices'][0]['message']['content'])
                return 
            elif message_type == "CONTEXT":
                print('sources: ',ret['choices'][0]['knowledge_sources'])

main.ws_client = DummyWebSocket()

def generate_answer(query, temperature=0.7, enable_q_q_match=False, enable_debug=True, retrieval_only=False):
    event = {
        "requestContext":{
            "eventType":"MESSAGE",
            "connectionId":"123"
        },
        "body": json.dumps(
            {
                "requestContext":{
                    "eventType":"MESSAGE"
                },
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "aos_faq_index": "chatbot-index-9",
                "aos_ug_index": "chatbot-index-1",
                # "model": "knowledge_qa",
                "temperature": temperature,
                "enable_q_q_match": enable_q_q_match,
                "enable_debug": enable_debug,
                "retrieval_only": retrieval_only,
                # "type": "market_chain",
                "type": "common",
                # "model": "chat"
                # "model": "strict_q_q",
                "model": "knowledge_qa"
            }
        )
    }
    context = None
    response = main.lambda_handler(event, context)
    if response is None:
        return
    body = json.loads(response["body"])
    answer = body["choices"][0]["message"]["content"]
    knowledge_sources = body["choices"][0]["message"]["knowledge_sources"]
    debug_info = body["debug_info"]
    return (answer,
            knowledge_sources,
            debug_info)

def retrieval(query, temperature=0.7, enable_q_q_match=False, enable_debug=True, retrieval_only=True):
    event = {
        "body": json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "aos_faq_index": "chatbot-index-9",
                "aos_ug_index": "chatbot-index-1",
                "model": "knowledge_qa",
                "temperature": temperature,
                "enable_q_q_match": enable_q_q_match,
                "enable_debug": enable_debug,
                "retrieval_only": retrieval_only, 
                # "type": "dgr"
            }
        )
    }
    context = None
    response = main.lambda_handler(event, context)
    body = json.loads(response["body"])
    knowledges = body["knowledges"]
    debug_info = body["debug_info"]
    return (knowledges, debug_info)

def retrieval_test(top_k = 20):
    error_log = open("error.log", "w")
    with open('test/techbot-qa-test-3.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["URL"] == "repost-qa-csdc/20230915" or row["URL"].startswith("https://repost"):
                continue
            query = row["TechBot Question"]
            knowledges, debug_info = retrieval(query)
            source_list = []
            for knowledge in knowledges[:top_k]:
                source_list.append(knowledge["source"])
            # gt_answer = row['Answer'].replace('\n', ' ')
            correct_url = row['URL'].split('#')[0]
            correct_url_2 = correct_url.replace("zh_cn/", "")
            if correct_url not in source_list and correct_url_2 not in source_list:
                logger.info(f"QUERY:{query} URL: {source_list} CORRECT URL: {correct_url}")
                error_log.write(f"{query}\t{source_list}\t{correct_url}\n")
    error_log.close()

def eval():
    result_file = open("result.json", "w")
    debug_info_file = open("debug.json", "w")
    result_list = []
    debug_info_list = []
    with open('test/techbot-qa-test-3.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            answer, source, debug_info = generate_answer(row["TechBot Question"])[:3]
            answer = answer.replace('\n', ' ')
            result = {
                "question": row['TechBot Question'],
                "answer": answer,
                "source": source
            }
            if len(row["URL"]) == 32:
                correct_url = "dgr-oncall"
            else:
                correct_url = row['URL'].split('#')[0]
                correct_url_2 = correct_url.replace("zh_cn/", "")
            if correct_url not in source and correct_url_2 not in source:
                logger.info(f"ERROR URL: {source} CORRECT URL: {correct_url}")
            result_list.append(result)
            debug_info_list.append(debug_info)
    json.dump(result_list, result_file, ensure_ascii=False)
    json.dump(debug_info_list, debug_info_file, ensure_ascii=False)

if __name__ == "__main__":
    # dgr
    # generate_answer("Amazon Fraud Detector 中'entityId'和'eventId'的含义与注意事项")
    # generate_answer("我想调用Amazon Bedrock中的基础模型，应该使用什么API?")
    # generate_answer("polly是什么？")
    # mkt
    generate_answer("ECS容器中的日志，可以配置输出到S3上吗？")
    # generate_answer("只要我付款就可以收到发票吗")
    # generate_answer("找不到发票怎么办")
    # generate_answer("发票内容有更新应怎么办")