import json
import sys
import csv
import os 
import time 
from dotenv import load_dotenv
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__),'.env_global')
)

import logging
log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# sys.path.append("llm-bot/source/lambda/executor/utils")
sys.path.append("../executor")
# sys.path.append("utils")
# sys.path.append(".")
# import aos_utils
# from requests_aws4auth import AWS4Auth
# import boto3
# region = "us-east-1"
# credentials = boto3.Session().get_credentials()
# aos_utils.awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)

# import os
# region = os.environ["AWS_REGION"]
# print(region)
import main
import os

class DummyWebSocket:
    def post_to_connection(self,ConnectionId,Data):
        data = json.loads(Data)
        ret = data
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
            print('knowledge_sources num',ret['choices'][0]['knowledge_sources'])
            if ret['choices'][0].get('contexts'):
                print('contexts num',len(ret['choices'][0].get('contexts')))
                print('contexts avg len: ', sum(len(i) for i in ret['choices'][0]['contexts'])/len(ret['choices'][0]['contexts']))
                # print('sources: ',ret['choices'][0]['contexts'])

main.ws_client = DummyWebSocket()

def generate_answer(query,
                    # temperature=0.7,
                    enable_debug=True,
                    retrieval_only=False,
                    type="market_chain",
                    model="knowledge_qa",
                    stream=False,
                    retriever_index="test-index",
                    session_id=None,
                    rag_parameters=None
                    ):
    rag_parameters = rag_parameters or {}
    body = {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            # "temperature": temperature,
            # "enable_debug": enable_debug,
            # "retrieval_only": retrieval_only,
            # "retriever_index": retriever_index,
            "type": type,
            "model": model,
            "session_id":session_id,
            "enable_debug":True,
            }
    body.update(rag_parameters)
    event = {
        "body": json.dumps(body)
    }
    if stream:
        event["requestContext"] = {
            "eventType":"MESSAGE",
            "connectionId":f'test_{int(time.time())}'
        }

    context = None
    response = main.lambda_handler(event, context)
    if response is None:
        return
    if not stream:
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
    debug_log = open("debug.log", "w")
    with open('test/techbot-qa-test-3.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["URL"] == "repost-qa-csdc/20230915" or row["URL"].startswith("https://repost"):
                continue
            query = row["TechBot Question"]
            docs, debug_info = retrieval(query)
            source_list = []
            for doc in docs[:top_k]:
                source_list.append(doc["metadata"]["source"].lower())
            # gt_answer = row['Answer'].replace('\n', ' ')
            correct_url = row['URL'].split('#')[0].lower()
            correct_url_2 = correct_url.replace("zh_cn/", "")
            correct_url_3 = correct_url.replace("userguide/", "windowsguide/")
            if correct_url not in source_list and correct_url_2 not in source_list and correct_url_3 not in source_list:
                logger.info(f"ERROR QUERY:{query} URL: {source_list} CORRECT URL: {correct_url}")
                error_log.write(f"{query}\t{source_list}\t{correct_url}\n")
            else:
                logger.info(f"CORRECT QUERY:{query} URL: {source_list} CORRECT URL: {correct_url}")
            debug_log.write(f"{query}\n{json.dumps(debug_info, indent=4, ensure_ascii=False)}\n")
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


def multiturn_chat_test():
    session_id = f'test_{int(time.time())}'
    generate_answer(
        "《七里香》的演唱者是谁？",
        model='chat',
        stream=True,
        session_id=session_id
        )
    generate_answer(
        "他还有其他什么歌曲",
        model='chat',
        stream=True,
        session_id=session_id
        )
    
    generate_answer(
        "请总结前面的对话。",
        model='chat',
        stream=True,
        session_id=session_id
        )
    
def multiturn_strict_qq_test():
    session_id = f'test_{int(time.time())}'
    generate_answer(
        "IoT Core是否支持Qos2？", 
        model='strict_q_q',
        stream=True,
        session_id=session_id
        )
    generate_answer(
        "IoT Core是否支持Qos2？", 
        model='strict_q_q',
        stream=True,
        session_id=session_id
        )

def qq_match_test():
    r = generate_answer(
        "IoT Core是否支持Qos2？", 
        # model="auto", 
        model="strict_q_q", 
        stream=True,
        type="market_chain", 
    )

def knowledge_qa_test():
    r = generate_answer(
        "什么是Amazon Bedrock", 
        model="knowledge_qa", 
        stream=True,
        type="market_chain", 
    )
    r = generate_answer(
        "如何将Kinesis Data Streams配置为AWS Lambda的事件源？", 
        model="knowledge_qa", 
        stream=True,
        type="market_chain", 
    )
    # print(r[0])
    r = generate_answer(
        "Amazon EC2 提供了哪些功能来支持不同区域之间的数据恢复?", 
        model="knowledge_qa", 
        stream=False,
        type="market_chain", 
    )
    print(r[0])
    generate_answer(
        "Amazon EC2 提供了哪些功能来支持不同区域之间的数据恢复?", 
        model="knowledge_qa", 
        stream=True,
        type="market_chain", 
    )
    generate_answer(
        "Amazon EC2 提供了哪些功能来支持不同区域之间的数据恢复?", 
        model="knowledge_qa", 
        stream=True,
        type="market_chain", 
    )

def test_baichuan_model():
    session_id=f'test_{time.time()}'
    endpoint_name = 'baichuan2-13b-chat-4bits-2024-02-01-03-58-29-048'
    generate_answer(
        "《夜曲》是谁演唱的？", 
        session_id=session_id,
        model="chat", 
        type="market_chain", 
        stream=True,
        rag_parameters=dict(
            generator_llm_config={
                    "model_kwargs":{
                        "max_new_tokens": 2000,
                        "temperature": 0.1,
                        "top_p": 0.9
                    },
                    "model_id": "Baichuan2-13B-Chat-4bits",
                    "endpoint_name": endpoint_name,
                    "context_num": 2
        })
    )
    generate_answer(
        "他还有哪些其他歌曲？", 
        session_id=session_id,
        model="chat", 
        type="market_chain", 
        stream=True,
        rag_parameters=dict(
            generator_llm_config={
                    "model_kwargs":{
                        "max_new_tokens": 2000,
                        "temperature": 0.1,
                        "top_p": 0.9
                    },
                    "model_id": "Baichuan2-13B-Chat-4bits",
                    "endpoint_name": endpoint_name,
                    "context_num": 2
        })
    )

    r = generate_answer(
        "解释一下“温故而知新”", 
        model="chat", 
        type="market_chain", 
        stream=False,
        rag_parameters=dict(
            generator_llm_config={
                    "model_kwargs":{
                        "max_new_tokens": 2000,
                        "temperature": 0.1,
                        "top_p": 0.9
                    },
                    "model_id": "Baichuan2-13B-Chat-4bits",
                    "endpoint_name": endpoint_name,
                    "context_num": 2
        })
    )
    print(r[0])

    generate_answer(
        "Amazon EC2 提供了哪些功能来支持不同区域之间的数据恢复?", 
        model="knowledge_qa", 
        type="market_chain", 
        stream=True,
        rag_parameters=dict(
            generator_llm_config={
                    "model_kwargs":{
                        "max_new_tokens": 2000,
                        "temperature": 0.1,
                        "top_p": 0.9
                    },
                    "model_id": "Baichuan2-13B-Chat-4bits",
                    "endpoint_name": endpoint_name,
                    "context_num": 1
        })
    )

def test_internlm_model():
    session_id=f'test_{time.time()}'
    # endpoint_name = 'internlm2-chat-7b-4bits-2024-02-28-07-08-57-839'
    # model_id = "internlm2-chat-7b"

    endpoint_name = 'internlm2-chat-20b-4bits-2024-02-29-05-37-42-885'
    model_id = "internlm2-chat-20b"

    rag_parameters = {
        "get_contexts":True,
         "retriever_config":{
            "retriever_top_k": 1,
                "chunk_num": 2,
                "using_whole_doc": True,
                "reranker_top_k": 10,
                "enable_reranker": True
        },
        "query_process_config":{
            "conversation_query_rewrite_config":{
                "model_id":model_id,
                "endpoint_name":endpoint_name
            },
            "translate_config":{
                "model_id":model_id,
                "endpoint_name": endpoint_name
            },
            "stepback_config":{
                "model_id":model_id,
                "endpoint_name": endpoint_name
            }
        },
        "intent_config": {
            "model_id": model_id,
            "endpoint_name": endpoint_name
        },
        "generator_llm_config":{
            "model_kwargs":{
                "max_new_tokens": 2000,
                "temperature": 0.1,
                "top_p": 0.9,
                # 'repetition_penalty':1.1
            },
            "model_id": model_id,
            "endpoint_name": endpoint_name,
            "context_num": 1
        }
    }
    
    qq_match_test()
    generate_answer(
        "AWS支持上海region吗？", 
        model="auto", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )
    # print(sfg)
    generate_answer(
        "介绍一下Amazon EC2", 
        model="auto", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )
    # print(xfg)
    generate_answer(
        "什么是Amazon bedrock？", 
        model="auto", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )

    generate_answer(
        "《夜曲》是谁演唱的？", 
        session_id=session_id,
        model="chat", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )
    generate_answer(
        "他还有哪些其他歌曲？", 
        session_id=session_id,
        model="chat", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )

    r = generate_answer(
        "解释一下“温故而知新”", 
        model="auto", 
        type="market_chain", 
        stream=False,
        rag_parameters=rag_parameters
    )
    print(r[0])


def test_internlm_model_mkt():
    session_id=f'test_{time.time()}'
    # endpoint_name = 'internlm2-chat-7b-4bits-2024-02-28-07-08-57-839'
    # model_id = "internlm2-chat-7b"
    endpoint_name = 'internlm2-chat-20b-4bits-2024-03-04-06-32-53-653'
    model_id = "internlm2-chat-20b"

    os.environ['llm_model_id'] = model_id
    os.environ['llm_model_endpoint_name'] = endpoint_name

    rag_parameters = {
        "get_contexts":True,
    }
    
    qq_match_test()
    generate_answer(
        "AWS支持上海region吗？", 
        model="auto", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )
    # print(sfg)
    generate_answer(
        "介绍一下Amazon EC2", 
        model="auto", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )
    # print(xfg)
    generate_answer(
        "什么是Amazon bedrock？", 
        model="auto", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )

    generate_answer(
        "《夜曲》是谁演唱的？", 
        session_id=session_id,
        model="chat", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )
    generate_answer(
        "他还有哪些其他歌曲？", 
        session_id=session_id,
        model="chat", 
        type="market_chain", 
        stream=True,
        rag_parameters=rag_parameters
    )

    r = generate_answer(
        "解释一下“温故而知新”", 
        model="auto", 
        type="market_chain", 
        stream=False,
        rag_parameters=rag_parameters
    )
    print(r[0])



def test_internlm_model_mkt_knowledge_entry_qq_match():
    session_id=f'test_{time.time()}'
    # endpoint_name = 'internlm2-chat-7b-4bits-2024-02-28-07-08-57-839'
    # model_id = "internlm2-chat-7b"
    endpoint_name = 'internlm2-chat-20b-4bits-2024-03-04-06-32-53-653'
    model_id = "internlm2-chat-20b"
    entry_type = "market_chain_knowledge"

    os.environ['llm_model_id'] = model_id
    os.environ['llm_model_endpoint_name'] = endpoint_name
    # workspace_ids = ["aos_index_mkt_faq_qq_m3", "aos_index_acts_qd_m3", "aos_index_mkt_faq_qd_m3"]
    
    questions = [
        "能否通过JDBC连接到RDS for PostgreSQL？ 有相关的指导吗？",
        "如何解决切换RI后网速变慢？",
        "如何升级EC2配置不改变IP",
        "如何/怎么关停账号",
        "请问怎么关闭账号？",
        "个人能否注册账号？",
        "怎么开具发票？",
        "怎么开发票？",
        "使用CDN服务要备案吗？"
    ]
    for question in questions:
        generate_answer(
           question, 
            model="knowledge_qa", 
            type=entry_type, 
            stream=True,
            rag_parameters={
            "get_contexts":True,
            "retriever_config":{
                # "qq_config": {
                #     "qq_match_threshold": 0.8,
                # },
                # "qd_config":{
                #     "qd_match_threshold": 2,
                # # "using_whole_doc": True
                # },
                # "workspace_ids": workspace_ids
            }
            }
        )


def test_internlm_model_mkt_knowledge_entry():
    session_id=f'test_{time.time()}'
    # endpoint_name = 'internlm2-chat-7b-4bits-2024-02-28-07-08-57-839'
    # model_id = "internlm2-chat-7b"
    endpoint_name = 'internlm2-chat-20b-4bits-2024-03-04-06-32-53-653'
    model_id = "internlm2-chat-20b"
    entry_type = "market_chain"

    os.environ['llm_model_id'] = model_id
    os.environ['llm_model_endpoint_name'] = endpoint_name
    # workspace_ids = ["aos_index_mkt_faq_qq","aos_index_acts_qd"]
    # workspace_ids = ["aos_index_mkt_faq_qq_m3", "aos_index_acts_qd_m3", "aos_index_mkt_faq_qd_m3"]
    
    rag_parameters={
        "get_contexts":True,
        # "session_id":session_id,
        "retriever_config":{}
    }

    # generate_answer(
    #     "请问亚马逊云科技中国区域的S3和EC2的价格",
    #     # "2024亚马逊云科技出海全球化论坛什么时候举办？",
    #     # "lambda是什么？",
    #     # "2024年出海全球化论坛的会议日程是什么？",
    #     # "亚马逊云科技有上海区域吗？",
    #     # "在亚马逊云科技网站上，完成所有账户注册步骤后，什么时候才可以开始使用？",
    #         # "亚马逊云科技中国区域免费套餐有哪几种不同类型的优惠？",
    #         # "什么是日志通",
    #         # "日志通是什么？",
    #         # model="knowledge_qa", 
    #         type=entry_type, 
    #         stream=True,
    #         rag_parameters=rag_parameters
    # )

    # print(sfg)

    # generate_answer(
    #     "2024亚马逊云科技出海全球化论坛什么时候举办？",
    #     # "lambda是什么？",
    #     # "2024年出海全球化论坛的会议日程是什么？",
    #     # "亚马逊云科技有上海区域吗？",
    #     # "在亚马逊云科技网站上，完成所有账户注册步骤后，什么时候才可以开始使用？",
    #         # "亚马逊云科技中国区域免费套餐有哪几种不同类型的优惠？",
    #         # "什么是日志通",
    #         # "日志通是什么？",
    #         # model="knowledge_qa", 
    #         type=entry_type, 
    #         stream=True,
    #         rag_parameters=rag_parameters
    # )
    # for _ in range(10):
    #     generate_answer(
    #         # "2024亚马逊云科技出海全球化论坛什么时候举办？",
    #         # "lambda是什么？",
    #         # "2024年出海全球化论坛的会议日程是什么？",
    #         "亚马逊云科技有上海区域吗？",
    #         # "在亚马逊云科技网站上，完成所有账户注册步骤后，什么时候才可以开始使用？",
    #             # "亚马逊云科技中国区域免费套餐有哪几种不同类型的优惠？",
    #             # "什么是日志通",
    #             # "日志通是什么？",
    #             # model="knowledge_qa", 
    #             type=entry_type, 
    #             stream=True,
    #             rag_parameters=rag_parameters
    #     )


    # print(sfgdfhb)

    # r = generate_answer(
    #         # "lambda",
    #         "什么是日志通",
    #         # model="knowledge_qa", 
    #         type=entry_type, 
    #         stream=False,
    #         rag_parameters=rag_parameters
    # )
    # print(r[0])

    # pritn(sfg)


    # generate_answer(
    #         "火山引擎和阿里云比较",
    #         # "什么是日志通",
    #         # model="knowledge_qa", 
    #         type=entry_type, 
    #         stream=True,
    #         rag_parameters=rag_parameters
    # )

    # print(fbdfb)

    # generate_answer(
    #         "如何快速搭建一个网站？",
    #         # "什么是日志通",
    #         # model="knowledge_qa", 
    #         type=entry_type, 
    #         stream=True,
    #         rag_parameters=rag_parameters
    # )
    # generate_answer(
    #         "客户联系时间",
    #         # "什么是日志通",
    #         # model="knowledge_qa", 
    #         type=entry_type, 
    #         stream=True,
    #         rag_parameters=rag_parameters
    # )
    # generate_answer(
    #         "客户工作时间",
    #         # "什么是日志通",
    #         # model="knowledge_qa", 
    #         type=entry_type, 
    #         stream=True,
    #         rag_parameters=rag_parameters
    # )
 
    # print(sgf)

    market_test_cases = [
    'EC2',
    "LAMBDA",
    '亚马逊云科技中国区域免费套餐有哪几种不同类型的优惠？',
    'Amazon Lambda的免费套餐包含什么？',
    '在亚马逊云科技网站上，完成所有账户注册步骤后，什么时候才可以开始使用？',
    'Amazon Lambda函数是什么？',
    '日志通是什么？',
    'lambda是什么？',
    '2024北京国际车展上，亚马逊云科技会参加吗？',
    '3月份在深圳有生成式AI的活动吗？',
    '2024年会举办出海全球化论坛吗？',
    '2024年出海全球化论坛的会议日程是什么？',
    '2024亚马逊云科技出海全球化论坛什么时候举办？',
    '请问怎么关闭账号？',
    '个人能否注册账号？',
    '怎么开发票？',
    '使用CDN服务要备案吗？',
    '今天是几月几号？',
    '亚马逊云科技有上海区域吗？',
    '我上一个问题是什么？',
    '如何注册AWS账号?',
    '如何注册亚马逊云科技账号',
    '怎么申请免费试用？',
    '怎么试用服务器？',
    '无法连接服务器',
    '连接不上服务器',
    '账号被停用了怎么解决',
    '备案流程',
    '怎么备案',
    '人工服务',
    '为什么产生了费用？不是免费试用吗？',
    '申请退款',
    '服务器报价/服务器多少钱？'
    ]
    ret = []
    for question in market_test_cases:
        r = generate_answer(
            question,
            # "什么是日志通",
            # model="knowledge_qa", 
            type=entry_type, 
            stream=False,
            rag_parameters=rag_parameters
        )
        ret.append((question,r[0]))

    for q,a in ret:
        print('*-'*50)
        print(f'Q: {q},\nA: {a}')


def test_internlm_model_mkt_knowledge_entry_langgraph():
    session_id=f'test_{time.time()}'
    # endpoint_name = 'internlm2-chat-7b-4bits-2024-02-28-07-08-57-839'
    # model_id = "internlm2-chat-7b"
    endpoint_name = 'internlm2-chat-20b-4bits-2024-03-04-06-32-53-653'
    model_id = "internlm2-chat-20b"
    entry_type = "market_chain_knowledge_langgraph"

    os.environ['llm_model_id'] = model_id
    os.environ['llm_model_endpoint_name'] = endpoint_name
    generate_answer(
        "今天是几月几号？", 
        model="knowledge_qa", 
        type=entry_type, 
        stream=True,
        rag_parameters={
            "session_id":session_id,
            "get_contexts":True,
            "retriever_config":{
                "qq_config": {
                    "q_q_match_threshold": 0.9,
                },
                "qd_config":{
                    "qd_match_threshold": 2,
                    "using_whole_doc": True
                },
                "workspace_ids": ["aos_index_mkt_faq_qq","aos_index_acts_qd"]
        }
        }
    )

    generate_answer(
        "日志通是什么？", 
        model="knowledge_qa", 
        type=entry_type, 
        stream=True,
        rag_parameters={
            "session_id":session_id,
            "get_contexts":True,
            "retriever_config":{
                "qq_config": {
                    "q_q_match_threshold": 0.9,
                },
                "qd_config":{
                    "qd_match_threshold": 2,
                    "using_whole_doc": True
                },
                "workspace_ids": ["aos_index_mkt_faq_qq","aos_index_acts_qd"]
        }
        }
    )

    generate_answer(
        "AWS支持上海region吗？", 
        model="knowledge_qa", 
        type=entry_type, 
        stream=True,
        rag_parameters={
            "session_id":session_id,
            "get_contexts":True,
            "retriever_config":{
                "qq_config": {
                    "q_q_match_threshold": 0.9,
                },
                "qd_config":{
                    "qd_match_threshold": 2,
                    "using_whole_doc": True
                },
                "workspace_ids": ["aos_index_mkt_faq_qq","aos_index_acts_qd"]
        }
        }
    )


def market_summary_test():
    session_id = f'test_{int(time.time())}'
    generate_answer(
        "lambda 减少冷启动的方法",
        model='knowledge_qa',
        stream=True,
        session_id=session_id
        )
    generate_answer(
        "请介绍一下bedrock",
        model='knowledge_qa',
        stream=True,
        session_id=session_id
        )
    
    generate_answer(
        "今天几号",
        model='chat',
        stream=True,
        session_id=session_id
        )
 
    generate_answer(
        "",
        model='chat',
        stream=True,
        session_id=session_id,
        type='market_conversation_summary'
        )

def market_summary_test2():
    messages = [
            {
                "role": "user",
                "content": "lambda 减少冷启动的方法"
            },
            {
                 "role": "ai",
                 "content": " 好的,这里有几个减少AWS Lambda函数冷启动时间的建议:\n\n1. 优化代码和依赖项。只加载实际需要的代码和库,避免不必要的初始化。使用较新的运行时版本,它们启动更快。\n\n2. 增加函数的内存。更多的内存意味着更多的CPU被分配,这可以加速初始化过程。\n\n3. 启用预置并发和预热。这将确保总是有热启动容器可用。\n\n4. 将不经常更改的初始化逻辑移到层中。层在冷启动之间重用,可以跳过一些初始化工作。\n\n5. 使用容器重用。这将重用最近使用的容器而不是完全冷启动。\n\n6. 跟踪和分析冷启动时间以查找瓶颈。使用工具如 AWS X-Ray 来识别需要优化的地方。\n\n7. 减少函数的复杂性和依赖关系。简单快速的函数冷启动更快。\n\n8. 将不经常更改的资源预置在 /tmp 中。该目录在启动之间保存文件。\n\n9. 使用异步调用模式避免同步冷启动。\n\n希望这些提示可以帮助你加速 Lambda 函数的初始化!让我知道如果你有任何其他问题。"
            },
            {
                "role":"user",
                "content":"请介绍一下bedrock"
            },
            {
                "role":"ai",
                "content":" Amazon Bedrock是一个机器学习服务,可以帮助您构建和部署基于大语言模型的对话式人工智能应用程序。\n\nBedrock的主要特点和功能包括:\n\n1. 预训练模型库 - Bedrock提供了多个预训练好的对话模型,如Claude等,开箱即用。\n\n2. 自定义模型微调 - 您可以使用自己的数据集对这些预训练模型进行微调,使其适应您的特定应用和对话域。\n\n3. 知识库支持 - 您可以将Bedrock与知识库集成,为对话应用提供背景知识和上下文感知能力。\n\n4. 多种对话通道 - Bedrock支持通过语音、文本或多模态方式进行对话交互。\n\n5. 自动评估和日志记录 - Bedrock会自动评估对话质量,并记录会话日志以进行分析。\n\n6. 简单易用的API - Bedrock提供了简单的API来部署和管理对话模型,无需机器学习专业知识。\n\n7. 完全托管的云服务 - 作为AWS托管服务,Bedrock使您无需管理任何基础设施。\n\n总的来说,Bedrock通过其预训练模型、自定义微调和知识库支持等功能,可以显著降低构建对话AI系统的门槛,加速部署。它使任何规模的公司都可以利用大语言模型的力量来创建人工智能助手和其他对话应用。"
            },
            {
                "role":"user",
                "content": "今天几号"
            },
            {
                "role":"ai",
                "content":" 抱歉,我没有访问当前日期的方式。作为一个AI助手,我不知道“今天”具体指的是哪一天。我建议您直接问我您想知道的具体日期,例如“2022年2月14日是星期几”。或者您也可以询问能够访问当前日期的人这个问题。请让我知道还有什么可以帮助您的!"
            }
        ]
     
    # endpoint_name = 'internlm2-chat-7b-4bits-2024-02-28-07-08-57-839'
    # model_id = "internlm2-chat-7b"

    endpoint_name = 'internlm2-chat-20b-4bits-2024-03-04-06-32-53-653'
    model_id = "internlm2-chat-20b"
    # entry_type = "market_chain_knowledge_langgraph"

    os.environ['llm_model_id'] = model_id
    os.environ['llm_model_endpoint_name'] = endpoint_name


#     body = {
#         "client_type": "zhenghm-postman-summary",
#         "session_id": "af50b380-abec-4f1a-a6e7-acdbca22ebc9",
#         "time_window": {
#         "start_time": 1709787023.0,
#         "end_time": 1709787263.0
#         },
#         "type": "market_conversation_summary",
#         "temperature": 0.1
# }
    body = {
    "client_type": "GCRChat",
    "session_id": "961515e9-1d91-402a-8996-aba68f4211c5",
    "enable_debug": True,
    "time_window": {
      "start_time": 1710976467.939,
      "end_time": 1710994497.327
    },
    "type": "market_conversation_summary",
    "temperature": 0.1
}
    # body = {
    #         "messages": messages,
    #         "type": 'market_conversation_summary',
    #         "mkt_conversation_summary_config": {
    #             "model_id": model_id,
    #             "endpoint_name": endpoint_name
    #         }
    #         # "model":"chat"
    #         }
    event = {
        "body": json.dumps(body)
    }
  
    event["requestContext"] = {
        "eventType":"MESSAGE",
        "connectionId":f'test_{int(time.time())}'
    }
    context = None
    main.lambda_handler(event, context)
    # body = json.loads(response["body"])
    # answer = body["choices"][0]["message"]["content"]
    # print(answer)

def code_chat_test():
    session_id = f'test_{int(time.time())}'
    generate_answer(
        "来点复杂的js code", 
        model="chat",  
        stream=True,
        type="market_chain", 
        session_id=session_id
    )
    generate_answer(
        "Lambda冷启动怎么解决", 
        model="knowledge_qa", 
        stream=True,
        type="market_chain", 
        session_id=session_id
    )

            
def market_deploy_test():
    multiturn_strict_qq_test()
    multiturn_chat_test()
    knowledge_qa_test()
    
    generate_answer(
        "今天天气怎么样？", 
        model="auto", 
        stream=True,
        type="market_chain", 
    )
    qq_match_test()

    market_summary_test2()



def dgr_deploy_test():
    os.environ['llm_model_id'] = "anthropic.claude-3-sonnet-20240229-v1:0"
    session_id = f'test_{int(time.time())}'
    questions = [
        "Amazon EC2 提供了哪些功能来支持不同区域之间的数据恢复?",
        "请问Amazon ECS Fargate支持哪些操作系统和CPU架构?",
        "如何将镜像推送到 Amazon Elastic Container Registry？",
        "Amazon S3 静态网站端点支持 HTTPS吗？",
        "如何禁用QuickSight账户中的电子邮件邀请新用户选项？",
        "如何解决Linux实例在升级和重启后未通过实例检查、无法连接的问题？"
    ]
    for question in questions:
        generate_answer(
            question, 
            stream=True,
            type="dgr",
            # session_id=session_id
        )
    

if __name__ == "__main__":
    # market_summary_test()
    # multiturn_chat_test()
    # market_summary_test()
    # code_chat_test()
    # market_summary_test2()
    # market_deploy_test()
    # market_deploy_cn_test()

    # generate_answer(
    #     "Amazon EC2 提供了哪些功能来支持不同区域之间的数据恢复?", 
    #     model="chat", 
    #     stream=True,
    #     type="market_chain", 
    # )
    # market_deploy_test()
    # knowledge_qa_test()
    # r = generate_answer(
    #     "怎么开发票？", 
    #     model="knowledge_qa", 
    #     stream=True,
    #     type="market_chain", 
    # )
    # knowledge_qa_test()

    
    # market_deploy_test()
    # test_baichuan_model()
    # market_summary_test2()
    # test_internlm_model()
    # dgr_deploy_test()
    test_internlm_model_mkt_knowledge_entry()
    # test_internlm_model_mkt_knowledge_entry_qq_match()
    # test_internlm_model_mkt_knowledge_entry_langgraph()
    # test_baichuan_model()
    
    # market_deploy_test()
    # dgr
    # generate_answer(
    #     # "如何将Kinesis Data Streams配置为AWS Lambda的事件源？",
    #     # "Amazon EC2 提供了哪些功能来支持不同区域之间的数据恢复?",
    #     "什么是Amazon bedrock？",
    #     model="knowledge_qa", 
    #     stream=True,
    #     type="market_chain", 
    #     rag_parameters=dict(
    #         get_contexts = True,
    #         retriever_config={
    #             "retriever_top_k": 1,
    #             "chunk_num": 2,
    #             "using_whole_doc": False,
    #             "reranker_top_k": 10,
    #             "enable_reranker": True
    # },
    # )
    # )

    # r = generate_answer("请写一首诗",model='caht')
    # multiturn_chat_test()
    # generate_answer(
    #     "我想调用Amazon Bedrock中的基础模型，应该使用什么API?",
    #     stream=True,
    #     model='auto'
    #     )
    # LLM
    # generate_answer("Amazon EC2 提供了哪些功能来支持不同区域之间的数据恢复?", model="knowledge_qa", stream=False)
    # generate_answer("什么是 CodeDeploy？", model="knowledge_qa", stream=True)
    # Q-Q
    # generate_answer("在相同的EMR Serverless应用程序中，不同的Job可以共享Worker吗？", model="knowledge_qa", stream=True)
    # generate_answer("polly是什么？", model="auto")
    # generate_answer("DynamoDB API\n要使用 Amazon DynamoDB，您的应用程序必须使用一些简单的 API 操作。下面汇总了这些操作（按类别组织）。")
    # generate_answer("polly是什么？")
    # mkt
    # generate_answer("ECS容器中的日志，可以配置输出到S3上吗？")
    # generate_answer("只要我付款就可以收到发票吗")
    # generate_answer("找不到发票怎么办")
    # generate_answer("发票内容有更新应怎么办", model="strict_q_q")
    # generate_answer("发票内容有更新应怎么办", type="common", stream=False)