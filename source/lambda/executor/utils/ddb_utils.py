import json
import boto3
from datetime import date

def get_session(session_id, chat_session_table):

    table_name = chat_session_table
    dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb.Table(table_name)
    operation_result = ""

    response = table.get_item(Key={'session_id': session_id})

    if "Item" in response.keys():
        # print("****** " + response["Item"]["content"])
        operation_result = json.loads(response["Item"]["content"])
    else:
        # print("****** No result")
        operation_result = ""

    return operation_result


# param:    session_id
#           question
#           answer
# return:   success
#           failed
def update_session(session_id, chat_session_table, question, answer, knowledge_sources):

    table_name = chat_session_table
    dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb.Table(table_name)
    operation_result = ""

    response = table.get_item(Key={'session_id': session_id})

    item = {
            "session_id":session_id,
            "question_content":question,
            "question_answer":answer,
            "revised_answer":"",
            "answer_rating":"",
            "knowledge_sources": knowledge_sources,
        }

    # inserting values into table
    response = table.put_item(
        Item=item
    )

    if "ResponseMetadata" in response.keys():
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            operation_result = "success"
        else:
            operation_result = "failed"
    else:
        operation_result = "failed"

    return operation_result

# For Wechat Miniprogram
# param:    session_id
#           user
#           message
#           timestamp
#           isFirstUpdate
# return:   success
#           failed
def update_history(session_id, chat_session_table, user, message, timestamp, isFirstUpdate):

    table_name = chat_session_table
    dynamodb = boto3.resource('dynamodb')

    # table name
    table = dynamodb.Table(table_name)
    operation_result = ""

    response = table.get_item(Key={'session_id': session_id})

    if "Item" in response.keys():
        # print("****** " + response["Item"]["content"])
        chat_history = json.loads(response["Item"]["content"])
    else:
        # print("****** No result")
        chat_history = []

    chat_history.append([user, message, timestamp])
    content = json.dumps(chat_history)
    TodayDate = date.today()

    # inserting values into table
    if isFirstUpdate:
        response = table.put_item(
            Item={
                'session_id': session_id,
                'content': content,
                'sessionCreationDate': TodayDate,
                'lastUpdateDate': TodayDate
            }
        )
    else:
        response = table.put_item(
            Item={
                'session_id': session_id,
                'content': content,
                'lastUpdateDate': TodayDate
            }
        )
    

    if "ResponseMetadata" in response.keys():
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            operation_result = "success"
        else:
            operation_result = "failed"
    else:
        operation_result = "failed"

    return operation_result