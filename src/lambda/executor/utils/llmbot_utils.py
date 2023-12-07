from enum import Enum

QA_SEP = "=>"
AWS_Free_Chat_Prompt = """你是云服务AWS的智能客服机器人{B}，能够回答{A}的各种问题以及陪{A}聊天，如:{chat_history}\n\n{A}: {question}\n{B}: """
AWS_Knowledge_QA_Prompt = """你是云服务AWS的智能客服机器人{B}，请严格根据反括号中的资料提取相关信息\n```\n{fewshot}\n```\n回答{A}的各种问题，比如:\n\n{A}: {question}\n{B}: """
Fewshot_prefix_Q="问题"
Fewshot_prefix_A="回答"

class QueryType(Enum):
    KeywordQuery   = "KeywordQuery"       #用户仅仅输入了一些关键词（2 token)
    KnowledgeQuery = "KnowledgeQuery"     #用户输入的需要参考知识库有关来回答
    Conversation   = "Conversation"       #用户输入的是跟知识库无关的问题

def combine_recalls(opensearch_knn_respose, opensearch_query_response):
    '''
    filter knn_result if the result don't appear in filter_inverted_result
    '''
    knn_threshold = 0.2
    inverted_theshold = 5.0
    filter_knn_result = { item["content"] : [item['source'], item["score"], item["doc"]] for item in opensearch_knn_respose if item["score"]> knn_threshold }
    filter_inverted_result = { item["content"] : [item['source'], item["score"], item["doc"]] for item in opensearch_query_response if item["score"]> inverted_theshold }

    combine_result = []
    for content, doc_info in filter_knn_result.items():
        # if doc in filter_inverted_result.keys():
        combine_result.append({ "content": content, "doc" : doc_info[2], "score" : doc_info[1], "source" : doc_info[0] })
    return combine_result

def concat_recall_knowledge(recall_knowledge_list):
    """
    Concat recall knowledge result from OpenSearch into a single string.
    """
    return "\n".join([item["doc"] for item in recall_knowledge_list])

def process_input_messages(messages):
    # 1. If two messages are from the same role, we need to merge them into one message.
    # Make sure new_messages follows this order: [user, assistant, user, assistant, ...]
    new_messages = []
    previous_role = None
    for message in messages:
        if message['role'] not in ("user", "assistant"):
            continue

        if len(new_messages) == 0:
            if message['role'] == 'user':
                new_messages.append(message['content'])
        else:
            if message['role'] == previous_role:
                new_messages[-1] += f"\n{message['content']}"
            else:
                new_messages.append(message['content'])
        previous_role = message['role']

    # 2. Generate history and question
    if len(new_messages) % 2 == 0:
        print("The number of messages is even, which is not expected.")
        history = [[new_messages[i], new_messages[i+1]] for i in range(0, len(new_messages)-2, 2)]
        question = new_messages[-2]
    else:
        history = [[new_messages[i], new_messages[i+1]] for i in range(0, len(new_messages)-1, 2)]
        question = new_messages[-1]

    return history, question

def build_conversation_prompt(post_text, conversations, role_a, role_b):
    """
    Build conversation prompt for LLM.
    In current version, we concatenate all conversation history into a single prompt.

    :param post_text: user post text
    :param conversations: conversation history
    :param role_a: role name, e.g. "用户"
    :param role_b: role name, e.g. "AWSBot"
    :return: conversation prompt string
    """
    chat_history = [f"{role_a}: {item[0]}\n{role_b}: {item[1]}" for item in conversations]
    chat_histories = "\n\n".join(chat_history)
    chat_histories = f"\n\n{chat_histories}" if chat_histories else ""

    conversation_prompt = AWS_Free_Chat_Prompt.format(chat_history=chat_histories, question=post_text, A=role_a, B=role_b)
    
    return conversation_prompt

def build_knowledge_qa_prompt(post_text, qa_recalls, role_a, role_b):
    """
    build prompt using qa for LLM. 
    For Knowledge QA, it will merge all retrieved related document paragraphs into a single prompt
    
    :param post_text: user post text
    :param qa_recalls: all retrieved related document paragraphs from OpenSearch
    :param role_a: role name, e.g. "用户"
    :param role_b: role name, e.g. "AWSBot"
    """
    qa_pairs = [ obj["doc"].split(QA_SEP) for obj in qa_recalls ]
    qa_fewshots = [f"{Fewshot_prefix_Q}: {pair[0]}\n{Fewshot_prefix_A}: {pair[1]}" for pair in qa_pairs]
    fewshots_str = "\n\n".join(qa_fewshots[-3:])

    knowledge_qa_prompt = AWS_Knowledge_QA_Prompt.format(fewshot=fewshots_str, question=post_text, A=role_a, B=role_b)
    return knowledge_qa_prompt

def build_final_prompt(query_input, session_history, exactly_match_result, recall_knowledge, role_a, role_b):
    """
    built final prompt for generating answer for user LLM.

    :param query_input: user post text
    :param session_history: conversation history from DynamoDB
    :param exactly_match_result: exactly match result from OpenSearch
    :param recall_knowledge: knowledge recall result from OpenSearch
    :param role_a: role name, e.g. "用户"
    :param role_b: role name, e.g. "AWSBot"

    :return: (answer, final_prompt, query_type)
    """
  
    answer = None
    final_prompt = None
    query_type = None

    if exactly_match_result and recall_knowledge:
        query_type = QueryType.KeywordQuery
        answer = exactly_match_result[0]["doc"]
        final_prompt = ""
    elif recall_knowledge:
        query_type = QueryType.KnowledgeQuery
        final_prompt = build_knowledge_qa_prompt(query_input, recall_knowledge, role_a=role_a, role_b=role_b)
    else:
        query_type = QueryType.Conversation
        free_chat_coversions = [item for item in session_history if item[2] == QueryType.Conversation]
        final_prompt = build_conversation_prompt(query_input, free_chat_coversions[-2:], role_a=role_a, role_b=role_b)

    return (answer, final_prompt, query_type)