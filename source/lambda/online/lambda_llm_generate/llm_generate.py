import json  
from common_utils.logger_utils  import get_logger
from lambda_llm_generate.llm_generate_utils.llm_utils import LLMChain
from common_utils.lambda_invoke_utils import chatbot_lambda_call_wrapper
from common_utils.serialization_utils import JSONEncoder


logger = get_logger("llm_generate")


@chatbot_lambda_call_wrapper
def lambda_handler(event_body, context=None):
    logger.info(f'config: {json.dumps(event_body,ensure_ascii=False,indent=2,cls=JSONEncoder)}')
    llm_chain_config = event_body['llm_config']
    llm_chain_inputs = event_body['llm_input']

    chain = LLMChain.get_chain(
        **llm_chain_config
    )
    output = chain.invoke(llm_chain_inputs)
    # response = {"statusCode": 200, "headers": {"Content-Type": "application/json"}}
    # state["answer"] = "finish llm generate test"
    # response["body"] = {"state": state}

    return output
