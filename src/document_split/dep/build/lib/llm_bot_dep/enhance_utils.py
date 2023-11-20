# python shell only support boto3 1.22.5 (1.28.68), according to https://docs.aws.amazon.com/glue/latest/dg/add-job-python.html#python-shell-limitations
import os
import boto3
import json
import logging
import openai
from typing import Dict, List
from langchain.docstore.document import Document
import nltk

# print the log to stdout
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# token number to slice a document
slice_size = 50
# number of questions to generate
question_num = 5

en_prompt_template = """
Here is snippet of {solution}'s manual document within backticks
```
{page}
```
Please generate {question_num} questions and corresponding answers based on these document fragments, with the questions being as diverse as possible and containing details, following the rules below:
1. "{solution}" needs to be included in the Question continuously
2. The question part needs to start with "Question: "
3. The answer part needs to start with "Answer: "
4. All questions and answers need to be in Chinese
"""

zh_prompt_template = """
如下三个反括号中是{solution}的产品文档片段
```
{page}
```
请基于这些文档片段自动生成{question_num}个问题以及对应答案, 问题需要尽可能多样化并包含细节, 且遵循如下规则:
1. "{solution}"需要一直被包含在Question中
2. 问题部分需要以"Question: "开始
3. 答案部分需要以"Answer: "开始
4. 所有问题和答案需要为中文
"""

class EnhanceWithBedrock:
    def __init__(self, prompt: str, solution_title: str, document: Document, zh: bool = True):
        BEDROCK_REGION = str(boto3.session.Session().region_name)
        # TODO, pass such credentials from CloudFormation creation and store in SSM
        openai.api_key = os.getenv("OPENAI_API_KEY")
        session = boto3.Session()
        self.bedrock_client = session.client(
            service_name='bedrock',
            region_name=BEDROCK_REGION,
            endpoint_url='https://bedrock-runtime.{}.amazonaws.com'.format(BEDROCK_REGION)
        )
        self.prompt = prompt
        self.solution_title = solution_title
        self.document = document
        self.zh = zh

    def EnhanceWithClaude(self, prompt: str, solution_title: str, document: Document, zh: bool = True) -> List[Dict[str, str]]:
        """
        Enhance the given prompt using the Claude model by Anthropic. This function constructs a new prompt using the given solution title and page content,
        sends a request to the Claude model, and retrieves the model's response.

        Parameters:
        - prompt (str): The original prompt to be enhanced.
        - solution_title (str): The title of the solution to be included in the new prompt.
        - page_content (str): The content of the page to be included in the new prompt.
        - zh (bool): A flag indicating whether to use the Chinese or English prompt template. Default is True (Chinese).

        Returns:
        - List[Dict[str, str]]: A list of dictionaries, each containing a question and its corresponding answer.

        Example:
        ```python
        prompt = "Do we have any solution offer to Stable Diffusion?"
        solution_title = "Stable Diffusion AWS Extensions"
        page_content = "Stable Diffusion AWS Extensions is a CSDC solution that..."
        enhanced_prompt = EnhanceWithClaude(prompt, solution_title, page_content)
        ```

        Note:
        - Deprecated: Claude v2 does not output Chinese characters in experiment, so Claude v1 is used here.
        """
        # Initialize an empty list to store the Document objects
        # documents = []
        prompt_template = zh_prompt_template if zh else en_prompt_template
        prompt = prompt_template.format(solution=solution_title, page=document.page_content, question_num=question_num)
        prompt = "\n\nHuman:{}".format(prompt) + "\n\nAssistant:"
        # schema keep changing, refer to https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html#model-parameters-claude for latest schema
        body = json.dumps({
            "prompt": prompt,
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 0,
            "max_tokens_to_sample": 500,
            "stop_sequences": ["\n\nHuman:"]
        })
        # note v2 is not output chinese characters
        modelId = "anthropic.claude-v2"
        accept = "*/*"
        contentType = "application/json"

        response = self.bedrock_client.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())
        raw_completion = response_body.get("completion").split('\n')

        # Initialize an empty list to store the Q&A pairs
        qa_list = []

        # Initialize an empty dictionary to store the current Q&A pair
        qa_dict = {}
        for line in raw_completion:
            # Check if the line contains a question
            if line.startswith('Question:'):
                # If there's already a Q&A pair in qa_dict, append it to qa_list
                if qa_dict:
                    qa_list.append(qa_dict)
                    qa_dict = {}  # Reset qa_dict for the next Q&A pair
                qa_dict['Question'] = line.replace('Question:', '').strip()
            # Check if the line contains an answer
            elif line.startswith('Answer:'):
                qa_dict['Answer'] = line.replace('Answer:', '').strip()

        # Append the last Q&A pair to qa_list
        if qa_dict:
            qa_list.append(qa_dict)
        return qa_list

    def EnhanceWithOpenAI(self, prompt: str, solution_title: str, document: Document, zh: bool = True) -> List[Dict[str, str]]:
        """
        Enhances a given prompt with additional information and performs a chat completion using OpenAI's GPT-3.5 Turbo model.
        
        Parameters:
        - prompt (str): The initial prompt to be enhanced.
        - solution_title (str): The title of the solution to be included in the enhanced prompt.
        - page_content (str): The content of the page to be included in the enhanced prompt.
        - zh (bool, optional): A flag to indicate whether to use a Chinese prompt template. Defaults to True.
        
        Returns:
        - List[Dict[str, str]]: A list of dictionaries containing the questions and answers generated by the model.
        
        Example:
        >>> EnhanceWithOpenAI("What is it?", "Solution Title", "Page Content")
        [{'Question': 'What is Solution Title?', 'Answer': 'It is ...'}]
        """
        prompt_template = zh_prompt_template if zh else en_prompt_template
        prompt = prompt_template.format(solution=solution_title, page=document.page_content, question_num=question_num)
        messages = [{"role": "user", "content": f"{prompt}"}]
        # error and retry handling for openai api due to request cap limit
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=2048
            )
        except Exception as e:
            logger.error("OpenAI API request failed: {}".format(e))
            return []
        raw_completion = response.choices[0]["message"]["content"].split('\n')
        logger.info("raw_completion: {}".format(raw_completion))
        # Initialize an empty list to store the Q&A pairs
        qa_list = []

        # Initialize an empty dictionary to store the current Q&A pair
        qa_dict = {}
        for line in raw_completion:
            # Check if the line contains a question
            if line.startswith('Question:'):
                # If there's already a Q&A pair in qa_dict, append it to qa_list
                if qa_dict:
                    qa_list.append(qa_dict)
                    qa_dict = {}  # Reset qa_dict for the next Q&A pair
                qa_dict['Question'] = line.replace('Question:', '').strip()
            # Check if the line contains an answer
            elif line.startswith('Answer:'):
                qa_dict['Answer'] = line.replace('Answer:', '').strip()

        # Append the last Q&A pair to qa_list
        if qa_dict:
            qa_list.append(qa_dict)
        return qa_list

    def SplitDocumentByTokenNum(self, document: Document, token_num: str) -> List[Document]:
        """
        Splits a given document into multiple documents, each containing a slice of the original document.

        Parameters:
        - document (Document): The document to be split.
        - token_num (int): The number of tokens to include in each document.

        Returns:
        - List[Document]: A list of documents, each containing a slice of the original document.
        """
        # Get the token number of input paragraph
        tokens = nltk.word_tokenize(document.page_content)
        # Calculate the total number of tokens and chunk number
        total_tokens = len(tokens)
        chunk_num = total_tokens // slice_size + 1

        # Initial document list to sttore ducoment slices seperated by 50 tokens
        documents_list = []
        # Iterate through the list of tokens, extracting slices of 50 tokens at a time
        for i in range(0, len(tokens), slice_size):
            token_slice = tokens[i:i+slice_size]
            # Join the slice of tokens back into a string
            document_slice = ' '.join(token_slice)
            # Create new Document object to store the slice
            document = Document(page_content=document_slice)
            # Append the Document object to the list of documents
            documents_list.append(document)
        return documents_list

# local debugging purpose
if __name__ == "__main__":
    # test the function
    prompt = "Do we have any solution offer to Stable Diffusion?"
    solution_title = "Stable Diffusion AWS Extensions"
    page_content = """
    Stable Diffusion AWS Extensions is a CSDC solution that...
    """
    # construct a Document object
    document = Document(page_content=page_content)
    ewb = EnhanceWithBedrock(prompt, solution_title, document)
    document_list = ewb.SplitDocumentByTokenNum(document, slice_size)
    # test the function
    for document in document_list:
        prompt = "Do we have any solution offer to Stable Diffusion?"
        solution_title = "Stable Diffusion AWS Extensions"
        enhanced_prompt = ewb.EnhanceWithClaude(prompt, solution_title, document)
        logger.info("Enhanced prompt: {}".format(enhanced_prompt))

    # test the function
    for document in document_list:
        prompt = "Do we have any solution offer to Stable Diffusion?"
        solution_title = "Stable Diffusion AWS Extensions"
        enhanced_prompt = ewb.EnhanceWithOpenAI(prompt, solution_title, document)
        logger.info("Enhanced prompt: {}".format(enhanced_prompt))
