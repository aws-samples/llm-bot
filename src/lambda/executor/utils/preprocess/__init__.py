import re
import boto3
from .service_intent_recognition import get_service_name


# language symbols
CHINESE = 'zh'
ENGLISH = 'en'


def language_check(query):
    """直接通过是否包含中文字符来判断
    Args:
        query (_type_): _description_

    Returns:
        _type_: _description_
    """
    r = re.findall('[\u4e00-\u9fff]+', query)
    if not r:
        return ENGLISH 
    else:
        return CHINESE


def is_api_query(query)-> bool:
    """
    Args:
        query (_type_): _description_

    Returns:
        bool: _description_
    """
    return 'api' in query.lower()


class Translator:
    client = None

    @staticmethod
    def get_client():
        client = boto3.client('translate')
        return client

    @classmethod
    def translate(cls,query,source_lang,target_lang):
        """

        Args:
            query (_type_): _description_
            source_lang (_type_): _description_
            target_lang (_type_): _description_
        """
        if cls.client is None:
            cls.client = cls.get_client()
        
        r = cls.client.translate_text(
            Text=query,
            SourceLanguageCode=source_lang,
            TargetLanguageCode=target_lang
        )
        return r['TranslatedText']

def query_translate(query,lang):
    source_lang = lang 

    if lang == CHINESE:
        target_lang = ENGLISH
    else:
        target_lang = CHINESE
    
    translated_text = Translator.translate(
        query,source_lang,target_lang
        )
    
    return translated_text

def run_preprocess(query):
    """
    1. whether API's query.
    2. Language check.
    3. query translate

    Args:
        text (_type_): _description_
    """
    is_api = is_api_query(query)
    query_lang = language_check(query)

    translated_text = query_translate(query,lang=query_lang)

    service_names = get_service_name(query)

    ret = {
        'query':query,
        'is_api_query': is_api,
        'translated_text': translated_text,
        'query_lang':query_lang,
        'service_names': service_names
    }

    return ret  


