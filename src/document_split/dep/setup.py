from setuptools import setup, find_packages

setup(
    name='llm_bot_dep',
    version='0.1.0',
    packages=['llm_bot_dep'],
    install_requires=[
        'langchain',
        'opensearch-py',
        'faiss_cpu',
        'sagemaker',
        'requests_aws4auth',
        'unstructured',
        'boto3',
    ],
)