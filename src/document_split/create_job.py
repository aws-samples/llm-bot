import boto3
import shutil
import argparse
import time

# session = boto3.session.Session(profile_name='default')
# glue = session.client('glue', region_name='us-west-2')
# s3 = session.client('s3', region_name='us-west-2')

# if __name__ == '__main__':
#     # load arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--glue_asset_bucket', type=str, default='icyxu-llm-glue-assets')
#     parser.add_argument('--glue_asset_prefix', type=str, default='document_split')
#     args = parser.parse_args()

#     glue_asset_bucket = args.glue_asset_bucket
#     glue_asset_prefix = args.glue_asset_prefix

#     # check if the s3 bucket exists
#     response = s3.list_buckets()
#     buckets = [bucket['Name'] for bucket in response['Buckets']]
#     if args.glue_asset_bucket not in buckets:
#         s3.create_bucket(Bucket=args.glue_asset_bucket)
#     else:
#         print('Bucket already exists')
    


#     # upload the dependencies
#     s3.upload_file('./dep/dist/llm_bot_dep-0.1.0-py3-none-any.whl', glue_asset_bucket, f"{glue_asset_prefix}/python-modules/llm_bot_dep-0.1.0-py3-none-any.whl")
#     s3.upload_file('glue-job-script.py', glue_asset_bucket, f"{glue_asset_prefix}/code/glue-job-script.py")
#     job_name = 'llm-test-glue-job'
#     glue.delete_job(
#         JobName=job_name
#     )

#     myJob = glue.create_job(Name=job_name, Role='GlueAdmin', GlueVersion='4.0',
#                             Command={'Name': 'glueetl',
#                                     'ScriptLocation': f"s3://{glue_asset_bucket}/{glue_asset_prefix}/code/glue-job-script.py"},
#                             ExecutionProperty={
#                                 'MaxConcurrentRuns': 3
#                             },
#                             MaxCapacity=10,
#                             )

#     myNewJobRun = glue.start_job_run(JobName=job_name,
#                                     Arguments={
#         '--additional-python-modules': "pdfminer.six==20221105,gremlinpython==3.7.0,langchain==0.0.312,beautifulsoup4==4.12.2,requests-aws4auth==1.2.3,boto3==1.28.69,nougat==0.3.3,opensearch-py==2.3.1,openai==0.28.0,unstructured==0.10.5",
#         '--extra-py-files': f"s3://{glue_asset_bucket}/{glue_asset_prefix}/python-modules/llm_bot_dep-0.1.0-py3-none-any.whl",
#         '--JOB_NAME': job_name,
#         '--S3_BUCKET': 'icyxu-llm-glue-assets',
#         '--S3_PREFIX': 'test_data/userguide/docs0.json',
#         '--AOS_ENDPOINT': 'search-icyxu-test-openai-7uy5ljdiijvyw4aajaytntf7vu.us-west-2.es.amazonaws.com',
#         '--EMBEDDING_MODEL_ENDPOINT': 'embedding-endpoint',
#         '--REGION': 'us-west-2',
#         '--OFFLINE': 'true'
#     })


session = boto3.session.Session(profile_name="atl")
glue = session.client('glue', region_name='us-west-2')
s3 = session.client('s3', region_name='us-west-2')

if __name__ == '__main__':
    # load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--glue_asset_bucket', type=str, default='llm-bot-glue-assets-atl-3163279526-us-west-2')
    parser.add_argument('--glue_asset_prefix', type=str, default='document_split_code')
    args = parser.parse_args()

    glue_asset_bucket = args.glue_asset_bucket
    glue_asset_prefix = args.glue_asset_prefix

    # check if the s3 bucket exists
    response = s3.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    if args.glue_asset_bucket not in buckets:
        s3.create_bucket(Bucket=args.glue_asset_bucket)
    else:
        print('Bucket already exists')


    # upload the dependencies
    s3.upload_file('./dep/dist/llm_bot_dep-0.1.0-py3-none-any.whl', glue_asset_bucket, f"{glue_asset_prefix}/python-modules/llm_bot_dep-0.1.0-py3-none-any.whl")
    s3.upload_file('glue_job_script_open.py', glue_asset_bucket, f"{glue_asset_prefix}/code/glue_job_script_open.py")
    job_name = 'llm-test-glue-job-2'
    glue.delete_job(
        JobName=job_name
    )

    myJob = glue.create_job(Name=job_name, Role='llm-bot-glue-role', GlueVersion='4.0',
                            Command={'Name': 'glueetl',
                                    'ScriptLocation': f"s3://{glue_asset_bucket}/{glue_asset_prefix}/code/glue_job_script_open.py"},
                            ExecutionProperty={
                                'MaxConcurrentRuns': 128
                            },
                            MaxCapacity=2,
                            Connections={
                                'Connections': [
                                    'llm-bot-test-glue-connection',
                                ]
                            },
                            )

    # s3_prefix_list is docs0.json, docs1.json, docs2.json, docs3.json, docs4.json, docs5.json, docs6.json, docs7.json, docs8.json, docs9.json
    import json
    docs_s3_bucket = "llm-bot-glue-docs-atl-3163279526-us-west-2"
    # docs = json.load(open("/home/ubuntu/Project/llm-bot/src/lambda/test/ug.json"))
    # docs = json.load(open("/home/ubuntu/Project/llm-bot/src/lambda/test/dgr_csdc_0830_1013_doc.json"))
    docs = json.load(open("/home/ubuntu/Project/llm-bot/src/lambda/test/ug_add_api_tag.json"))
    index_name = "ug-index-3"
    content_type = "ug"
    shard_num = 32
    import numpy as np
    s3_prefix_list = []
    shards = np.array_split(docs, shard_num)
    for i, shard in enumerate(shards):
        json.dump(shard.tolist(), open(f"tmp/doc_{i}.json", "w"))
        # s3.upload_file(f"tmp/doc_{i}.json", docs_s3_bucket, f"ug/doc_{i}.json")
        s3.upload_file(f"tmp/doc_{i}.json", docs_s3_bucket, f"tmp/doc_{i}.json")
        # s3_prefix_list.append(f"ug/doc_{i}.json")
        s3_prefix_list.append(f"tmp/doc_{i}.json")
    
    for s3_prefix_id, s3_prefix in enumerate(s3_prefix_list):
        myNewJobRun = glue.start_job_run(JobName=job_name,
                                        Arguments={
            '--additional-python-modules': "pdfminer.six==20221105,gremlinpython==3.7.0,langchain==0.0.312,beautifulsoup4==4.12.2,requests-aws4auth==1.2.3,boto3==1.28.69,nougat==0.3.3,opensearch-py==2.3.1,openai==0.28.0,unstructured==0.10.5",
            '--extra-py-files': f"s3://{glue_asset_bucket}/{glue_asset_prefix}/python-modules/llm_bot_dep-0.1.0-py3-none-any.whl",
            '--JOB_NAME': job_name,
            '--S3_BUCKET': docs_s3_bucket,
            '--S3_PREFIX': s3_prefix,
            # '--AOS_ENDPOINT': 'search-llm-bot-test-2-k7grhi5u2r336elfd5sb7o3lqe.us-east-1.es.amazonaws.com',
            '--AOS_ENDPOINT': "vpc-domain66ac69e0-prbg1iy4iido-ksu2bpd7eblmz6buvk5viespdq.us-west-2.es.amazonaws.com",
            '--EMBEDDING_MODEL_ENDPOINT': 'bge-large-zh-v1-5-2023-11-15-06-52-26-105-endpoint,bge-large-zh-v1-5-2023-11-15-06-52-26-105-endpoint,bge-large-en-v1-5-2023-11-15-06-19-29-526-endpoint,bge-large-en-v1-5-2023-11-15-06-19-29-526-endpoint',
            '--REGION': 'us-west-2',
            '--EMBEDDING_LANG': 'zh,zh,en,en',
            '--EMBEDDING_TYPE': 'similarity,relevance,similarity,relevance',
            '--INDEX_NAME': index_name,
            '--CONTENT_TYPE': content_type,
            '--OFFLINE': 'true'
        })
        time.sleep(0.5)

        # while True:
        #     # Get the status of the job run
        #     status = glue.get_job_run(JobName=job_name, RunId=myNewJobRun['JobRunId'])['JobRun']['JobRunState']
        #     print(status)

        #     # If finished, and current_s3_prefix_index is not the last one, then increase current_s3_prefix_index by 1
        #     if status == 'SUCCEEDED':
        #         print(f"Job {myNewJobRun['JobRunId']} finished successfully")
        #         break
        #     elif status == 'FAILED':
        #         print(f"Job {myNewJobRun['JobRunId']} failed")
        #         break
        #     else:
        #         time.sleep(60)
