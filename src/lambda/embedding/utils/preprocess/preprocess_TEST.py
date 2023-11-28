from preprocess import run_preprocess  

import unittest


class TestDict(unittest.TestCase):
    test_data = [
        {
            "query":'我想调用Amazon Bedrock中的基础模型，应该使用什么API?',
            "is_api": True,
            'query_lang': 'zh'
        },
        {
            "query":'S3 CopyObject API最大可以复制多大的对象？',
            "is_api": True,
            'query_lang': 'zh'
        },
        {
            "query":'AWS Backup 支持对 Amazon S3 存储桶进行连续备份吗?',
            "is_api": False,
            'query_lang': 'zh'
        },
        {
            "query":'AWS有支持Intel可信计算技术Software Guard Extensions (SGX) 功能的EC2实例吗？',
            "is_api": False,
            'query_lang': 'zh'
        },
        {
            "query":'如何设置AWS IAM策略禁用特定的 AWS 区域？',
            "is_api": False,
            'query_lang': 'zh'
        },
        {
            "query":'Unable to disable SMS MFA for root user',
            "is_api": False,
            'query_lang': 'en'
        }]
    

    def test_valid(self):
        for datum in self.test_data:
            r = run_preprocess(datum['query'])
            print(r)
            self.assertEqual(r['is_api_query'],datum['is_api'])
            
            self.assertEqual(r['query_lang'],datum['query_lang'])
            




if __name__ == '__main__':
    unittest.main()