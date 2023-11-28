from service_intent_recognition import get_service_name  

import unittest


class TestDict(unittest.TestCase):
    test_data = [
        {
            "query":'如何将RDS数据库实例还原到指定时间点？',
            'service': []
        },
        {
            "query":'AWS有支持Intel可信计算技术Software Guard Extensions (SGX) 功能的EC2实例吗？',
            "service": ["Amazon EC2"]
        },
        {
            "query":'S3 CopyObject API最大可以复制多大的对象？',
            "service": ["Amazon S3"]
        },
        {
            "query":'AWS Backup 支持对 Amazon S3 存储桶进行连续备份吗?',
            "service": ["Amazon S3","AWS Backup"]
        },
        {
            "query":'如何设置AWS IAM策略禁用特定的 AWS 区域？',
            "service": ["AWS IAM Identity Center"]
        }]
    

    def test_valid(self):
        for datum in self.test_data:
            r = get_service_name(datum['query'])
            print(r)
            self.assertEqual(set(r),set(datum['service']))


if __name__ == '__main__':
    unittest.main()