# coding=utf-8
from kafka import KafkaProducer, SimpleProducer, KafkaClient
import json


# 单例类装饰器
def singleton(cls):
    _instance = {}

    def _deco(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _deco


@singleton
class KafkaSpider(object):
    def __init__(self, topicle='spider'):
        self.host1 = 'hadoop3:9092'
        self.host2 = 'hadoop4:9092'
        self.host3 = 'kafka001:9092'
        self.host4 = 'kafka002:9092'
        self.host5 = 'kafka003:9092'
        self.spider_topicle = topicle
        try:
            self.producer2 = KafkaProducer(bootstrap_servers=[self.host3, self.host4, self.host5],
                                           request_timeout_ms=3000)
        except Exception as e:
            self.producer2 = None
            print("kafka--{}请求超时：{}".format(self.spider_topicle, e))

    def producer_kafka(self, res, topicle=None):
        if not topicle:
            topicle = self.spider_topicle
        res_json = json.dumps(res, ensure_ascii=False)
        # b_res_str = bytes("{}".format(res_json), 'utf-8')
        try:
            b_res_str = str.encode(res_json)
        except Exception as e:
            res_json = res_json.encode()
            b_res_str = str.encode(res_json)
        if self.producer2 is not None:
            self.producer2.send(topicle, value=b_res_str)
            self.producer2.flush()
            print('topicle=={}  kafka数据写入成功'.format(topicle))
        else:
            print('topicle=={}  kafka数据写入失败'.format(topicle))
