# coding=utf-8

from utils.segment.segment_service import SegmentService
from utils.redis.RedisUtils import RedisUtils
from utils.BaseUtil import *
import re


class FilteringService(object):
    """"过滤服务类"""

    def __init__(self, redis_filter_key_pro, min_score=0.5, min_number_count=2, redis_expire=60 * 60 * 6):
        self.segment_service = SegmentService()
        self.src = RedisUtils().src
        # 默认数据过期时间
        self.default_redis_expire = redis_expire
        self.default_min_number_count = min_number_count
        if not redis_filter_key_pro:
            raise Exception("redis_filter_key_pro 不能为空")
        self.default_redis_filter_key_pro = redis_filter_key_pro
        self.default_min_score = min_score

    def is_filter(self, _id, doc):
        """
        :param _id: 唯一ID
        :param doc: 文本内容
        :return: 是否过滤, 相似ID
        """
        key_pro = self.default_redis_filter_key_pro
        min_number_count = self.default_min_number_count
        redis_expire = self.default_redis_expire
        min_score = self.default_min_score

        key_set = self.src.keys(pattern=key_pro + "_*")

        word_set = set(self.segment_service.cut(doc))
        print(word_set)
        print(self.get_number_set(word_set))
        for key in key_set:
            _word_set = self.src.smembers(key)
            set_number = compare_set(_word_set, word_set)
            score = set_number / round(min(len(_word_set), len(word_set)))
            repeat_id = key.split("_")[-1]
            if score >= min_score:
                return True, "分数相似过滤, _id：{}".format(repeat_id)
            elif compare_set(self.get_number_set(_word_set),
                             self.get_number_set(word_set)) >= min_number_count:
                return True, "数字相似过滤, _id：{}".format(repeat_id)

        key = key_pro + '_{}'.format(_id)
        self.insert_redis(key, word_set, redis_expire)
        return False, None

    @staticmethod
    def get_number_set(word_set):
        """
        获取标志性信息
        :param word_set: 词列表
        :param vest_id: 马甲ID
        :return: 有标志性的词列表
        """
        sign_set = set()
        for word in word_set:
            if is_number(word):
                if "%" in word:
                    word = word.split("%")[0]  # 获取百分数
                    if '.' in word:
                        word = re.search(r".*?\..?", word).group()
                    word += "%"
                elif '.' in word:
                    word = re.search(r".*?\..?", word).group()
                sign_set.add(word)
        return sign_set

    def insert_redis(self, key, word_set, redis_expire):
        if len(word_set) > 0:
            self.src.sadd(key, *word_set)
            self.src.expire(key, redis_expire)


if __name__ == '__main__':
    fs = FilteringService(redis_filter_key_pro="test_video")
    is_filter, reason = fs.is_filter("5", "2月19日 近日，人社部、公安部、交通运输部、国家卫生健康委、国家铁路集团印发")
    print(is_filter)
    print(reason)
