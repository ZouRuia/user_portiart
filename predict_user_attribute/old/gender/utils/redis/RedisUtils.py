# coding=utf-8
# import redis
from redis import StrictRedis
import logging
from rediscluster import RedisCluster


class RedisUtils(object):
    def __init__(self):
        self.redis_num = 60 * 60 * 3
        # self.redis_num = 60 * 3
        try:
            # 构建所有的节点，Redis会使⽤CRC16算法，将键和值写到某个节点上
            self.startup_nodes = [
                {'host': 'redis_cluster_w', 'port': '6373'},
                {'host': 'redis_cluster_w', 'port': '6374'},
                {'host': 'redis_cluster_w', 'port': '6375'},
                {'host': 'redis_cluster_w', 'port': '6376'},
                {'host': 'redis_cluster_w', 'port': '6377'},
                {'host': 'redis_cluster_w', 'port': '6378'},
            ]
            # 构建StrictRedisCluster对象
            self.src = RedisCluster(startup_nodes=self.startup_nodes, decode_responses=True)
        except Exception as e:
            self.src = None
            print(e)

    def number_redis(self, content_id, number_set):
        if len(number_set) > 0:
            key = 'hour24_number_set_{}'.format(content_id)
            self.src.sadd(key, *number_set)
            self.src.expire(key, self.redis_num)
            logging.info('写入redis成功：{}'.format(key))


if __name__ == '__main__':
    src = RedisUtils().src
    # for key in src.keys(pattern="development-hour24_number_set_*"):
    #     src.delete(key)
    # print(src.keys(pattern="development-hour24_number_set_*"))
    #
    #
    # print(src.hset("auto_montent_tag_test", "0", "1577886607"))
    # print(src.hset("auto_montent_tag_test", "1", "1577886607"))
    # print(src.hset("auto_montent_tag_test", "2", "1577886607"))
    # print(src.hset("auto_montent_tag_test", "3", "1577886607"))
    # print(src.hset("auto_montent_tag_test", "4", "1577886607"))
    # print(src.hset("auto_montent_tag_test", "5", "1577886607"))
    # print(src.hget("auto_montent_tag_test", 5))

    # print(src.keys(pattern="test_video_*"))

    for key in src.keys(pattern="hx_moment_set_*"):
        print(key)

