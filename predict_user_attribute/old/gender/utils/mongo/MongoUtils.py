# coding=utf-8
import pymongo
from configure.ConfigureUtils import *
import configure.constans as ct


class MongoUtils(object):

    def __init__(self):
        configure_utils = get_configure_utils()
        mongo_host = configure_utils.get(ct.new_mongo_host)
        mongo_port = configure_utils.getint(ct.new_mongo_port)
        self.new_mongodb = pymongo.MongoClient(mongo_host, mongo_port)
        mongo_db_str = configure_utils.get(ct.mongo_db)
        self.db = self.new_mongodb[mongo_db_str]
        self.hx_moment_sources = self.db.hx_moment_sources
        self.hx_moment_spider_filter = self.db.hx_moment_spider_filter
        self.hx_moment_notype = self.db.hx_moment_notype
        self.hx_moment_record = self.db.hx_moment_record


if __name__ == '__main__':
    configure_utils = init_configure_utils("test")
    mongo_utils = MongoUtils()
    one = mongo_utils.hx_moment_notype.find_one({'_id': "ef76c5b22104798ea70c9198f935871b"})
    print(one)
