# coding=utf-8
import configparser
import os
from utils.BaseUtil import *
import logging


@singleton
class ConfigureUtils(object):
    def __init__(self, model=None):
        self.model = model
        self.verification()
        logging.info("初始化配置工具类")
        logging.info("配置工具类模式: {}".format(self.model))

        self.public = "public"

        configure_path = os.path.dirname(os.path.realpath(__file__)) + "/configure.conf"
        self.config_parser = configparser.ConfigParser()
        self.config_parser.read(configure_path, encoding='utf-8')

    def get(self, option):
        # 判断该配置是否为本地配置
        local_configure = self.config_parser.has_option(self.model, option)
        if local_configure:
            return self.config_parser.get(self.model, option)
        else:
            return self.config_parser.get(self.public, option)

    def getint(self, option):
        # 判断该配置是否为本地配置
        local_configure = self.config_parser.has_option(self.model, option)
        if local_configure:
            return self.config_parser.getint(self.model, option)
        else:
            return self.config_parser.getint(self.public, option)

    def verification(self):
        if not self.model:
            print("配置工具未初始化")
            raise Exception("配置工具未初始化")

"""该方法请勿轻易调用，该方法应该在服务入口处调用"""
def init_configure_utils(model="test"):
    if model:
        configure_utils = ConfigureUtils(model)
    else:
        configure_utils = ConfigureUtils()
    return configure_utils

def get_configure_utils():
    configure_utils = ConfigureUtils()
    return configure_utils
