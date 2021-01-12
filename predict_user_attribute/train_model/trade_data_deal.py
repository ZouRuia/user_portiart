import json
import os
import sys
import logging
import configparser
import collections
import numpy as np
import pandas as pd

# import tensorflow as tf
import LAC
import happybase
import joblib
import sklearn.utils
from sklearn.feature_selection import RFE

APP_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(APP_DIR)
config_ini_dict = configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR, "config.ini"))
logging.info(config_ini_dict)


class Trade_data_deal(object):
    def __init__(
        self,
        data_input_path,
        title_tokenize_path,
        content_tokenize_path,
        trade_output_data_path,
    ):
        self.data_input_path = data_input_path
        self.title_tokenize_path = title_tokenize_path
        self.content_tokenize_path = content_tokenize_path
        self.trade_output_data_path = trade_output_data_path

    def trade_data_output(self):
        df_list = []
        with open(self.data_input_path) as f:
            for line in f:
                df_list.append(line.strip().split("\001"))
        df = pd.DataFrame(df_list)
        df[1] = df[1].apply(lambda x: x.lower())
        all_counter = collections.Counter([v for v in df[1]])

        def temp_function(x):
            # x = datetime.datetime.now().year - int(x.split("-")[0])
            y = 0
            if x in set(["it", "互联网", "tmt从业人员", "人工智能/前沿科技", "企业服务/云计算/大数据"]):
                y = "IT/移动互联网"
            elif x in set(["教育/专业服务/培训", "内容/营销/传播", "营销", "贸易/零售批发", "酒店/餐饮/旅游"]):
                y = "零售消费"
            elif x in set(["金融", "投资人"]):
                y = "金融"
            elif x in set(
                [
                    "汽车/出行",
                ]
            ):
                y = "汽车"
            elif x in set(["电商/仓储物流", "仓储物流", "商业服务", "商业服务-o2o/服务/社区"]):
                y = "电商"
            elif x in set(["加工制造"]):
                y = "制造业"
            elif x in set(["智能硬件/硬件制造"]):
                y = "IOT"
            elif x in set(["文化/体育/娱乐业"]):
                y = "游戏"
            elif x in set(
                [
                    "房地产",
                ]
            ):
                y = "房地产"
            elif x in set(
                [
                    "制药医疗/生物/卫生保健",
                ]
            ):
                y = "医疗"
            elif x in set(
                ["创业者", "政府/非盈利机构", "法律", "其他", "媒体", "能源相关", "农业", "学生", "-1"]
            ):
                y = "其他"
            else:
                temp = "a"
            return y

        df[1] = df[1].apply(lambda x: temp_function(x))
        df.columns = ["aid", "trade", *["c{}".format(v) for v in range(10)]]
        article_dict = {}
        with open(self.title_tokenize_path, encoding="utf8") as f:
            for line in f:
                line_dict = json.loads(line.strip())
                aid = line_dict["aid"]
                article_dict[aid] = {}
                article_dict[aid]["title_tokenize"] = line_dict["title_tokenize"]
        with open(self.content_tokenize_path, encoding="utf8") as f:
            for line in f:
                line_dict = json.loads(line.strip())
                aid = line_dict["aid"]
                if not aid in article_dict:
                    print(aid)
                    continue
                article_dict[aid]["content_tokenize"] = line_dict["content_tokenize"][
                    :512
                ]
        text_list = []
        y_list = []
        for index in df.index:
            y = df.loc[index, "trade"]
            temp_text_list = []
            for i in range(10):
                c_text_list = []
                c = "c{}".format(i)
                aid = str(df.loc[index, c]).strip()
                if aid:
                    if aid in article_dict:
                        c_text_list = [
                            *article_dict[aid]["title_tokenize"],
                            "<PADDING>",
                            *article_dict[aid]["content_tokenize"][:512],
                        ]
                    else:
                        print("aid不存在", index, c, aid)
                else:
                    print(index)
                temp_text_list.append(c_text_list)
            text_list.append(temp_text_list)
            y_list.append(y)
        with open(self.trade_output_data_path, "w") as f:
            for tag, content_tokenize in zip(y_list, text_list):
                f.write(
                    json.dumps(
                        {"tag": tag, "all_content_tokenize": content_tokenize},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


data_input_path = config_ini_dict["file"]["trade_input_path"]
title_tokenize_path = config_ini_dict["file"]["title_tokenize_path"]
content_tokenize_path = config_ini_dict["file"]["content_tokenize_path"]
trade_output_data_path = config_ini_dict["file"]["trade_output_data_path"]
trade = Trade_data_deal(
    data_input_path=config_ini_dict["file"]["trade_input_path"],
    title_tokenize_path=config_ini_dict["file"]["title_tokenize_path"],
    content_tokenize_path=config_ini_dict["file"]["content_tokenize_path"],
    trade_output_data_path=config_ini_dict["file"]["trade_output_data_path"],
)
trade.trade_data_output()
