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

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))))
sys.path.append(APP_DIR)
config_ini_dict = configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR, "config.ini"))
logging.info(config_ini_dict)


class Age_data_deal(object):
    def __init__(
        self,
        data_input_path,
        title_tokenize_path,
        content_tokenize_path,
        age_output_data_path,
    ):
        self.data_input_path = data_input_path
        self.title_tokenize_path = title_tokenize_path
        self.content_tokenize_path = content_tokenize_path
        self.age_output_data_path = age_output_data_path

    def age_data_output(self):
        df_list = []
        with open(self.data_input_path, encoding="UTF-8") as f:
            for line in f:
                df_list.append(line.strip().split("\001"))
        df = pd.DataFrame(df_list)
        df_columns_list = ["udid", "age"]
        for i in range(10):
            df_columns_list.append("c{}".format(i))
        df.columns = df_columns_list
        aid_set = set()
        for i in range(10):
            aid_set.update(df["c{}".format(i)])  # update 集合加入方法
        a_dict = {}
        with open(self.title_tokenize_path) as f:
            for line in f:
                temp_dict = json.loads(line.strip())
                aid = temp_dict["aid"]
                a_dict[aid] = {}
                title_tokenize = temp_dict["title_tokenize"]
                a_dict[aid]["title_tokenize"] = title_tokenize
        with open(self.content_tokenize_path) as f:
            for line in f:
                temp_dict = json.loads(line.strip())
                aid = temp_dict["aid"]
                if not aid in a_dict:
                    print(aid)
                    continue
                content_tokenize = temp_dict["content_tokenize"][:512]
                a_dict[aid]["content_tokenize"] = content_tokenize

        def temp_function(x):
            # x = datetime.datetime.now().year - int(x.split("-")[0])
            y = 0
            if x < 20:
                y = "~20"
            elif x < 25:
                y = "21~25"
            elif x < 29:
                y = "26~29"
            elif x < 35:
                y = "30~35"
            elif x < 45:
                y = "36~45"
            elif x < 50:
                y = "46~50"
            else:
                y = "50~"
            return y

        title_tokenize_list = []
        age_list = []
        for index in df.index:
            age = temp_function(int(df.loc[index, "age"]))
            temp_text_list = []
            for i in range(10):
                c_text_list = []
                c = "c{}".format(i)
                aid = str(df.loc[index, c]).strip()
                if aid:
                    if aid in a_dict:
                        c_text_list = [
                            *a_dict[aid]["title_tokenize"],
                            "<PADDING>",
                            *a_dict[aid]["content_tokenize"][:512],
                        ]
                    else:
                        print("aid不存在", index, c, aid)
                else:
                    print(index)
                temp_text_list.append(c_text_list)
            title_tokenize_list.append(temp_text_list)
            age_list.append(age)
        with open(self.age_output_data_path, "w", encoding="UTF-8") as f:
            for tag, content_tokenize in zip(age_list, title_tokenize_list):
                f.write(
                    json.dumps(
                        {"tag": tag, "all_content_tokenize": content_tokenize},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# data_input_path = config_ini_dict["file"]["agedata_input_path"]
# title_tokenize_path = config_ini_dict["file"]["title_tokenize_path"]
# content_tokenize_path = config_ini_dict["file"]["content_tokenize_path"]
# age_output_data_path = config_ini_dict["file"]["age_output_data_path"]
age = Age_data_deal(
    data_input_path=config_ini_dict["file"]["agedata_input_path"],
    title_tokenize_path=config_ini_dict["file"]["title_tokenize_path"],
    content_tokenize_path=config_ini_dict["file"]["content_tokenize_path"],
    age_output_data_path=config_ini_dict["file"]["age_output_data_path"],
)
age.age_data_output()
