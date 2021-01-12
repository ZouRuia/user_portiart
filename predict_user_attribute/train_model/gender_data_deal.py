import json
import os
import sys
import logging
import configparser
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import LAC
import happybase
import sklearn.utils

APP_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(APP_DIR)
config_ini_dict = configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR, "config.ini"))
logging.info(config_ini_dict)


class Gender_data_deal(object):
    def __init__(
        self,
        data_input_path,
        title_tokenize_path,
        content_tokenize_path,
        gender_output_data_path,
    ):
        self.data_input_path = data_input_path
        self.title_tokenize_path = title_tokenize_path
        self.content_tokenize_path = content_tokenize_path
        self.gender_output_data_path = gender_output_data_path

    def gender_data_output(self):
        df_list = []
        with open(self.data_input_path) as f:
            for line in f:
                df_list.append(line.strip().split("\001"))
        df = pd.DataFrame(df_list)
        df_columns_list = ["udid", "gender"]
        for i in range(10):
            df_columns_list.append("c{}".format(i))
        df.columns = df_columns_list
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
        text_list = []
        y_list = []
        for index in df.index:
            y = df.loc[index, "gender"]
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
            text_list.append(temp_text_list)
            y_list.append(y)
        with open(self.gender_output_data_path, "w") as f:
            for tag, content_tokenize in zip(y_list, text_list):
                f.write(
                    json.dumps(
                        {"tag": tag, "all_content_tokenize": content_tokenize},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# data_input_path = config_ini_dict["file"]["genderdata_input_path"]
# title_tokenize_path = config_ini_dict["file"]["title_tokenize_path"]
# content_tokenize_path = config_ini_dict["file"]["content_tokenize_path"]
# gender_output_data_path = config_ini_dict["file"]["gender_output_data_path"]
gender = Gender_data_deal(
    data_input_path=config_ini_dict["file"]["genderdata_input_path"],
    title_tokenize_path=config_ini_dict["file"]["title_tokenize_path"],
    content_tokenize_path=config_ini_dict["file"]["content_tokenize_path"],
    gender_output_data_path=config_ini_dict["file"]["gender_output_data_path"],
)
gender.gender_data_output()
