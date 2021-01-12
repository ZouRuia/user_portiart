# 导入Flask类

# import json
import time
import os
import logging
import sys
import pandas as pd

# import collections
import configparser
from flask import Flask, jsonify, request

from model.User_portrait import UserProfileSVMModel

APP_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(APP_DIR)

config_ini_dict = configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR, "config.ini"))
logging.info(config_ini_dict)
# # 实例化，可视为固定格式
app = Flask(__name__)

# route()方法用于设定路由；类似spring路由配置
@app.route("/test_1/", methods=["post", "get"])
def predict():
    if request.method == "POST":
        start = time.time()
        params = request.json
        out_put = userprofile.predict(params)
        end = time.time()
        print("time: {:.2f} s".format(end - start))
        return jsonify(out_put)


if __name__ == "__main__":
    userprofile = UserProfileSVMModel(
        filepath_age=config_ini_dict["file"]["filepath_age"],
        filepath_position=config_ini_dict["file"]["filepath_position"],
        filepath_trade=config_ini_dict["file"]["filepath_trade"],
        filepath_gender=config_ini_dict["file"]["filepath_gender"],
        data_jsonl=config_ini_dict["file"]["data_jsonl"],
        word2vector_file_path=config_ini_dict["file"]["word2vector_file_path"],
        age200=pd.read_csv(config_ini_dict["file"]["age_200"]),
        gender200=pd.read_csv(config_ini_dict["file"]["gender_200"]),
        trade200=pd.read_csv(config_ini_dict["file"]["trade_200"]),
        input_number=10,
        sentence_maxlen=512,
    )

    app.run(host="0.0.0.0", port=8889)
