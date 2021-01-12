import json
import os
import re
import collections
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import LAC
import logging
import configparser
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import happybase
from sklearn.cluster import KMeans
import joblib
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from gender_number_transform import Number_Transform

APP_DIR = os.path.dirname(os.path.realpath("__file__"))
config_ini_dict = configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR, "config.ini"))
logging.info(config_ini_dict)


class Gender_Model(object):
    def __init__(self, gender_rank_path, model_path):
        self.gender_rank_path = gender_rank_path
        self.model_path = model_path

    def train(self, x_npa, y_npa, C):
        y_npa = pd.DataFrame(y_npa)
        y_npa.columns = ["gender"]
        x_npa = x_npa.reshape(len(y_npa), 5120)
        x_npa = pd.DataFrame(x_npa)
        x_npa.columns = [*["c{}".format(v) for v in range(5120)]]
        gender_data = pd.merge(y_npa, x_npa, on=y_npa.index)
        gender_data = gender_data.drop("key_0", axis=1)
        temp_0 = gender_data.groupby(gender_data.index).filter(
            lambda x: float(x["gender"]) == 0
        )
        temp_1 = gender_data.groupby(gender_data.index).filter(
            lambda x: float(x["gender"]) == 1
        )
        temp_0 = temp_0.drop("gender", axis=1)
        temp_1 = temp_1.drop("gender", axis=1)
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_0)
        labels = model.predict(temp_0)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_00 = pd.merge(labels, temp_0, on=labels.index)
        temp_00 = temp_00.drop("key_0", axis=1)
        temp_000 = gender_data.groupby(gender_data.index).filter(
            lambda x: float(x["gender"]) == 0
        )
        temp_000 = temp_000["gender"]
        temp_end_0 = pd.merge(temp_000, temp_00, on=temp_00.index)
        temp_end_0 = temp_end_0.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_0["b"]))
        b = max(a, key=a.get)
        temp_end_0 = temp_end_0.groupby(temp_end_0.index).filter(
            lambda x: float(x["b"]) == b
        )
        model = KMeans(n_clusters=2)
        model.fit(temp_1)
        labels = model.predict(temp_1)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_11 = pd.merge(labels, temp_1, on=labels.index)
        temp_11 = temp_11.drop("key_0", axis=1)
        temp_111 = gender_data.groupby(gender_data.index).filter(
            lambda x: float(x["gender"]) == 1
        )
        temp_111 = temp_111["gender"]
        temp_end_1 = pd.merge(temp_111, temp_11, on=temp_11.index)
        temp_end_1 = temp_end_1.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_1["b"]))
        b = max(a, key=a.get)
        temp_end_1 = temp_end_1.groupby(temp_end_1.index).filter(
            lambda x: float(x["b"]) == b
        )
        x_npa1 = pd.concat([temp_end_0, temp_end_1], axis=0, ignore_index=True)
        x_npa1 = x_npa1.drop("b", axis=1)
        x_data = x_npa1.drop("gender", axis=1)
        y_data = x_npa1["gender"]
        ss = StandardScaler()
        x_data = ss.fit_transform(x_data)
        x_data = pd.DataFrame(x_data)
        estimator = SVC(kernel="linear")
        selector = RFE(estimator, n_features_to_select=200, step=64)
        selector = selector.fit(x_data, y_data)
        rank = selector.ranking_
        rank = pd.DataFrame(rank)
        rank.columns = ["rank"]
        rank.to_csv(self.gender_rank_path, index=False)
        x_T = x_data.T
        rank_data = pd.merge(rank, x_T, on=rank.index)
        rank_data = rank_data.drop("key_0", axis=1)
        rank_data_200 = rank_data.groupby(rank_data.index).filter(
            lambda x: float(x["rank"]) == 1
        )
        rank_data_200 = rank_data_200.drop("rank", axis=1)
        rank_data_200 = rank_data_200.T
        all_data = pd.merge(y_data, rank_data_200, on=y_data.index)
        all_data = all_data.drop("key_0", axis=1)
        data_x = all_data.drop("gender", axis=1)
        data_y = all_data["gender"]
        x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(
            data_x, data_y, random_state=155
        )
        svm = SVC(C=C, kernel="linear")
        svm.fit(x_train_data, y_train_data)
        # lr.fit(x_train,y_train)
        preds = svm.predict(x_test_data)
        print("准确率为%f" % ((preds == y_test_data).sum() / float(y_test_data.shape[0])))
        return svm

    def save(self, svm):
        joblib.dump(svm, self.model_path)

    def load(self):
        model_gender = joblib.load(self.model_path)
        return model_gender


model = Gender_Model(
    gender_rank_path=config_ini_dict["file"]["gender_rank_path"],
    model_path=config_ini_dict["file"]["gender_model_path"],
)
nt = Number_Transform(
    gender_output_data_path=config_ini_dict["file"]["gender_output_data_path"],
    word2vector_file_path=config_ini_dict["file"]["word2vector_file_path"],
    input_number=10,
    sentence_maxlen=512,
)
x_npa, y_npa = nt.out_x_y()
model.train(x_npa, y_npa, C=1)
