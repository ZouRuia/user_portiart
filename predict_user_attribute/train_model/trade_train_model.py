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
from trade_number_transform import Number_Transform

APP_DIR = os.path.dirname(os.path.realpath("__file__"))
config_ini_dict = configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR, "config.ini"))
logging.info(config_ini_dict)


class Trade_Model(object):
    def __init__(self, trade_rank_path, model_path):
        self.trade_rank_path = trade_rank_path
        self.model_path = model_path

    def train(self, x_npa, y_npa, C):
        np.shape(x_npa)
        x_npa_1 = x_npa.reshape(len(y_npa), 5120)
        x_npa_df = pd.DataFrame(x_npa_1)
        y_npa_df = pd.DataFrame(y_npa)
        y_npa_df.columns = ["trade"]
        df_trade = pd.merge(y_npa_df, x_npa_df, on=y_npa_df.index)
        df_trade = df_trade.drop("key_0", axis=1)
        df_trade.columns = ["trade", *["c{}".format(v) for v in range(5120)]]
        temp_0 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 0
        )
        temp_1 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 1
        )
        temp_2 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 2
        )
        temp_3 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 3
        )
        temp_4 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 4
        )
        temp_5 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 5
        )
        temp_6 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 6
        )
        temp_7 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 7
        )
        temp_8 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 8
        )
        temp_9 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 9
        )
        temp_10 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 10
        )
        temp_0 = temp_0.drop("trade", axis=1)
        temp_1 = temp_1.drop("trade", axis=1)
        temp_2 = temp_2.drop("trade", axis=1)
        temp_3 = temp_3.drop("trade", axis=1)
        temp_4 = temp_4.drop("trade", axis=1)
        temp_5 = temp_5.drop("trade", axis=1)
        temp_6 = temp_6.drop("trade", axis=1)
        temp_7 = temp_7.drop("trade", axis=1)
        temp_8 = temp_8.drop("trade", axis=1)
        temp_9 = temp_9.drop("trade", axis=1)
        temp_10 = temp_10.drop("trade", axis=1)
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_0)
        labels = model.predict(temp_0)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_00 = pd.merge(labels, temp_0, on=labels.index)
        temp_00 = temp_00.drop("key_0", axis=1)
        temp_000 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 0
        )
        temp_000 = temp_000["trade"]
        temp_end_0 = pd.merge(temp_000, temp_00, on=temp_00.index)
        temp_end_0 = temp_end_0.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_0["b"]))
        b = max(a, key=a.get)
        temp_end_0 = temp_end_0.groupby(temp_end_0.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_1)
        labels = model.predict(temp_1)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_11 = pd.merge(labels, temp_1, on=labels.index)
        temp_11 = temp_11.drop("key_0", axis=1)
        temp_111 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 1
        )
        temp_111 = temp_111["trade"]
        temp_end_1 = pd.merge(temp_111, temp_11, on=temp_11.index)
        temp_end_1 = temp_end_1.drop("key_0", axis=1)
        # temp_end_1.groupby('b').count()
        a = dict(collections.Counter(temp_end_1["b"]))
        b = max(a, key=a.get)
        temp_end_1 = temp_end_1.groupby(temp_end_1.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_2)
        labels = model.predict(temp_2)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_22 = pd.merge(labels, temp_2, on=labels.index)
        temp_22 = temp_22.drop("key_0", axis=1)
        temp_222 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 2
        )
        temp_222 = temp_222["trade"]
        temp_end_2 = pd.merge(temp_222, temp_22, on=temp_22.index)
        temp_end_2 = temp_end_2.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_2["b"]))
        b = max(a, key=a.get)
        temp_end_2 = temp_end_2.groupby(temp_end_2.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_3)
        labels = model.predict(temp_3)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_33 = pd.merge(labels, temp_3, on=labels.index)
        temp_33 = temp_33.drop("key_0", axis=1)
        temp_333 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 3
        )
        temp_333 = temp_333["trade"]
        temp_end_3 = pd.merge(temp_333, temp_33, on=temp_33.index)
        temp_end_3 = temp_end_3.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_3["b"]))
        b = max(a, key=a.get)
        temp_end_3 = temp_end_3.groupby(temp_end_3.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_4)
        labels = model.predict(temp_4)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_44 = pd.merge(labels, temp_4, on=labels.index)
        temp_44 = temp_44.drop("key_0", axis=1)
        temp_444 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 4
        )
        temp_444 = temp_444["trade"]
        temp_end_4 = pd.merge(temp_444, temp_44, on=temp_44.index)
        temp_end_4 = temp_end_4.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_4["b"]))
        b = max(a, key=a.get)
        temp_end_4 = temp_end_4.groupby(temp_end_4.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_5)
        labels = model.predict(temp_5)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_55 = pd.merge(labels, temp_5, on=labels.index)
        temp_55 = temp_55.drop("key_0", axis=1)
        temp_555 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 5
        )
        temp_555 = temp_555["trade"]
        temp_end_5 = pd.merge(temp_555, temp_55, on=temp_55.index)
        temp_end_5 = temp_end_5.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_5["b"]))
        b = max(a, key=a.get)
        temp_end_5 = temp_end_5.groupby(temp_end_5.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_6)
        labels = model.predict(temp_6)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_66 = pd.merge(labels, temp_6, on=labels.index)
        temp_66 = temp_66.drop("key_0", axis=1)
        temp_666 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 6
        )
        temp_666 = temp_666["trade"]
        temp_end_6 = pd.merge(temp_666, temp_66, on=temp_66.index)
        temp_end_6 = temp_end_6.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_6["b"]))
        b = max(a, key=a.get)
        temp_end_6 = temp_end_6.groupby(temp_end_6.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_7)
        labels = model.predict(temp_7)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_77 = pd.merge(labels, temp_7, on=labels.index)
        temp_77 = temp_77.drop("key_0", axis=1)
        temp_777 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 7
        )
        temp_777 = temp_777["trade"]
        temp_end_7 = pd.merge(temp_777, temp_77, on=temp_77.index)
        temp_end_7 = temp_end_7.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_7["b"]))
        b = max(a, key=a.get)
        temp_end_7 = temp_end_7.groupby(temp_end_7.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_8)
        labels = model.predict(temp_8)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_88 = pd.merge(labels, temp_8, on=labels.index)
        temp_88 = temp_88.drop("key_0", axis=1)
        temp_888 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 8
        )
        temp_888 = temp_888["trade"]
        temp_end_8 = pd.merge(temp_888, temp_88, on=temp_88.index)
        temp_end_8 = temp_end_8.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_8["b"]))
        b = max(a, key=a.get)
        temp_end_8 = temp_end_8.groupby(temp_end_8.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_9)
        labels = model.predict(temp_9)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_99 = pd.merge(labels, temp_9, on=labels.index)
        temp_99 = temp_99.drop("key_0", axis=1)
        temp_999 = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 9
        )
        temp_999 = temp_999["trade"]
        temp_end_9 = pd.merge(temp_999, temp_99, on=temp_99.index)
        temp_end_9 = temp_end_9.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_9["b"]))
        b = max(a, key=a.get)
        temp_end_9 = temp_end_9.groupby(temp_end_9.index).filter(
            lambda x: float(x["b"]) == b
        )
        # 聚类
        model = KMeans(n_clusters=2)
        model.fit(temp_10)
        labels = model.predict(temp_10)
        labels = pd.DataFrame(labels)
        labels.columns = ["b"]
        temp_ten = pd.merge(labels, temp_10, on=labels.index)
        temp_ten = temp_ten.drop("key_0", axis=1)
        temp_doubleten = df_trade.groupby(df_trade.index).filter(
            lambda x: float(x["trade"]) == 10
        )
        temp_doubleten = temp_doubleten["trade"]
        temp_end_10 = pd.merge(temp_doubleten, temp_ten, on=temp_ten.index)
        temp_end_10 = temp_end_10.drop("key_0", axis=1)
        a = dict(collections.Counter(temp_end_10["b"]))
        b = max(a, key=a.get)
        temp_end_10 = temp_end_10.groupby(temp_end_10.index).filter(
            lambda x: float(x["b"]) == b
        )
        x_npa1 = pd.concat([temp_end_0, temp_end_1], axis=0, ignore_index=True)
        x_npa1 = pd.concat([x_npa1, temp_end_2], axis=0, ignore_index=True)
        x_npa1 = pd.concat([x_npa1, temp_end_3], axis=0, ignore_index=True)
        x_npa1 = pd.concat([x_npa1, temp_end_4], axis=0, ignore_index=True)
        x_npa1 = pd.concat([x_npa1, temp_end_5], axis=0, ignore_index=True)
        x_npa1 = pd.concat([x_npa1, temp_end_6], axis=0, ignore_index=True)
        x_npa1 = pd.concat([x_npa1, temp_end_7], axis=0, ignore_index=True)
        x_npa1 = pd.concat([x_npa1, temp_end_8], axis=0, ignore_index=True)
        x_npa1 = pd.concat([x_npa1, temp_end_9], axis=0, ignore_index=True)
        x_npa1 = pd.concat([x_npa1, temp_end_10], axis=0, ignore_index=True)
        x_npa1 = x_npa1.drop("b", axis=1)
        x_data = x_npa1.drop("trade", axis=1)
        y_data = x_npa1["trade"]
        ss = StandardScaler()
        x_data = ss.fit_transform(x_data)
        x_data = pd.DataFrame(x_data)
        estimator = SVC(kernel="linear")
        selector = RFE(estimator, n_features_to_select=200, step=64)
        selector = selector.fit(x_data, y_data)
        rank = selector.ranking_
        rank = pd.DataFrame(rank)
        rank.columns = ["rank"]
        rank.to_csv(self.trade_rank_path, index=False)
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
        data_x = all_data.drop("trade", axis=1)
        data_y = all_data["trade"]
        x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(
            data_x, data_y, random_state=155
        )
        svm = SVC(C=C, kernel="linear")
        svm.fit(x_train_svm, y_train_svm)
        # lr.fit(x_train,y_train)
        preds = svm.predict(x_test_svm)
        print("准确率为%f" % ((preds == y_test_svm).sum() / float(y_test_svm.shape[0])))
        return svm

    def save(self, svm):
        joblib.dump(svm, self.model_path)

    def load(self):
        model_trade = joblib.load(self.model_path)
        return model_trade


model = Trade_Model(
    trade_rank_path=config_ini_dict["file"]["trade_rank_path"],
    model_path=config_ini_dict["file"]["trade_model_path"],
)
nt = Number_Transform(
    trade_output_data_path=config_ini_dict["file"]["trade_output_data_path"],
    word2vector_file_path=config_ini_dict["file"]["word2vector_file_path"],
    input_number=10,
    sentence_maxlen=512,
)
x_npa, y_npa = nt.out_x_y()
model.train(x_npa, y_npa, C=1)
