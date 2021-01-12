import numpy as np
import scipy as sp
import pandas as pd
import joblib
import pickle
import os
import gc
import sys
from settings import *
from segment.segment_service import SegmentService

from doc_vector.doc_vector_client import DocVectorClient
from configure.ConfigureUtils import *
import lightgbm as lgb

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

dir_path = os.path.dirname(os.path.abspath(__file__))


def save_sex_data(data):
    if data in [0, 1, 2]:
        return data
    try:
        datas = data.split(',')
    except:
        datas = ['', ]
    if type(datas) == list:
        the_data = datas[0]
        if the_data:
            if the_data == '-1' or the_data == 'null':
                return 0
            else:
                return int(the_data)
        return 0


def change_sex_map_data(tag):
    return sex_map_data.get(tag, '默认')


def save_work_type(data):
    try:
        datas = data.split(',')
    except:
        datas = ['', ]
    new_datas = ['未知', ]
    for the_data in datas:
        if the_data != '-1' and the_data:
            new_datas.append(the_data)
    return new_datas[-1]


def drop_channel_null(data):
    try:
        datas = data.split(',')
    except:
        datas = ['', ]
    new_datas = []
    for the_data in datas:
        if the_data != '-1' and the_data:
            new_datas.append(the_data)
        else:
            new_datas.append('暂未归类')
    return ','.join(new_datas)


def drop_type_null(data):
    try:
        datas = data.split(',')
    except:
        datas = ['', ]
    new_datas = []
    for the_data in datas:
        if the_data != '-1' and the_data and the_data != '[]':
            new_datas.append(the_data)
        else:
            new_datas.append('空')
    return ','.join(new_datas)


all_labels = set()


def drop_label_null(data):
    try:
        datas = data.split(',')
    except:
        datas = ['', ]
    new_datas = []
    for the_data in datas:
        if the_data != '-1' and the_data:
            new_datas.append(the_data)
            all_labels.add(the_data)
        else:
            new_datas.append('无')
            all_labels.add('无')
    return ','.join(new_datas)


all_moment = set()


def drop_moment_null(data):
    try:
        datas = data.split(',')
    except:
        datas = ['', ]
    new_datas = []
    for the_data in datas:
        if the_data != '-1' and the_data:
            new_datas.append(the_data)
            moment_words = segment.cut(the_data)
            _words = [word for word in moment_words if not segment.is_number(word)]
            all_moment.update(_words)
        else:
            new_datas.append('无')
            all_moment.add('无')
    return ','.join(new_datas)


def change_sex_to_num(sex):
    return sex_map_train.get(sex, 0)


class PredictGender(object):
    def __init__(self):
        if os.path.exists(os.path.join(dir_path, "class_model", 'model.txt')):
            print('正在加载过去模型，若想更新模型，请删除旧模型并重新启动。')
            try:
                with open(os.path.join(dir_path, "class_model", 'key_word2vec.pkl'), 'rb') as f:
                    self.word2vec_dict = pickle.load(f)
            except:
                print('词向量丢失，请删除所有模型，重新启动！')
                sys.exit()

            try:
                with open(os.path.join(dir_path, "class_model", 'minmax.pkl'), 'rb') as f:
                    self.minmax = pickle.load(f)
            except:
                print('归一化数据丢失，请删除所有模型，重新启动！')
                sys.exit()

            try:
                # 行业 OneHotEncoder
                self.work_type_enc = joblib.load(os.path.join(dir_path, "class_model", 'work_type_enc.pkl'))
            except:
                print('行业 OneHotEncoder丢失，请删除所有模型，重新启动！')
                sys.exit()

            try:
                # 文章频道 OneHotEncoder
                self.channel_enc = joblib.load(os.path.join(dir_path, "class_model", 'channel_enc.pkl'))
            except:
                print('文章频道 OneHotEncoder丢失，请删除所有模型，重新启动！')
                sys.exit()

            try:
                # 文章类型 OneHotEncoder
                self.type_enc = joblib.load(os.path.join(dir_path, "class_model", 'type_enc.pkl'))
            except:
                print('文章类型 OneHotEncoder丢失，请删除所有模型，重新启动！')
                sys.exit()

        else:
            self.read_data()

    def read_data(self):
        """将符合的数据读入"""
        data_path = os.path.join(dir_path, "data/")
        file_list = os.listdir(data_path)

        # 用户文件名
        user_files = []
        # 文章文件名
        article_files = []
        for file_name in file_list:
            if 'user_' in file_name:
                user_files.append(file_name)
            elif 'part-' in file_name:
                article_files.append(file_name)
            else:
                pass
        user_files.sort()
        article_files.sort()
        all_file_data = []
        for file_name in user_files:
            print('读取用户文件：',file_name)
            all_file_data.append(pd.read_csv(os.path.join(dir_path, "data", file_name), sep='\001', header=None))
        user_table = pd.concat(all_file_data, axis=0, ignore_index=True)
        del all_file_data
        user_table.columns = [USER_ID, USER_NAME, USER_GENDER, USER_BIRTHDAY, USER_COMPANY, USER_POSITION,
                              USER_TRADE, USER_WEIXIN_NAME, USER_WEIXIN_GENDER, USER_PROVINCE, USER_CITY,
                              USER_VIP, USER_VIP_RANK, CONCERN_ACTOR, ACTOR_TYPE, CONCERN_ARTICLE_TYPE]
        # 获得指定列
        user_sex_useful_data = user_table[[USER_ID, USER_GENDER, USER_TRADE]]
        del user_table

        all_file_data = []
        for file_name in article_files:
            print('读取文章文件：',file_name)
            all_file_data.append(pd.read_csv(os.path.join(dir_path, "data", file_name), sep='\001', header=None))
        article_table = pd.concat(all_file_data, axis=0, ignore_index=True)
        del all_file_data
        article_table.columns = [USER_ID, ARTICLE_ID, ARTICLE_ELITE, BROWSE_TIME, BROWSE_DURATION, ARTICLE_CHANNEL,
                                 ARTICLE_TYPE, ARTICLE_LABEL, ARTICLE_PIC_NUM, ARTICLE_WORD_NUM,
                                 USER_NEW_PROVINCE, USER_NEW_CITY, MOMENT_CONTENT, PAY_NUM, EQUIPMENT_TYPE,
                                 NETWORK_TYPE]
        # 获得指定列
        article_sex_useful_data = article_table[[USER_ID, ARTICLE_CHANNEL, ARTICLE_TYPE, ARTICLE_LABEL, MOMENT_CONTENT]]
        del article_table
        user_sex_useful_data.reset_index(inplace=True, drop=True)
        article_sex_useful_data.reset_index(inplace=True, drop=True)
        user_sex_useful_data[USER_GENDER] = user_sex_useful_data[USER_GENDER].map(save_sex_data)
        user_sex_useful_data[USER_GENDER] = user_sex_useful_data[USER_GENDER].map(change_sex_map_data)
        user_sex_useful_data[USER_TRADE] = user_sex_useful_data[USER_TRADE].map(save_work_type)
        article_sex_useful_data[ARTICLE_CHANNEL] = article_sex_useful_data[ARTICLE_CHANNEL].map(drop_channel_null)
        article_sex_useful_data[ARTICLE_TYPE] = article_sex_useful_data[ARTICLE_TYPE].map(drop_type_null)
        article_sex_useful_data[ARTICLE_LABEL] = article_sex_useful_data[ARTICLE_LABEL].map(drop_label_null)
        article_sex_useful_data[MOMENT_CONTENT] = article_sex_useful_data[MOMENT_CONTENT].map(drop_moment_null)

        # 提取性别分析所需用户信息
        train_user = user_sex_useful_data[user_sex_useful_data[USER_GENDER] != '默认']
        train_user = train_user[[USER_ID, USER_GENDER, USER_TRADE]]
        train_user.reset_index(inplace=True, drop=True)

        work_types = set()
        for i in train_user.index:
            work_type = train_user.loc[i, USER_TRADE]
            work_types.add(work_type)
        work_types = list(work_types)

        self.work_type_enc = OneHotEncoder()
        joblib.dump(self.work_type_enc, os.path.join(dir_path, "class_model", 'work_type_enc.pkl'))
        self.work_type_enc.fit(np.array(work_types).reshape(-1, 1))
        work_type_np_data = np.zeros((len(train_user), len(work_types)))

        for i in train_user.index:
            work_type = train_user.loc[i, USER_TRADE]
            work_type_np_data[i] = self.work_type_enc.transform([[work_type]]).toarray()[0]

        train_user = train_user[[USER_ID, USER_GENDER]]
        train_user = pd.concat([train_user, pd.DataFrame(work_type_np_data)], axis=1, ignore_index=True)
        train_user.columns = [USER_ID, USER_GENDER] + [USER_TRADE + '_{}'.format(i) for i in range(len(work_types))]
        print('获取keyword字典')
        # 提取性别分析所需文章信息
        if os.path.exists(os.path.join(dir_path, "class_model", 'key_word2vec.pkl')):
            with open(os.path.join(dir_path, "class_model", 'key_word2vec.pkl'), 'rb') as f:
                self.word2vec_dict = pickle.load(f)
        else:
            self.word2vec_dict = dict()
            for key_word in list(all_labels) + list(all_moment):
                self.word2vec_dict[key_word] = np.array(np.array(doc_vector.avg_vector(words=[key_word, ])) * 1000000,
                                                        dtype=np.int32)
            # 保存word2vec字典
            with open(os.path.join(dir_path, "class_model", 'key_word2vec.pkl'), 'wb') as f:
                pickle.dump(self.word2vec_dict, f)

        channel_names = set()
        arcitle_type = set()
        for i in article_sex_useful_data.index:
            data_channel = article_sex_useful_data.loc[i, ARTICLE_CHANNEL].split(',')
            data_type = article_sex_useful_data.loc[i, ARTICLE_TYPE].split(',')
            for the_channel in data_channel:
                if the_channel:
                    channel_names.add(the_channel)
            for the_type in data_type:
                if the_type:
                    arcitle_type.add(the_type)
        channel_names = list(channel_names)
        arcitle_type = list(arcitle_type)
        print('获得onehot')
        self.channel_enc = OneHotEncoder()
        self.channel_enc.fit(np.array(channel_names).reshape(-1, 1))
        joblib.dump(self.channel_enc, os.path.join(dir_path, "class_model", 'channel_enc.pkl'))
        self.type_enc = OneHotEncoder()
        self.type_enc.fit(np.array(arcitle_type).reshape(-1, 1))
        joblib.dump(self.type_enc, os.path.join(dir_path, "class_model", 'type_enc.pkl'))

        # article_sex_useful_data.to_csv(os.path.join(dir_path, "data", 'article_sex_useful_data.csv'),index=False)
        # train_user.to_csv(os.path.join(dir_path, "data", 'train_user.csv'),index=False)
        # del article_sex_useful_data
        # del train_user
        # gc.collect()
        # print('删除数据')
        #
        # article_sex_useful_data = pd.read_csv(os.path.join(dir_path, "data", 'article_sex_useful_data.csv'))
        # train_user = pd.read_csv(os.path.join(dir_path, "data", 'train_user.csv'))
        all_np_data = np.zeros((len(article_sex_useful_data), len(channel_names) + len(arcitle_type) + 2))
        for i in article_sex_useful_data.index:
            data_channel = article_sex_useful_data.loc[i, ARTICLE_CHANNEL].split(',')
            data_type = article_sex_useful_data.loc[i, ARTICLE_TYPE].split(',')
            data_label = article_sex_useful_data.loc[i, ARTICLE_LABEL].split(',')
            data_moment = article_sex_useful_data.loc[i, MOMENT_CONTENT].split(',')
            num_channel = np.zeros((len(channel_names)))
            num_type = np.zeros((len(arcitle_type)))
            num_label = np.zeros((200))
            num_moment = np.zeros((200))
            for the_channel in data_channel:
                num_channel += (self.channel_enc.transform([[the_channel]]).toarray()[0])
            for the_type in data_type:
                num_type += (self.type_enc.transform([[the_type]]).toarray()[0])
            for the_lable in data_label:
                num_label += np.array(self.word2vec_dict.get(the_lable, [0] * 200))
            for the_moment in data_moment:
                num_moment += np.array(self.word2vec_dict.get(the_moment, [0] * 200))

            num_label = num_label.astype(np.int).astype(np.str).tolist()
            num_label = ','.join(num_label)
            num_moment = num_moment.astype(np.int).astype(np.str).tolist()
            num_moment = ','.join(num_moment)
            all_np_data[i] = np.hstack(
                (num_channel, num_type, np.array([hash(num_label)]), np.array([hash(num_moment)])))
        train_article = pd.DataFrame(article_sex_useful_data[USER_ID].tolist())
        train_article = pd.concat([train_article, pd.DataFrame(all_np_data)], axis=1, ignore_index=True)
        train_article.columns = [USER_ID] + [ARTICLE_CHANNEL + '_{}'.format(i) for i in range(len(channel_names))] + \
                                [ARTICLE_TYPE + '_{}'.format(i) for i in range(len(arcitle_type))] + [ARTICLE_LABEL] + [
                                    MOMENT_CONTENT]
        self.minmax = {
            "max_label": train_article[ARTICLE_LABEL].max(),
            "min_label": train_article[ARTICLE_LABEL].min(),
            "max_content": train_article[MOMENT_CONTENT].max(),
            "min_content": train_article[MOMENT_CONTENT].min(),
        }
        print('计算归一化')
        with open(os.path.join(dir_path, "class_model", 'minmax.pkl'), 'wb') as f:
            pickle.dump(self.minmax, f)
        train_article[ARTICLE_LABEL] = train_article[ARTICLE_LABEL].map(
            lambda x: x / (self.minmax["max_label"] - self.minmax["min_label"]))
        train_article[MOMENT_CONTENT] = train_article[MOMENT_CONTENT].map(
            lambda x: x / (self.minmax["max_content"] - self.minmax["min_content"]))

        self.train_data = pd.merge(train_user, train_article, how='inner', on=USER_ID)
        self.all_data = pd.merge(train_user, train_article, how='outer', on=USER_ID)
        self.train_data[USER_GENDER] = self.train_data[USER_GENDER].map(change_sex_to_num)
        self.all_data[USER_GENDER] = self.all_data[USER_GENDER].map(change_sex_to_num)
        self.all_data.to_csv(os.path.join(dir_path, "data", '全量数据.csv'),index=False)
        self.train_data.to_csv(os.path.join(dir_path, "data", '训练数据.csv'),index=False)


    def train_model(self):

        self.all_data = pd.read_csv(os.path.join(dir_path, "data", '全量数据.csv'))
        self.train_data = pd.read_csv(os.path.join(dir_path, "data", '训练数据.csv'))
        train_columns = list(self.train_data.columns)
        train_columns.remove(USER_GENDER)
        X = self.train_data[train_columns]
        Y = self.train_data[USER_GENDER]
        self.user_ids = np.array(X)[:, :1]
        self.X = np.array(X)[:, 1:]
        self.Y = np.array(Y)

    def lbg_train(self):
        self.train_model()
        maxIter = 2500
        params = {
            "objective": "binary",
            "learning_rate": 0.01,
            "default": "gpu",
        }
        trainX, valX, trainy, valy = train_test_split(self.X, self.Y, test_size=0.2, random_state=1)
        trainDataSet = lgb.Dataset(trainX, label=trainy)
        valDataSet = lgb.Dataset(valX, label=valy)
        evaluation = {}
        model = lgb.train(params, trainDataSet, maxIter, valid_sets=[valDataSet],
                          early_stopping_rounds=100, verbose_eval=20, evals_result=evaluation,
                          )
        model.save_model(os.path.join(dir_path, "class_model", 'model.txt'))

    def get_result(self, uid):
        if not os.path.exists(os.path.join(dir_path, "class_model", 'model.txt')):
            self.lbg_train()
        self.all_data = pd.read_csv(os.path.join(dir_path, "data", '全量数据.csv'))
        self.train_data = pd.read_csv(os.path.join(dir_path, "data", '训练数据.csv'))
        model = lgb.Booster(model_file=os.path.join(dir_path, "class_model", 'model.txt'))

        all_user_id = self.all_data[USER_ID].tolist()

        if uid in all_user_id:
            the_test_data = self.all_data[self.all_data[USER_ID] == uid]
            the_test_data = np.array(the_test_data)[:, 2:]
            predY = model.predict(the_test_data, num_iteration=model.best_iteration)
            predY[predY < THRESHOLD] = 0
            predY[predY >= THRESHOLD] = 1
            predY = int(predY[0])
            if predY == 1:
                return '男'
            else:
                return '女'
        else:
            return "查无此人！！！"


if __name__ == '__main__':
    init_configure_utils(model="development")
    segment = SegmentService()
    doc_vector = DocVectorClient()
    a = PredictGender()

    res1 = a.get_result('00092b0f004c6602d2661be8a9145d2b1ab06975')
    res2 = a.get_result('020FB7E2625C0B2971403827FEFCC1DF')
    res3 = a.get_result('113ad4bb864f99096301e2d95aab3a42f5a3902e')
    res4 = a.get_result('1afcd018d45221053d97ccb19ed4288116f59844')

    print(res1)
    print(res2)
    print(res3)
    print(res4)
