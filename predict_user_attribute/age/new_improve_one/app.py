# 导入Flask类

import json
import time
import os
import logging
import sys
import collections
import configparser
from flask import Flask, jsonify, request
import LAC
import gensim
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.utils
import sklearn.model_selection
#import happybase
import joblib
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all" 
from sklearn.cluster import KMeans 
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from keras.models import load_model

APP_DIR=os.path.dirname(os.path.abspath('__file__'))
sys.path.append(APP_DIR)

config_ini_dict=configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR,"config.ini"))
logging.info(config_ini_dict)
# 实例化，可视为固定格式
app =Flask(__name__)

def pre_chuli(data_list_list,type_list):
    title_tokenize_list = []
    #age_list = []
    for i in range(len(data_list_list)):
     #   age = temp_function(int(df.loc[index,"age"]))
        temp_text_list = []
        for j in range(10):
            c_text_list = [data_list_list[i][j]["title"],"<PADDING>",data_list_list[i][j]["content"][:512]]
            temp_text_list.append(c_text_list)     
        title_tokenize_list.append(temp_text_list)
    with open(config_ini_dict["file"]["data_jsonl"],"w",encoding='UTF-8') as f:
        for content_tokenize in zip(title_tokenize_list):
            f.write(json.dumps({"all_content_tokenize":content_tokenize},ensure_ascii=False) + "\n")
    return type_list
APP_DIR  = os.path.dirname(os.path.realpath('__file__'))

def trans_gensim_word2vec2tf_embedding(word2vector_file_path:str):
    """把gensim的word2vec结果转化为tf.keras.layers.Embedding需要的结果
    """

    word2vec_model = gensim.models.Word2Vec.load(word2vector_file_path)

    #所有的词
    word_list = [word for word, word_info in word2vec_model.wv.vocab.items()]

    #词到index的映射
    word2index_dict = {"<PADDING>": 0, "<UNK>":1}

    #保存特殊词的padding
    specical_word_count = len(word2index_dict)

    #词到词向量的映射
    word2vector_dict = {}

    #初始化embeddings_matrix

    embeddings_matrix = np.zeros((len(word_list) + specical_word_count, word2vec_model.vector_size))
    #初始化unk为-1,1分布
    embeddings_matrix[word2index_dict["<UNK>"]] = (1 / np.sqrt(len(word_list) + specical_word_count) * (2 * np.random.rand(word2vec_model.vector_size) - 1))

    for i,word in enumerate(word_list):
        #从0开始
        word_index = i + specical_word_count
        word2index_dict[str(word)] = word_index
        word2vector_dict[str(word)] = word2vec_model.wv[word] # 词语：词向量
        embeddings_matrix[word_index] = word2vec_model.wv[word]  # 词向量矩阵
    print()

    #写入文件
    with open(os.path.join(APP_DIR,"data","word2index.json"),"w",encoding="utf8") as f:
        json.dump(word2index_dict,f,ensure_ascii=False)

    return embeddings_matrix,word2vector_dict,word2index_dict
def trans2index(word2index_dict,word):
    """转换"""
    if word in word2index_dict:
        return word2index_dict[word]
    else:
        if "<UNK>" in word2index_dict:
            return word2index_dict["<UNK>"]
        else:
            raise ValueError("没有这个值，请检查")


def trans_data2tf_data(data_file_path:str,x_max_length:int=None,word2index_dict=None):
    """把data文件转化为tf.data
    """

   # tag2index_dict = {}
   # tag_index_count = len(tag2index_dict)
    lac = LAC.LAC(mode="seg")

    df = pd.read_csv(data_file_path)

    x_list = []
    for doc in df["content"]:
        word_list = lac.run(doc)
        x_list.append([trans2index(word2index_dict,word) for word in word_list])
    x_npa = np.array(x_list)


    if not x_max_length:
        x_max_length0 = np.max(np.array([len(v) for v in x_list]))
        x_max_length = int(np.max(np.percentile(np.array([len(v) for v in x_list]),99.7)))
        print("数据集中最长的句子长度为:{},设定的最长的句子长度为:{}".format(x_max_length0,x_max_length))

    x_npa = tf.keras.preprocessing.sequence.pad_sequences(x_npa,maxlen=x_max_length,dtype=np.int32,truncating="post", padding='post',value=0)

    return x_npa
def trans_tokenize_data2tf_data(data_file_path:str,x_max_length:int=None,word2index_dict=None):
    """把已经分好词的data文件转化为tf.data
    """

   # tag2index_dict = {}
#    tag_index_count = len(tag2index_dict)
    lac = LAC.LAC(mode="seg")

    x_list = []
 #   y_list = []
    with open(data_file_path) as f:
        for line in f:
            temp_dict = json.loads(line.strip())
            word_list = temp_dict["content_tokenize"]
            x_list.append([trans2index(word2index_dict,word) for word in word_list])
    x_npa = np.array(x_list)

#    print("x_list[:1]:{}".format(x_list[:1]))

    if not x_max_length:
        x_max_length0 = np.max(np.array([len(v) for v in x_list]))
        x_max_length = int(np.max(np.percentile(np.array([len(v) for v in x_list]),99.7)))
        print("数据集中最长的句子长度为:{},设定的最长的句子长度为:{}".format(x_max_length0,x_max_length))

    x_npa = tf.keras.preprocessing.sequence.pad_sequences(x_npa,maxlen=x_max_length,dtype=np.int32,truncating="post", padding='post',value=0)
 #   print("x_npa[:1]:{}".format(x_npa[:1]))
  #  print("x_npa.shape = {}".format(x_npa.shape))
    return x_npa


def trans_multi_input_tokenize_data2npa(data_file_path:str,x_max_length:int=None,word2index_dict=None):
    """把已经分好词的data文件转化为tf.data , 多输入版本
    """
    x_list = []
    with open(data_file_path) as f:
        for line in f:
            temp_dict = json.loads(line.strip())
            text_tokenize_list = temp_dict["all_content_tokenize"]
            text_tokenize_list = sum(text_tokenize_list,[])
            x_list.append([[trans2index(word2index_dict,word) for word in word_list] for word_list in text_tokenize_list])

  #  print("x_list[:1]:{}".format(x_list[:1]))

    if not x_max_length:
        x_max_length0 = np.max(np.array([len(v) for v in x_list]))
        x_max_length = int(np.max(np.percentile(np.array([len(v) for v in x_list]),99.7)))
    #    print("数据集中最长的句子长度为:{},设定的最长的句子长度为:{}".format(x_max_length0,x_max_length))
    
    for i in range(len(x_list)):
        x_list[i] = tf.keras.preprocessing.sequence.pad_sequences(x_list[i],maxlen=x_max_length,dtype=np.int32,truncating="post", padding='post',value=0)
    x_npa = np.array(x_list,dtype=np.int32)
#    x_npa_position = x_npa

    return x_npa

APP_DIR  = os.path.dirname(os.path.realpath('__file__'))
if not os.path.exists(os.path.join(APP_DIR,"data")):
    os.makedirs(os.path.join(APP_DIR,"data"))

def split_train_eval_test_dataset(dataset):
    """区分训练验证测试集
    """
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
  #  print("总共有数据{}条".format(dataset_size))
    dataset = dataset.shuffle(dataset_size,seed=1)
    train_size = int(0.6 * dataset_size)
    eval_size = int(0.2 * dataset_size)
    test_size = int(0.2 * dataset_size)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    eval_dataset = test_dataset.skip(eval_size)
    test_dataset = test_dataset.take(test_size)
    return train_dataset.prefetch(tf.data.experimental.AUTOTUNE), \
        eval_dataset.prefetch(tf.data.experimental.AUTOTUNE), \
        test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

def split_train_eval_test_npa(x_npa,y_npa):
    x_train,x_test,y_train,y_test =  sklearn.model_selection.train_test_split(x_npa, y_npa, test_size=0.2, random_state=24)
    return x_train,y_train,x_test,y_test

def data_mining(x_npa):       
    x_npa = x_npa.reshape(len(x_npa),5120)
    x_npa = pd.DataFrame(x_npa)
    x_npa.columns = [*["c{}".format(v) for v in range(5120)]]
    age_data = x_npa.T
    df_22 = age_data[0:999]
 #   rank0_1000 = pd.read_csv('../data/predict_user_attribute20200911/raw_data/user_profile/age/rank0_1000.csv')
    rank_temp = config_ini_dict["file"]["rank0_1000"]
    rank_temp = rank_temp.split(",")
    rank0_1000 = []
    for i in rank_temp:
        b = int(i)
        rank0_1000.append(b)
    rank0_1000 = pd.DataFrame(rank0_1000)
    rank0_1000.columns = ['rank']
    df0_1000 = pd.merge(rank0_1000,df_22,on = rank0_1000.index)
    df0_1000 = df0_1000.drop('key_0',axis = 1)
    df0_1000 = df0_1000.groupby(df0_1000.index).filter(lambda x:float(x['rank'])==1)
    #df_22 = df_22.T
    df_22 = age_data[999:1999]
#    rank1000_2000 = pd.read_csv('../data/predict_user_attribute20200911/raw_data/user_profile/age/rank1000_2000.csv')
    rank_temp1 = config_ini_dict["file"]["rank1000_2000"]
    rank_temp1 = rank_temp1.split(",")
    rank1000_2000 = []
    for i in rank_temp1:
        b = int(i)
        rank1000_2000.append(b)
    rank1000_2000 = pd.DataFrame(rank1000_2000)
    rank1000_2000.columns = ['rank']
    df1000_2000 = pd.merge(rank1000_2000,df_22,on = rank1000_2000.index)
    df1000_2000 = df1000_2000.drop('key_0',axis = 1)
    df1000_2000 = df1000_2000.groupby(df1000_2000.index).filter(lambda x:float(x['rank'])==1)
    df_22 = age_data[1999:2999]
    #df_22 = df_22.T
#    rank2000_3000 = pd.read_csv('../data/predict_user_attribute20200911/raw_data/user_profile/age/rank2000_3000.csv')
#    rank2000_3000 = rank2000_3000.drop('Unnamed: 0',axis = 1)
    rank_temp2 = config_ini_dict["file"]["rank2000_3000"]
    rank_temp2 = rank_temp2.split(",")
    rank2000_3000 = []
    for i in rank_temp1:
        b = int(i)
        rank2000_3000.append(b)
    rank2000_3000 = pd.DataFrame(rank2000_3000)
    rank2000_3000.columns = ['rank']
    df2000_3000 = pd.merge(rank2000_3000,df_22,on = rank2000_3000.index)
    df2000_3000 = df2000_3000.drop('key_0',axis = 1)
    df2000_3000 = df2000_3000.groupby(df2000_3000.index).filter(lambda x:float(x['rank'])==1)
    df_22 = age_data[2998:5121]
    #df_22 = df_22.T
#    rank3000_5122 = pd.read_csv('../data/predict_user_attribute20200911/raw_data/user_profile/age/rank3000_5122.csv')
#    rank3000_5122 = rank3000_5122.drop('Unnamed: 0',axis = 1)
    rank_temp3 = config_ini_dict["file"]["rank3000_5122"]
    rank_temp3 = rank_temp3.split(",")
    rank3000_5122 = []
    for i in rank_temp3:
        b = int(i)
        rank3000_5122.append(b)
    rank3000_5122 = pd.DataFrame(rank3000_5122)
    rank3000_5122.columns = ['rank']
    df3000_5122 = pd.merge(rank3000_5122,df_22,on = rank3000_5122.index)
    df3000_5122 = df3000_5122.drop('key_0',axis = 1)
    df3000_5122 = df3000_5122.groupby(df3000_5122.index).filter(lambda x:float(x['rank'])==1)
    x_drop_fea1 = pd.concat([df0_1000, df1000_2000], axis=0, ignore_index=True)
    x_drop_fea1 = pd.concat([x_drop_fea1, df2000_3000], axis=0, ignore_index=True)
    x_drop_fea1 = pd.concat([x_drop_fea1, df3000_5122], axis=0, ignore_index=True)
    x_drop_fea1 = x_drop_fea1.drop('rank',axis = 1)
#    rank_100 = pd.read_csv('../data/predict_user_attribute20200911/raw_data/user_profile/age/rank_100.csv')
#    rank_100 = rank_100.drop('Unnamed: 0',axis = 1)
    rank_temp4 = config_ini_dict["file"]["rank_100"]
    rank_temp4 = rank_temp4.split(",")
    rank_100 = []
    for i in rank_temp4:
        b = int(i)
        rank_100.append(b)
    rank_100 = pd.DataFrame(rank_100)
    rank_100.columns = ['rank']
    x_drop_100 = pd.merge(rank_100,x_drop_fea1,on = rank_100.index)
    x_drop_100 = x_drop_100.drop('key_0',axis = 1)
    x_drop_100 = x_drop_100.groupby(x_drop_100.index).filter(lambda x:float(x['rank'])==1)
    x_drop_100 = x_drop_100.drop('rank',axis = 1)
    x_drop_100 = x_drop_100.T
    ss = StandardScaler()
    x_drop1_100 = ss.fit_transform(x_drop_100)
    x_drop1_100 = pd.DataFrame(x_drop1_100)
    return x_drop1_100

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# route()方法用于设定路由；类似spring路由配置
@app.route('/test_1/',methods=['post','get'])
def predict():
    
    if request.method == 'POST':
        start = time.time()
        content_list = request.json['data_list_list']
        type_list = request.json['type_list']
      #  print(content_id_list)
        pre_chuli(content_list,type_list)
        result = {"result": []}
        
        d = dict()
        for type_1 in type_list:
            if type_1 == "age":
                pre_age = model_age.predict(x_drop1_100)
                d['age'] = pre_age.tolist()
            elif type_1 == "position":
                pre_position = model_position.predict(x_drop1_100)
                d['position'] = pre_position.tolist()
            elif type_1 == 'trade':
                pre_trade = model_trade.predict(x_drop1_100)
                d['trade'] = pre_trade.tolist()
            else:
                pre_gender = model_gender.predict(x_drop1_100)
                d['gender'] = pre_trade.tolist()
       # pre = model.predict(x_drop1_100)
        result["result"].append(d)
     #   print(result)
       # pre = pre.tolist()
        dic_age = {0:'36~45',1:'30~35',2:'26~29',
               3:'~20',4:'46~50',5:'21~25',6:'50~'}
        dic_position = {0:'企业业务决策层',1:'企业普通员工',2:'企业CEO',3:'企业一般管理人员',
                        4:'其他',5:'在校学生',6:'事业单位员工'}
        dic_trade = {0:'金融',
                     1:'IT/移动互联网',
                    2:'其他',
                     3:'制造业',
                     4:'房地产',
                     5:'零售消费',
                     6:'汽车',
                     7:'电商',
                     8:'IOT',
                     9:'游戏',
                     10:'医疗'}
        dic_gender = {0:'男',
                      1:'女',
                      2:'aaaf'}
        out_list = []
        for type_2 in type_list:
            if type_2 == 'age':
                for i in result["result"][0]['age']:
                    out_list.append(dic_age[i])
            if type_2 == 'position':
                for i in result['result'][0]['position']:
                    out_list.append(dic_position[i])
            if type_2 == 'trade':
                for i in result['result'][0]['trade']:
                    out_list.append(dic_trade[i])
            if type_2 == 'gender':
                for i in result['result'][0]['gender']:
                    out_list.append(dic_gender[i])
            end = time.time()
        print("time: {:.2f} s".format(end - start))
        return jsonify({"result" : out_list})
   
    
if __name__ == '__main__':
    filepath_age = config_ini_dict["file"]["filepath_age"]
    #filepath_position = "../code/model_position"
    data_jsonl = config_ini_dict["file"]["data_jsonl"]
    word2vector_file_path = config_ini_dict["file"]["word2vector_file_path"]
    model_age = joblib.load(filepath_age)
#    tf.keras.models.Model
    model_position = joblib.load(filepath_age)
    model_trade = joblib.load(filepath_age)
    model_gender = joblib.load(filepath_age)
    input_number = 10
    sentence_maxlen = 512        
    embedding_matrix,word2vector_dict,word2index_dict = trans_gensim_word2vec2tf_embedding(word2vector_file_path)
    vocab_size,embedding_dim = embedding_matrix.shape
    x_npa = trans_multi_input_tokenize_data2npa(data_jsonl,sentence_maxlen,word2index_dict) 
    x_npa_position = x_npa
    x_drop1_100 = data_mining(x_npa)  
    # app.run(host, port, debug, options)
    # 默认值：host="127.0.0.1", port=5000, debug=False
    app.run(host="0.0.0.0", port=8888)    
    