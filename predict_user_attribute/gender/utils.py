import json
import gensim
import os
import LAC
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.utils

APP_DIR  = os.path.dirname(os.path.realpath(__file__))

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

    #写入文件
    with open(os.path.join(APP_DIR,"data/word2index.json"),"w",encoding="utf8") as f:
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

    tag2index_dict = {}
    tag_index_count = len(tag2index_dict)
    lac = LAC.LAC(mode="seg")

    df = pd.read_csv(data_file_path)

    x_list = []
    for doc in df["content"]:
        word_list = lac.run(doc)
        x_list.append([trans2index(word2index_dict,word) for word in word_list])
    x_npa = np.array(x_list)

    y_list = []
    for tag in df["tag"]:
        tag = tag.strip()
        if not (tag in tag2index_dict):
            tag2index_dict[tag] = tag_index_count
            tag_index_count += 1
        y_list.append(tag2index_dict[tag])
    y_npa = np.array(y_list,dtype=np.uint8)

    print("x_list[:1]:{}".format(x_list[:1]))
    print("y_list[:1]:{}".format(y_list[:1]))

    #写入文件
    with open(os.path.join(APP_DIR,"data/tag2index.json"),"w",encoding="utf8") as f:
        json.dump(tag2index_dict,f,ensure_ascii=False)

    if not x_max_length:
        x_max_length0 = np.max(np.array([len(v) for v in x_list]))
        x_max_length = int(np.max(np.percentile(np.array([len(v) for v in x_list]),99.7)))
        print("数据集中最长的句子长度为:{},设定的最长的句子长度为:{}".format(x_max_length0,x_max_length))

    x_npa = tf.keras.preprocessing.sequence.pad_sequences(x_npa,maxlen=x_max_length,dtype=np.int32,truncating="post", padding='post',value=0)

    x_npa,y_npa = sklearn.utils.shuffle(x_npa,y_npa,random_state=0)
    print("x_npa[:1]:{}".format(x_npa[:1]))
    print("y_npa[:1]:{}".format(y_npa[:1]))
    print("x_npa.shape = {}".format(x_npa.shape))
    print("y_npa.shape = {}".format(y_npa.shape))

    return x_npa,y_npa,tag2index_dict



def trans_tokenize_data2tf_data(data_file_path:str,x_max_length:int=None,word2index_dict=None):
    """把已经分好词的data文件转化为tf.data
    """

    tag2index_dict = {}
    tag_index_count = len(tag2index_dict)
    lac = LAC.LAC(mode="seg")

    x_list = []
    y_list = []
    with open(data_file_path) as f:
        for line in f:
            temp_dict = json.loads(line.strip())
            word_list = temp_dict["content_tokenize"]
            tag = temp_dict["tag"].strip()
            if not (tag in tag2index_dict):
                tag2index_dict[tag] = tag_index_count
                tag_index_count += 1
            x_list.append([trans2index(word2index_dict,word) for word in word_list])
            y_list.append(tag2index_dict[tag])
    x_npa = np.array(x_list)
    y_npa = np.array(y_list,dtype=np.uint8)

    print("x_list[:1]:{}".format(x_list[:1]))
    print("y_list[:1]:{}".format(y_list[:1]))

    #写入文件
    with open(os.path.join(APP_DIR,"data/tag2index.json"),"w",encoding="utf8") as f:
        json.dump(tag2index_dict,f,ensure_ascii=False)

    if not x_max_length:
        x_max_length0 = np.max(np.array([len(v) for v in x_list]))
        x_max_length = int(np.max(np.percentile(np.array([len(v) for v in x_list]),99.7)))
        print("数据集中最长的句子长度为:{},设定的最长的句子长度为:{}".format(x_max_length0,x_max_length))

    x_npa = tf.keras.preprocessing.sequence.pad_sequences(x_npa,maxlen=x_max_length,dtype=np.int32,truncating="post", padding='post',value=0)

    x_npa,y_npa = sklearn.utils.shuffle(x_npa,y_npa,random_state=0)
    print("x_npa[:1]:{}".format(x_npa[:1]))
    print("y_npa[:1]:{}".format(y_npa[:1]))
    print("x_npa.shape = {}".format(x_npa.shape))
    print("y_npa.shape = {}".format(y_npa.shape))

    return x_npa,y_npa,tag2index_dict
