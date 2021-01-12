import json
import gensim
import os
import sys
import time
import logging
import configparser
import LAC
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.utils
import sklearn
import collections
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.svm import SVC
from tensorflow.keras.models import load_model

APP_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(APP_DIR)
config_ini_dict = configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR, "config.ini"))
logging.info(config_ini_dict)
APP_DIR = os.path.dirname(os.path.realpath("__file__"))
if not os.path.exists(os.path.join(APP_DIR, "data")):
    os.makedirs(os.path.join(APP_DIR, "data"))


class UserProfileSVMModel(object):
    def __init__(
        self,
        data_jsonl,
        filepath_age,
        filepath_position,
        filepath_trade,
        filepath_gender,
        word2vector_file_path,
        age200,
        gender200,
        trade200,
        input_number,
        sentence_maxlen,
    ):

        self.filepath_age = filepath_age
        self.filepath_position = filepath_position
        self.filepath_trade = filepath_trade
        self.filepath_gender = filepath_gender
        self.data_jsonl = data_jsonl
        self.word2vector_file_path = word2vector_file_path
        # trade5120 = config_ini_dict["file"]["trade5120"]
        # age200 = pd.read_csv('/home/zourui/data/predict_user_attribute20200911/raw_data/age/rank/rank.csv')
        self.age200 = age200
        self.gender200 = gender200
        self.trade200 = trade200
        self.input_number = input_number
        self.sentence_maxlen = sentence_maxlen
        self.load_model()

    def load_model(self):
        self.model_age = joblib.load(self.filepath_age)
        self.model_position = load_model(self.filepath_position)
        self.model_trade = joblib.load(self.filepath_trade)
        self.model_gender = joblib.load(self.filepath_gender)
        (
            self.embedding_matrix,
            self.word2vector_dict,
            self.word2index_dict,
        ) = self.trans_gensim_word2vec2tf_embedding()
        self.vocab_size, self.embedding_dim = self.embedding_matrix.shape

    def input_deal(self, params):
        title_list_list = []
        x_list_list = []
        data_list_list = params.get("data_list_list")
        for i in range(len(data_list_list)):
            temp_list = params["data_list_list"][i]
            title_list = []
            content_list = []
            for j in range(len(temp_list)):
                title_dic = {}
                content_dic = {}
                t_text = temp_list[j]["title"]
                c_text = temp_list[j]["content"]
                lac = LAC.LAC(mode="seg")
                word_title = lac.run(t_text)
                word_content = lac.run(c_text)
                title_dic["title"] = word_title
                content_dic["content"] = word_content
                title_list.append(title_dic)
                content_list.append(content_dic)
            title_list_list.append(title_list)
            x_list_list.append(content_list)
        title_tokenize_list = []
        # age_list = []
        for i in range(len(data_list_list)):
            #   age = temp_function(int(df.loc[index,"age"]))
            temp_text_list = []
            for j in range(10):
                c_text_list = [
                    *title_list_list[i][j]["title"],
                    "<PADDING>",
                    *x_list_list[i][j]["content"][:512],
                ]
                temp_text_list.append(c_text_list)
            title_tokenize_list.append(temp_text_list)
        with open(self.data_jsonl, "w", encoding="UTF-8") as f:
            for content_tokenize in zip(title_tokenize_list):
                f.write(
                    json.dumps(
                        {"all_content_tokenize": content_tokenize}, ensure_ascii=False
                    )
                    + "\n"
                )

    def trans_gensim_word2vec2tf_embedding(self):
        """把gensim的word2vec结果转化为tf.keras.layers.Embedding需要的结果"""

        word2vec_model = gensim.models.Word2Vec.load(self.word2vector_file_path)

        # 所有的词
        word_list = [word for word, word_info in word2vec_model.wv.vocab.items()]

        # 词到index的映射
        word2index_dict = {"<PADDING>": 0, "<UNK>": 1}

        # 保存特殊词的padding
        specical_word_count = len(word2index_dict)

        # 词到词向量的映射
        word2vector_dict = {}

        # 初始化embeddings_matrix

        embeddings_matrix = np.zeros(
            (len(word_list) + specical_word_count, word2vec_model.vector_size)
        )
        # 初始化unk为-1,1分布
        embeddings_matrix[word2index_dict["<UNK>"]] = (
            1
            / np.sqrt(len(word_list) + specical_word_count)
            * (2 * np.random.rand(word2vec_model.vector_size) - 1)
        )

        for i, word in enumerate(word_list):
            # 从0开始
            word_index = i + specical_word_count
            word2index_dict[str(word)] = word_index
            word2vector_dict[str(word)] = word2vec_model.wv[word]  # 词语：词向量
            embeddings_matrix[word_index] = word2vec_model.wv[word]  # 词向量矩阵

        # 写入文件
        with open(
            os.path.join(APP_DIR, "data", "word2index.json"), "w", encoding="utf8"
        ) as f:
            json.dump(word2index_dict, f, ensure_ascii=False)

        return embeddings_matrix, word2vector_dict, word2index_dict

    def trans2index(self, word2index_dict, word):
        """转换"""
        if word in word2index_dict:
            return word2index_dict[word]
        else:
            if "<UNK>" in word2index_dict:
                return word2index_dict["<UNK>"]
            else:
                raise ValueError("没有这个值，请检查")

    def trans_multi_input_tokenize_data2npa(
        self, data_file_path, x_max_length, word2index_dict
    ):
        """把已经分好词的data文件转化为tf.data , 多输入版本"""
        x_list = []
        with open(self.data_jsonl) as f:
            for line in f:
                temp_dict = json.loads(line.strip())
                text_tokenize_list = temp_dict["all_content_tokenize"]
                text_tokenize_list = sum(text_tokenize_list, [])
                x_list.append(
                    [
                        [
                            self.trans2index(self.word2index_dict, word)
                            for word in word_list
                        ]
                        for word_list in text_tokenize_list
                    ]
                )

        #  print("x_list[:1]:{}".format(x_list[:1]))

        if not x_max_length:
            #    x_max_length0 = np.max(np.array([len(v) for v in x_list]))
            x_max_length = int(
                np.max(np.percentile(np.array([len(v) for v in x_list]), 99.7))
            )
        #    print("数据集中最长的句子长度为:{},设定的最长的句子长度为:{}".format(x_max_length0,x_max_length))

        for i in range(len(x_list)):
            x_list[i] = tf.keras.preprocessing.sequence.pad_sequences(
                x_list[i],
                maxlen=x_max_length,
                dtype=np.int32,
                truncating="post",
                padding="post",
                value=0,
            )
        x_npa = np.array(x_list, dtype=np.int32)
        #    x_npa_position = x_npa
        return x_npa

    def data_trade_deal(self, x_npa):
        """trade数据预处理输入5120维输出200维"""
        x_npa = x_npa.reshape(len(x_npa), 5120)
        x_npa = pd.DataFrame(x_npa)
        x_npa1 = x_npa.T
        x_trade = pd.merge(self.trade200, x_npa1, on=self.trade200.index)
        x_trade = x_trade.drop("key_0", axis=1)
        x_trade = x_trade.groupby(x_trade.index).filter(lambda x: float(x["rank"]) == 1)
        x_trade = x_trade.drop("rank", axis=1)
        x_trade = x_trade.T
        ss = StandardScaler()
        x_trade = ss.fit_transform(x_trade)
        x_trade = pd.DataFrame(x_trade)
        return x_trade

    def data_gender_deal(self, x_npa):
        x_npa = x_npa.reshape(len(x_npa), 5120)
        x_gender = pd.DataFrame(x_npa)
        ss = StandardScaler()
        x_ss = ss.fit_transform(x_npa)
        x_ss = pd.DataFrame(x_ss)
        x_gender = x_ss.T
        gender_x = pd.merge(self.gender200, x_gender, on=self.gender200.index)
        gender_x = gender_x.drop("key_0", axis=1)
        gender_x = gender_x.groupby(gender_x.index).filter(
            lambda x: float(x["rank"]) == 1
        )
        gender_x = gender_x.drop("rank", axis=1)
        gender_x = gender_x.T
        return gender_x

    def data_mining(self, x_npa):
        """age数据处理输入5120维输出200维"""
        x_npa = x_npa.reshape(len(x_npa), 5120)
        x_npa = pd.DataFrame(x_npa)
        x_npa.columns = [*["c{}".format(v) for v in range(5120)]]
        ss = StandardScaler()
        x_ss = ss.fit_transform(x_npa)
        x_ss = pd.DataFrame(x_ss)
        x_age = x_ss.T
        age_x = pd.merge(self.age200, x_age, on=self.age200.index)
        age_x = age_x.drop("key_0", axis=1)
        age_x = age_x.groupby(age_x.index).filter(lambda x: float(x["rank"]) == 1)
        age_x = age_x.drop("rank", axis=1)
        age_x = age_x.T
        return age_x

    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """

        def default(self, obj):
            if isinstance(
                obj,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    def predict(self, params):
        start = time.time()
        self.input_deal(params)
        x_npa = self.trans_multi_input_tokenize_data2npa(
            self.data_jsonl, self.sentence_maxlen, self.word2index_dict
        )
        result = {"result": []}
        d = dict()
        type_list = params["type_list"]
        for type_1 in type_list:
            if type_1 == "age":
                x_drop1_100 = self.data_mining(x_npa)
                pre_age = self.model_age.predict(x_drop1_100)
                d["age"] = pre_age.tolist()
            elif type_1 == "position":
                pre = self.model_position.predict(
                    {"input{}".format(i): x_npa[:, i] for i in range(self.input_number)}
                )
                output_position = tf.keras.layers.Dense(
                    7,
                    activation="softmax",
                    use_bias=True,
                    kernel_regularizer=tf.keras.regularizers.l2(),
                )(pre)
                pre_position = np.argmax(output_position, axis=1)
                d["position"] = pre_position.tolist()
            elif type_1 == "trade":
                x_trade = self.data_trade_deal(x_npa)
                pre_trade = self.model_trade.predict(x_trade)
                d["trade"] = pre_trade.tolist()
            else:
                x_gender = self.data_gender_deal(x_npa)
                pre_gender = self.model_gender.predict(x_gender)
                d["gender"] = pre_gender.tolist()
        result["result"].append(d)
        dic_age = {
            0: "36~45",
            1: "30~35",
            2: "26~29",
            3: "~20",
            4: "46~50",
            5: "21~25",
            6: "50~",
        }
        dic_position = {
            0: "企业业务决策层",
            1: "企业普通员工",
            2: "企业CEO",
            3: "企业一般管理人员",
            4: "其他",
            5: "在校学生",
            6: "事业单位员工",
        }
        dic_trade = {
            0: "金融",
            1: "IT/移动互联网",
            2: "其他",
            3: "制造业",
            4: "房地产",
            5: "零售消费",
            6: "汽车",
            7: "电商",
            8: "IOT",
            9: "游戏",
            10: "医疗",
        }
        dic_gender = {
            0: "男",
            1: "女",
        }
        out_list = []
        data_list_list = params.get("data_list_list")
        for i in range(len(data_list_list)):
            dict_all = dict()
            dict_all["user_id"] = i
            dict_all["age"] = dic_age[result["result"][0]["age"][i]]
            dict_all["position"] = dic_position[result["result"][0]["position"][i]]
            dict_all["trade"] = dic_trade[result["result"][0]["trade"][i]]
            dict_all["gender"] = dic_gender[result["result"][0]["gender"][i]]
            out_list.append(dict_all)
        end = time.time()
        print("time: {:.2f} s".format(end - start))
        print({"result": out_list})
        out_put = {"result": out_list}
        return out_put


# params = {
#     "type_list": ["age", "position", "trade", "gender"],
#     "data_list_list": [
#         [
#             {
#                 "id": "349684",
#                 "title": "权限申请",
#                 "content": "编辑，运营，虎嗅PRO，市场部的同学需要开通虎嗅后台权限",
#             },
#             {"id": "349389", "title": "及时申请", "content": "请及时和你的上级领导沟通开通权限"},
#             {
#                 "id": "349248",
#                 "title": "离职流程",
#                 "content": "至少提前三天告知部门领导及对应HRBP，并在企业微信上提交实习生离职申请",
#             },
#             {"id": "349729", "title": "少量的词汇", "content": "少量的词汇可以自己用下面方法手动添加"},
#             {"id": "347400", "title": "可调节", "content": "可调节单个词语的词频，使其能（或不能）被分出来"},
#             {"id": "344255", "title": "待提取的文本", "content": "仅包括指定词性的词，默认值为空，即不筛选"},
#             {
#                 "id": "340988",
#                 "title": "关键词提取",
#                 "content": "关键词提取所使用逆向文件频率（IDF）文本语料库可以切换成自定义语料库的路径",
#             },
#             {"id": "348764", "title": "关键词提取所使用停止词", "content": "文本语料库可以切换成自定义语料库的路径"},
#             {"id": "3409358", "title": "词性标注", "content": "参数可指定内部使用的的原理后面再来说"},
#             {"id": "3432587", "title": "尽管我们这里不更新", "content": "实践是检验真理的唯一标准"},
#         ],
#         [
#             {"id": "5", "title": "4", "content": "2"},
#             {"id": "32", "title": "98", "content": "90"},
#             {"id": "12", "title": "6436", "content": "2341"},
#             {"id": "5", "title": "4", "content": "2"},
#             {"id": "32", "title": "98", "content": "90"},
#             {"id": "12", "title": "6436", "content": "2341"},
#             {"id": "5", "title": "4", "content": "2"},
#             {"id": "32", "title": "98", "content": "90"},
#             {"id": "388579", "title": "6436", "content": "2341"},
#             {"id": "214451", "title": "6436", "content": "2341"},
#         ],
#     ],
# }
# userprofile = UserProfileSVMModel(
#     filepath_age=config_ini_dict["file"]["filepath_age"],
#     filepath_position=config_ini_dict["file"]["filepath_position"],
#     filepath_trade=config_ini_dict["file"]["filepath_trade"],
#     filepath_gender=config_ini_dict["file"]["filepath_gender"],
#     data_jsonl=config_ini_dict["file"]["data_jsonl"],
#     word2vector_file_path=config_ini_dict["file"]["word2vector_file_path"],
#     age200=pd.read_csv(config_ini_dict["file"]["age_200"]),
#     gender200=pd.read_csv(config_ini_dict["file"]["gender_200"]),
#     trade200=pd.read_csv(config_ini_dict["file"]["trade_200"]),
#     input_number=10,
#     sentence_maxlen=512,
# )
# userprofile.predict(params)
# if __name__ == '__main__':
#     APP_DIR = os.path.dirname(os.path.abspath('__file__'))
#     sys.path.append(APP_DIR)
#     config_ini_dict = configparser.ConfigParser()
#     config_ini_dict.read(os.path.join(APP_DIR, "config.ini"))
#     logging.info(config_ini_dict)
#     filepath_age = config_ini_dict["file"]["filepath_age"]
#     filepath_position = config_ini_dict["file"]["filepath_position"]
#     filepath_trade = config_ini_dict['file']["filepath_trade"]
#     filepath_gender = config_ini_dict['file']["filepath_gender"]
#     data_jsonl = config_ini_dict["file"]["data_jsonl"]
#     word2vector_file_path = config_ini_dict["file"]["word2vector_file_path"]
#     #trade5120 = config_ini_dict["file"]["trade5120"]
#   #  age200 = pd.read_csv('/home/zourui/data/predict_user_attribute20200911/raw_data/age/rank/rank.csv')
#     age200 = pd.read_csv(config_ini_dict["file"]["age_200"])
#     gender200 = pd.read_csv(config_ini_dict["file"]["gender_200"])
#     trade200 = pd.read_csv(config_ini_dict["file"]["trade_200"])
#     APP_DIR = os.path.dirname(os.path.realpath('__file__'))
#     if not os.path.exists(os.path.join(APP_DIR, "data")):
#         os.makedirs(os.path.join(APP_DIR, "data"))
#     input_number = 10
#     sentence_maxlen = 512
#     model_age = joblib.load(filepath_age)
# #    tf.keras.models.Model
#     model_position = load_model(filepath_position)
#     model_trade = joblib.load(filepath_trade)
#     model_gender = joblib.load(filepath_gender)
#     user_profile_svm_model_obj = UserProfileSVMModel(config_ini_dict["file"]["filepath_position",data_jsonl)

#     embedding_matrix, word2vector_dict, word2index_dict = user_profile_svm_model_obj.trans_gensim_word2vec2tf_embedding(
#         word2vector_file_path)
#     vocab_size, embedding_dim = embedding_matrix.shape


# print(user_profile_svm_model_obj.data_jsonl)

# user_profile_svm_model_obj.predict()


# import numpy as np
# a = np.array([12,3,4])


# # data = 1
# def Array():

#     def __init__(self,input_list):
#         self.input_list = input_list

#     def sum(self):
#         data += 1
#         return math.sum(self.input_list)

# a.sum()
