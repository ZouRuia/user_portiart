import os
import time
import collections
import sklearn.model_selection
import gensim
import numpy as np
import tensorflow as tf
import json
import gensim
import LAC
import pandas as pd
import sklearn.utils

APP_DIR = os.path.dirname(os.path.realpath("__file__"))
if not os.path.exists(os.path.join(APP_DIR, "data")):
    os.makedirs(os.path.join(APP_DIR, "data"))


class Position_model(object):
    def __init__(self, position_output_data_path, word2vector_file_path):
        self.position_output_data_path = position_output_data_path
        self.word2vector_file_path = word2vector_file_path
        (
            self.embedding_matrix,
            self.word2vector_dict,
            self.word2index_dict,
        ) = self.trans_gensim_word2vec2tf_embedding(self.word2vector_file_path)
        self.vocab_size, self.embedding_dim = self.embedding_matrix.shape
        # x_npa,y_npa,tag2index_dict = self.trans_multi_input_tokenize_data2npa(self.position_output_data_path,self.sentence_maxlen,self.word2index_dict)
        # class_weight_dict = {tag:np.sqrt(len(y_npa)/number) for tag,number in enumerate(np.bincount(y_npa))}
        # x_train_eval,y_train_eval,x_test,y_test = self.split_train_eval_test_npa(x_npa,y_npa)
        # tag_size = len(tag2index_dict)

    def build_model(
        self,
        input_number: int,
        sentence_maxlen: int,
        vocab_size: int,
        tag_size: int,
        embedding_dim: int,
        embedding_matrix=None,
        is_embedding_training: bool = True,
        embedding_dropout_rate: float = 0.0,
        learning_rate=1e-3,
    ):
        """建立模型
        input:
            sentence_maxlen : 句子的长度
            vocab_size : 词的个数
            tag_size : 分类的个数
            embedding_dim : word2vec训练时设置的向量长度
            embedding_matrix : word2vec词向量矩阵
            is_embedding_training : embedding层是否加入训练
            embedding_dropout_rate : embedding层dropout的比率
        """

        input_list = [
            tf.keras.layers.Input(shape=(sentence_maxlen,), name="input{}".format(i))
            for i in range(input_number)
        ]
        # embedding层
        if not (embedding_matrix is None):
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                trainable=is_embedding_training,
                input_length=sentence_maxlen,
            )
        else:
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                trainable=is_embedding_training,
                input_length=sentence_maxlen,
            )
        output_list = [embedding_layer(input_list[i]) for i in range(input_number)]

        embedding_dropout_layer = tf.keras.layers.Dropout(embedding_dropout_rate)
        cnn_layer0 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=10, strides=1, padding="valid", activation="tanh"
        )
        cnn_layer1 = tf.keras.layers.MaxPool1D(2, padding="valid")
        cnn_layer2 = tf.keras.layers.Flatten()
        for layer in [embedding_dropout_layer, cnn_layer0, cnn_layer1, cnn_layer2]:
            output_list = [layer(output_list[i]) for i in range(input_number)]
        output = tf.keras.layers.Concatenate()(output_list)

        output = tf.keras.layers.Dropout(0.5)(output)

        if tag_size > 2:
            output = tf.keras.layers.Dense(
                tag_size,
                activation="softmax",
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l2(),
            )(output)
            # output = tf.keras.layers.Dense(tag_size, activation='softmax',use_bias=True)(output)
            print("这是一个多分类模型")
        elif tag_size == 2:
            output = tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l2(),
            )(output)
            # output = tf.keras.layers.Dense(1, activation='sigmoid',use_bias=True)(output)
            print("这是一个二分类模型")
        else:
            raise Exception("类别错误")

        model = tf.keras.Model(inputs=input_list, outputs=output, name="multi_textcnn")

        if tag_size > 2:
            metric_list = [
                tf.keras.metrics.SparseCategoricalAccuracy(),
            ]
            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    clipnorm=1.0,
                    clipvalue=0.5,
                ),
                metrics=metric_list,
            )
        else:
            metric_list = [
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.AUC(num_thresholds=10000),
            ]
            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=1e-3,
                    clipnorm=1.0,
                    clipvalue=0.5,
                ),
                metrics=metric_list,
            )
        return model

    def split_train_eval_test_dataset(self, dataset):
        """区分训练验证测试集"""
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()
        print("总共有数据{}条".format(dataset_size))
        dataset = dataset.shuffle(dataset_size, seed=1)
        train_size = int(0.6 * dataset_size)
        eval_size = int(0.2 * dataset_size)
        test_size = int(0.2 * dataset_size)

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        eval_dataset = test_dataset.skip(eval_size)
        test_dataset = test_dataset.take(test_size)
        return (
            train_dataset.prefetch(tf.data.experimental.AUTOTUNE),
            eval_dataset.prefetch(tf.data.experimental.AUTOTUNE),
            test_dataset.prefetch(tf.data.experimental.AUTOTUNE),
        )

    def split_train_eval_test_npa(self, x_npa, y_npa):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
            x_npa, y_npa, test_size=0.2, random_state=24
        )
        return x_train, y_train, x_test, y_test

    def build_model_callback(self):
        callback_path = os.path.join(
            APP_DIR, "model_callback", "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        )
        if not os.path.exists(os.path.dirname(callback_path)):
            os.makedirs(os.path.dirname(callback_path))

        model_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=callback_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            mode="auto",
            save_freq="epoch",
        )

        tf_board_dir = os.path.join(APP_DIR, "model_tensorboard")
        if not os.path.exists(os.path.dirname(tf_board_dir)):
            os.makedirs(os.path.dirname(tf_board_dir))

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tf_board_dir, histogram_freq=1, update_freq="batch"
        )
        return model_callback, tensorboard_callback

    def trans_gensim_word2vec2tf_embedding(self, word2vector_file_path: str):
        """把gensim的word2vec结果转化为tf.keras.layers.Embedding需要的结果"""

        word2vec_model = gensim.models.Word2Vec.load(word2vector_file_path)

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
        self, data_file_path: str, x_max_length: int = None, word2index_dict=None
    ):
        """把已经分好词的data文件转化为tf.data , 多输入版本"""

        tag2index_dict = {}
        tag_index_count = len(tag2index_dict)

        x_list = []
        y_list = []
        with open(data_file_path) as f:
            for line in f:
                temp_dict = json.loads(line.strip())
                text_tokenize_list = temp_dict["all_content_tokenize"]
                tag = temp_dict["tag"].strip()
                if not (tag in tag2index_dict):
                    tag2index_dict[tag] = tag_index_count
                    tag_index_count += 1
                x_list.append(
                    [
                        [self.trans2index(word2index_dict, word) for word in word_list]
                        for word_list in text_tokenize_list
                    ]
                )
                y_list.append(tag2index_dict[tag])
        y_npa = np.array(y_list, dtype=np.uint8)

        print("x_list[:1]:{}".format(x_list[:1]))
        print("y_list[:1]:{}".format(y_list[:1]))

        # 写入文件
        with open(
            os.path.join(APP_DIR, "data/tag2index.json"), "w", encoding="utf8"
        ) as f:
            json.dump(tag2index_dict, f, ensure_ascii=False)

        if not x_max_length:
            x_max_length0 = np.max(np.array([len(v) for v in x_list]))
            x_max_length = int(
                np.max(np.percentile(np.array([len(v) for v in x_list]), 99.7))
            )
            print("数据集中最长的句子长度为:{},设定的最长的句子长度为:{}".format(x_max_length0, x_max_length))

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

        x_npa, y_npa = sklearn.utils.shuffle(x_npa, y_npa, random_state=0)
        print("x_npa[:1]:{}".format(x_npa[:1]))
        print("y_npa[:1]:{}".format(y_npa[:1]))
        print("x_npa.shape = {}".format(x_npa.shape))
        print("y_npa.shape = {}".format(y_npa.shape))

        return x_npa, y_npa, tag2index_dict

    def train(
        self,
        input_number,
        sentence_maxlen,
        epochs,
        batch_size,
        learning_rate,
        is_embedding_training,
        embedding_dropout_rate,
    ):
        x_npa, y_npa, tag2index_dict = self.trans_multi_input_tokenize_data2npa(
            self.position_output_data_path, self.sentence_maxlen, self.word2index_dict
        )
        class_weight_dict = {
            tag: np.sqrt(len(y_npa) / number)
            for tag, number in enumerate(np.bincount(y_npa))
        }
        x_train_eval, y_train_eval, x_test, y_test = self.split_train_eval_test_npa(
            x_npa, y_npa
        )
        tag_size = len(tag2index_dict)
        # 建模模型
        # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        # with strategy.scope():
        model = self.build_model(
            input_number=input_number,
            sentence_maxlen=sentence_maxlen,
            vocab_size=vocab_size,
            tag_size=tag_size,
            embedding_dim=embedding_dim,
            embedding_matrix=self.embedding_matrix,
            is_embedding_training=is_embedding_training,
            embedding_dropout_rate=embedding_dropout_rate,
            learning_rate=learning_rate,
        )

        # model_callback = build_model_callback()

        print(model.summary())

        history = model.fit(
            {"input{}".format(i): x_train_eval[:, i] for i in range(input_number)},
            y_train_eval,
            validation_split=0.25,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            # callbacks=list(model_callback),
            # class_weight = class_weight_dict,
        )

        # 测试集
        model.evaluate(
            {"input{}".format(i): x_test[:, i] for i in range(input_number)}, y_test
        )

        # 模型保存路径
        model_path = os.path.join(APP_DIR, "model_output1")
        # 保存模型
        print("即将保存模型：{}".format(model_path))


#     model.save(model_path)


# if __name__ == "__main__":

# position_output_data_path = "/mnt/d/zourui/predict_user_attribute20201214/age/raw_data/user_profile/position/最终数据集tokenize8.jsonl"

# word2vector_file_path = "/home/zourui/data/dim256/word2vector.bin"
# #模型保存路径
# model_path = os.path.join(APP_DIR,"model_output1")
# ###以上是需要修改的部分

# if not os.path.exists(os.path.dirname(model_path)):
#     os.makedirs(os.path.dirname(model_path))

# 导入gensim的word2vector
# embedding_matrix,word2vector_dict,word2index_dict = self.trans_gensim_word2vec2tf_embedding(word2vector_file_path)
# vocab_size,embedding_dim = embedding_matrix.shape

# x_npa,y_npa,tag2index_dict = self.trans_multi_input_tokenize_data2npa(self.position_output_data_path,self.sentence_maxlen,self.word2index_dict)
# class_weight_dict = {tag:np.sqrt(len(y_npa)/number) for tag,number in enumerate(np.bincount(y_npa))}
# x_train_eval,y_train_eval,x_test,y_test = self.split_train_eval_test_npa(x_npa,y_npa)
# tag_size = len(tag2index_dict)

model = Position_model(
    position_output_data_path=config_ini_dict["file"]["position_output_data_path"],
    word2vector_file_path=config_ini_dict["file"]["word2vector_file_path"],
)
model.train(
    input_number=10,
    sentence_maxlen=512,
    epochs=7,
    batch_size=64,
    learning_rate=1e-3,
    is_embedding_training=False,
    embedding_dropout_rate=0.2,
)
