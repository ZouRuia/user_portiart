from model import build_model
import gensim
import numpy as np
import tensorflow as tf
import os
import time
import sklearn.model_selection
from utils import trans_gensim_word2vec2tf_embedding,trans_data2tf_data,trans_tokenize_data2tf_data
from model import build_model

APP_DIR  = os.path.dirname(os.path.realpath(__file__))

def split_train_eval_test_dataset(dataset):
    """区分训练验证测试集
    """
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    print("总共有数据{}条".format(dataset_size))
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
    x_train,x_test,y_train,y_test =  sklearn.model_selection.train_test_split(x_npa, y_npa, test_size=0.2, random_state=42)
    return x_train,y_train,x_test,y_test

def build_model_callback():
    callback_path = os.path.join(APP_DIR,"model_callback","weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    if not os.path.exists(os.path.dirname(callback_path)):
        os.makedirs(os.path.dirname(callback_path))

    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=callback_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch',
    )

    tf_board_dir = os.path.join(APP_DIR,"model_tensorborad")
    if not os.path.exists(os.path.dirname(tf_board_dir)):
         os.makedirs(os.path.dirname(tf_board_dir))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tf_board_dir,histogram_freq=1,update_freq='batch')
    return model_callback,tensorboard_callback



if __name__ == "__main__":
    #句子的最大长度
    sentence_maxlen = 128
    #训练次数
    epochs = 20
    #批大小
    batch_size = 64
    #学习率
    #learning_rate = 1e-3
    #embedding layer是否参加训练
    is_embedding_training = True
    #embedding的dropout比率
    embedding_dropout_rate = 0.25

    #数据文件
    #data_csv = os.path.join(APP_DIR,"data","最终数据集.csv")
    data_jsonl = os.path.join(APP_DIR,"data","最终数据集tokenize.jsonl")
    #word2vec路径
    word2vector_file_path = os.path.join(APP_DIR,"data","word2vector.bin")
    #模型保存路径
    model_path = os.path.join(APP_DIR,"model_output")
    ###以上是需要修改的部分

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    #导入gensim的word2vector
    embedding_matrix,word2vector_dict,word2index_dict = trans_gensim_word2vec2tf_embedding(word2vector_file_path)
    vocab_size,embedding_dim = embedding_matrix.shape

    #划分训练集+验证集，测试集
    #x_npa,y_npa,tag2index_dict = trans_data2tf_data(data_csv,sentence_maxlen,word2index_dict)
    x_npa,y_npa,tag2index_dict = trans_tokenize_data2tf_data(data_jsonl,sentence_maxlen,word2index_dict)
    x_train_eval,y_train_eval,x_test,y_test = split_train_eval_test_npa(x_npa,y_npa)
    print(x_train_eval[:3])
    print(y_train_eval[:3])
    tag_size = len(tag2index_dict)

    #建模模型
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    with strategy.scope():
        model = build_model(sentence_maxlen = sentence_maxlen,
            vocab_size = vocab_size,
            tag_size = tag_size,
            embedding_dim = embedding_dim,
            embedding_matrix = embedding_matrix,
            is_embedding_training = is_embedding_training,
            embedding_dropout_rate = embedding_dropout_rate
        )

    model_callback = build_model_callback()

    print(model.summary())

    history = model.fit(
        #train_dataset,
        x_train_eval,y_train_eval,
        validation_split = 0.25,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=list(model_callback),
    )

    #保存模型
    print("即将保存模型：{}".format(model_path))
    model.save(model_path)

    #测试集
    model.evaluate(x_test,y_test)


