import tensorflow as tf

def build_model(
    sentence_maxlen:int,
    vocab_size:int,
    tag_size:int,
    embedding_dim:int,
    embedding_matrix=None,
    is_embedding_training:bool=True,
    embedding_dropout_rate:float = 0.0,
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

    model = tf.keras.models.Sequential()

    #embedding层
    if not (embedding_matrix is None):
        embedding_layer = tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim = embedding_dim,
            embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
            trainable = is_embedding_training,
            input_length = sentence_maxlen,
        )
    else:
        embedding_layer = tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim = embedding_dim,
            trainable = is_embedding_training,
            input_length = sentence_maxlen,
        )
    model.add(embedding_layer)
    model.add(tf.keras.layers.Dropout(embedding_dropout_rate))
    model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='valid'))
    model.add(tf.keras.layers.MaxPool1D(2, padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(16, activation='relu'))

    if tag_size > 2:
        model.add(tf.keras.layers.Dense(tag_size, activation='softmax',use_bias=True))
        print("这是一个多分类模型")
    elif tag_size == 2:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid',use_bias=True))
        print("这是一个二分类模型")
    else:
        raise Exception("类别错误")

    if tag_size > 2:
        metric_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ]
        model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3,
                #clipnorm=1.0,
                #clipvalue=0.5,
            ),
            metrics = metric_list,
        )
    else:
        metric_list = [
            tf.keras.metrics.BinaryAccuracy(),
        ]
        model.compile(
            loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3,
                #clipnorm=1.0,
                #clipvalue=0.5,
            ),
            metrics = metric_list,
        )
    return model