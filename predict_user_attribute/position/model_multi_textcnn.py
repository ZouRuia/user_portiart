import tensorflow as tf

def build_model(
    input_number:int,
    sentence_maxlen:int,
    vocab_size:int,
    tag_size:int,
    embedding_dim:int,
    embedding_matrix=None,
    is_embedding_training:bool=True,
    embedding_dropout_rate:float = 0.0,
    learning_rate = 1e-3,
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

    input_list =[ 
        tf.keras.layers.Input(shape=(sentence_maxlen,),name="input{}".format(i))
        for i in range(input_number)
    ]
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
    output_list = [
        embedding_layer(input_list[i]) for i in range(input_number)
    ]

    embedding_dropout_layer = tf.keras.layers.Dropout(embedding_dropout_rate)
    cnn_layer0 = tf.keras.layers.Conv1D(filters=128, kernel_size=10, strides=1, padding='valid',activation="tanh")
    cnn_layer1 = tf.keras.layers.MaxPool1D(2, padding='valid')
    cnn_layer2 = tf.keras.layers.Flatten()
    for layer in [embedding_dropout_layer,cnn_layer0,cnn_layer1,cnn_layer2]:
        output_list = [
            layer(output_list[i]) for i in range(input_number)
        ]
    output = tf.keras.layers.Concatenate()(output_list)
    
    # output = tf.keras.layers.Concatenate()(output_list)
    # output = tf.keras.layers.Dropout(embedding_dropout_rate)(output)
    # output = tf.keras.layers.Conv1D(filters=512, kernel_size=20, strides=1, padding='valid',activation="tanh")(output)
    # output = tf.keras.layers.MaxPool1D(2, padding='valid')(output)
    # output = tf.keras.layers.Flatten()(output)

    output = tf.keras.layers.Dropout(0.5)(output)
    #output = tf.keras.layers.Dense(16, activation='tanh',kernel_regularizer=tf.keras.regularizers.l2())(output)

    if tag_size > 2:
        output = tf.keras.layers.Dense(tag_size, activation='softmax',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2())(output)
        # output = tf.keras.layers.Dense(tag_size, activation='softmax',use_bias=True)(output)
        print("这是一个多分类模型")
    elif tag_size == 2:
        output = tf.keras.layers.Dense(1, activation='sigmoid',use_bias=True,kernel_regularizer=tf.keras.regularizers.l2())(output)
        # output = tf.keras.layers.Dense(1, activation='sigmoid',use_bias=True)(output)
        print("这是一个二分类模型")
    else:
        raise Exception("类别错误")

    model = tf.keras.Model(inputs=input_list, outputs=output, name='multi_textcnn')

    if tag_size > 2:
        metric_list = [
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ]
        model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                clipnorm=1.0,
                clipvalue=0.5,
            ),
            metrics = metric_list,
        )
    else:
        metric_list = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(num_thresholds=10000)
        ]
        model.compile(
            loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3,
                clipnorm=1.0,
                clipvalue=0.5,
            ),
            metrics = metric_list,
        )
    return model