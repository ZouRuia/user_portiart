import tensorflow as tf

class build_lr_model(x_dim,y_onehot):
    """keras创建模型
    """
    y_dim = len(y_onehot.classes_)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    if y_dim == 2:
        model.add(
            tf.keras.layers.Dense(1, 
                activation='sigmoid',
                use_bias=True,
                #kernel_regularizer=tf.keras.regularizers.l1(0.01)
            )
        )
    elif y_dim > 2:
        model.add(
            tf.keras.layers.Dense(
                y_dim, 
                activation='softmax',use_bias=True,
                #kernel_regularizer=tf.keras.regularizers.l1(0.01)
            )
        )
    else:
        raise Exception("分类类别错误，需要 >=2 ，获得{}".format(y_dim))

    if y_dim == 2:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss = tf.keras.losses.BinaryCrossentropy(), 
            metrics=[tf.keras.metrics.BinaryAccuracy()])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(), 
            metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

