import os
import sys
import configparser

APP_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(APP_DIR)

from config import RAW_DATA_DIR,OUTPUT_DIR,COLUMN_LIST,X_COLUMN_LIST,ID_COLUMN,Y_COLUMN,Y_TRANSFER_DICT
from config import OUTPUT_PREDICT_DF,OUTPUT_TRAIN_DF,TRAIN_X,TRAIN_Y,VALID_X,VALID_Y,TEST_X,TEST_Y

def transfer_data():
    """处理数据，生成训练集，测试集，验证集
    """
    file_list = []
    for root,_,path_list in os.walk(RAW_DATA_DIR):
        for path in path_list:
            file_list.append(os.path.join(root,path))
    print("数据文件总共有{}个,正在读取数据".format())
    all_file_data = []
    for i,file_name in enumerate(file_list):
        path = os.path.join(RAW_DATA_DIR,file_name)
        print("正在读取第{}个文件：{}".format(i,path))
        all_file_data.append(pd.read_csv(,sep='\001',header=None,error_bad_lines=False))
    df = pd.concat(all_file_data,axis=0,ignore_index=True)
    del all_file_data
    #选取使用的列
    df = df[X_COLUMN_LIST + ID_COLUMN]
    
    #缺失值处理
    df.fillna("",inplace=True)

    #处理y_column的值
    df[Y_COLUMN] = df[Y_COLUMN].apply(lambda x : str(x))
    df[Y_COLUMN].replace(Y_TRANSFER_DICT,inplace=True)
    
    #区分 预测数据 和 非预测数据，并保存
    def split_train_predict(value,y_set = set(y_transfer_dict.values())):
        if value in y_set:
            return True
        return False
    predict_df = df[-df[y_col].apply(split_train_predict)]
    train_df = df[df[y_col].apply(split_train_predict)]

    predict_df.to_csv(OUTPUT_PREDICT_CSV,index=False)
    train_df.to_csv(OUTPUT_TRAIN_CSV,index=False)

    #生成训练数据
    # x multi_hot_column数据
    x_multi_hot_column_odict = collections.OrderedDict()
    x_multi_hot_column_set = set(X_COLUMN_LIST)
    for i,column in enumerate(multi_hot_column_set):
        multi_hot_column_odict[column] = sklearn.preprocessing.MultiLabelBinarizer()
        split_column_list = [item.split(",") if isinstance(item,str) else str(item) for item in train_df[column]]
        #multi_hot_column_odict[column].fit(split_column_list)
        train_df[column] = multi_hot_column_odict[column].fit_transform(split_column_list).tolist()
    

    
    train, test = sklearn.model_selection.train_test_split(train_df, test_size=0.2)
    train, val = sklearn.model_selection.train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    def generate_xy(data_df, y_col, multi_hot_column_odict, shuffle=True):
        """"""
        x_df = data_df.copy()
        x_list = []
        y_list = []
        for i,index in enumerate(x_df.index):
            temp_x_list = np.concatenate(x_df.loc[index,[column for column in multi_hot_column_odict]]).tolist()
            temp_y_list = x_df.loc[index,y_col]
            x_list.append(temp_x_list)
            y_list.append(temp_y_list)
            if i % 10000 == 0:
                print(i)
        return np.array(x_list,dtype=np.float32), np.array(y_list,dtype=np.uint8)

    x_train,y_train = generate_xy(train,y_col,multi_hot_column_odict)
    x_valid,y_valid = generate_xy(val,y_col,multi_hot_column_odict)
    x_test,y_test = generate_xy(test,y_col,multi_hot_column_odict)

    with open(TRAIN_X,"wb") as f:
        pickle.dump(x_train,f)
    with open(TRAIN_Y,"wb") as f:
        pickle.dump(y_train,f)

    with open(VALID_X,"wb") as f:
        pickle.dump(x_valid,f)
    with open(VALID_Y,"wb") as f:
        pickle.dump(y_valid,f)

    with open(TEST_X,"wb") as f:
        pickle.dump(x_test,f)    
    with open(TEST_Y,"wb") as f:
        pickle.dump(y_test,f)


def load_train_data():

    with open(TRAIN_X,"rb") as f:
        x = pickle.load(f)
    with open(TRAIN_Y,"rb") as f:
        y = pickle.load(f)

    print("x_train.shape={}".format(x.shape))
    print("y_train.shape={}".format(y.shape))
    return x,y

def load_valid_data():

    with open(VALID_X,"rb") as f:
        x = pickle.load(f)
    with open(VALID_Y,"rb") as f:
        y = pickle.load(f)

    print("x_valid.shape={}".format(x.shape))
    print("y_valid.shape={}".format(y.shape))
    return x,y

def load_test_data():

    with open(TEST_X,"rb") as f:
        x = pickle.load(f)
    with open(TEST_Y,"rb") as f:
        y = pickle.load(f)

    print("x_test.shape={}".format(x.shape))
    print("y_test.shape={}".format(y.shape))
    return x,y

def load_predict_data():

    



    
    
    

