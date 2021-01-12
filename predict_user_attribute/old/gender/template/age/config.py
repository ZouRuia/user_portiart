import os
import sys

#目录层级，定义当前目录为项目目录
APP_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(APP_DIR)

#原始文件，已经join好的
RAW_DATA_DIR = "" 

# 输出文件，会自动生成如下文件夹
# data 原始文件生成的预训练文件,以及data_helper对象
# model_callback model的callback文件 TODO
# model_tf_board model的tensorbord文件 TODO
# model_save model的保存文件
OUTPUT_DIR = ""

COLUMN_LIST = ['用户id','用户姓名','用户性别','用户出生时间','所在公司','所处职位','行业','微信网名','微信性别','常驻省','常驻市','是否付费用户','付费等级','关注的作者','作者类型','关注的马甲','关注的公司','文章id','是否精选文章','浏览文章时间','浏览时长','文章频道','文章类型','文章标签','文章图片数量','文章字数','新定位省','新定位市','评论内容','打赏金额','设备型号','使用网络']
X_COLUMN_LIST = 
ID_COLUMN = '用户id'
Y_COLUMN = '用户性别'
Y_TRANSFER_DICT = 

if not os.path.exists(os.path.join(OUTPUT_DIR,"data")):
    os.makedirs(os.path.join(OUTPUT_DIR,"data"))

if not 

OUTPUT_PREDICT_CSV = os.path.join(OUTPUT_DIR,"data","predict_df.csv")
OUTPUT_TRAIN_CSV = os.path.join(OUTPUT_DIR,"data","train_df.csv")
TRAIN_X = os.path.join(OUTPUT_DIR,"data","train_x.pkl")
TRAIN_Y = os.path.join(OUTPUT_DIR,"data","train_y.pkl")
VALID_X = os.path.join(OUTPUT_DIR,"data","valid_x.pkl")
VALID_Y = os.path.join(OUTPUT_DIR,"data","valid_y.pkl")
TEST_X = os.path.join(OUTPUT_DIR,"data","test_x.pkl")
TEST_Y = os.path.join(OUTPUT_DIR,"data","test_y.pkl")


