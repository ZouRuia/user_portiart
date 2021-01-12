import json
import os
import sys
import logging
import configparser
import collections
import numpy as np
import pandas as pd
import re

# import tensorflow as tf
import LAC
import happybase
import joblib
import sklearn.utils
from sklearn.feature_selection import RFE

APP_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(APP_DIR)
config_ini_dict = configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR, "config.ini"))
logging.info(config_ini_dict)


class Position_data_deal(object):
    def __init__(
        self,
        data_input_path,
        title_tokenize_path,
        content_tokenize_path,
        position_output_data_path,
    ):
        self.data_input_path = data_input_path
        self.title_tokenize_path = title_tokenize_path
        self.content_tokenize_path = content_tokenize_path
        self.position_output_data_path = position_output_data_path

    def position_data_output(self):
        df_list = []
        with open(self.data_input_path) as f:
            for line in f:
                df_list.append(line.strip().split("\001"))
        df = pd.DataFrame(df_list)

        all_set = set([v.lower() for v in df[1]])
        ceo_set = set()
        ceo_pattern = re.compile(
            "|".join(
                map(re.escape, ["董事长", "总裁", "ceo", "董事", "合伙人", "总经理", "副总经理", "老板"])
            ),
            flags=re.IGNORECASE,
        )
        # 企业业务决策层
        decision_level_set = set()
        decision_level_pattern = re.compile(
            "|".join(
                map(
                    re.escape,
                    [
                        "cio",
                        "cfo",
                        "coo",
                        "cto",
                        "cko",
                        "cpo",
                        "cgo",
                        "cmo",
                        "cso",
                        "首席",
                        "行长",
                        "投资经理",
                    ],
                )
            ),
            flags=re.IGNORECASE,
        )
        # 企业一般管理人员
        manager_set = set()
        manager_pattern = re.compile(
            "|".join(map(re.escape, ["总监", "主管", "总助", "负责人", "中层干部"])),
            flags=re.IGNORECASE,
        )
        # 事业单位员工
        public_institution_set = set()
        public_institution_pattern = re.compile(
            "|".join(
                map(
                    re.escape,
                    [
                        "公务员",
                        "主任",
                        "教务",
                        "老师",
                        "教师",
                        "列车",
                        "政府",
                        "常委",
                        "教師",
                        "人民公仆",
                        "院长",
                        "医",
                        "驻华大使",
                    ],
                )
            ),
            flags=re.IGNORECASE,
        )
        # 高校学生
        student_set = set()
        student_pattern = re.compile(
            "|".join(
                map(
                    re.escape,
                    [
                        "二战",
                        "大专",
                        "高三",
                        "大四",
                        "大一",
                        "大二",
                        "大一",
                        "学生",
                        "研究生",
                        "博士",
                        "大学",
                        "应届生",
                        "本科",
                        "硕士",
                        "博士",
                    ],
                )
            ),
            flags=re.IGNORECASE,
        )
        # 其他（自由职业）
        other_set = set()
        other_pattern = re.compile(
            "|".join(map(re.escape, ["自由", "啥都干", "流民", "浪人"])), flags=re.IGNORECASE
        )
        # 企业普通员工
        staff_set = set()
        staff_pattern = re.compile(
            "|".join(
                map(
                    re.escape,
                    [
                        "工程师",
                        "民工",
                        "分析师",
                        "设计师",
                        "苦力",
                        "编程",
                        "员工",
                        "工人",
                        "架构师",
                        "清洁",
                        "经理助理",
                        "客户经理",
                        "项目经理",
                        "从业者",
                        "规划师",
                        "部经理",
                        "公务员",
                        "销售经理",
                    ],
                )
            ),
            flags=re.IGNORECASE,
        )
        for v in all_set:
            if list(re.finditer(ceo_pattern, v)):
                ceo_set.add(v)
            elif list(re.finditer(decision_level_pattern, v)):
                decision_level_set.add(v)
            elif list(re.finditer(manager_pattern, v)):
                manager_set.add(v)
            elif list(re.finditer(public_institution_pattern, v)):
                public_institution_set.add(v)
            elif list(re.finditer(student_pattern, v)):
                student_set.add(v)
            elif list(re.finditer(staff_pattern, v)):
                staff_set.add(v)
            else:
                if list(re.finditer(other_pattern, v)) or len(v) > 10:
                    other_set.add(v)
                else:
                    staff_set.add(v)

        def temp_function(
            x,
            ceo_set=ceo_set,
            decision_level_set=decision_level_set,
            manager_set=manager_set,
            public_institution_set=public_institution_set,
            staff_set=staff_set,
            student_set=student_set,
            other_set=other_set,
        ):
            # x = datetime.datetime.now().year - int(x.split("-")[0])
            y = 0
            if x in ceo_set:
                y = "企业创始人/CEO"
            elif x in decision_level_set:
                y = "企业业务决策层"
            elif x in manager_set:
                y = "企业一般管理人员"
            elif x in public_institution_set:
                y = "事业单位员工"
            elif x in staff_set:
                y = "企业普通员工"
            elif x in student_set:
                y = "高校学生"
            else:
                y = "其他（自由职业）"
            return y

        df[1] = df[1].apply(lambda x: temp_function(x.lower()))
        df.columns = ["aid", "position", *["c{}".format(v) for v in range(10)]]
        a_dict = {}
        with open(self.title_tokenize_path) as f:
            for line in f:
                temp_dict = json.loads(line.strip())
                aid = temp_dict["aid"]
                a_dict[aid] = {}
                title_tokenize = temp_dict["title_tokenize"]
                a_dict[aid]["title_tokenize"] = title_tokenize
        with open(self.content_tokenize_path) as f:
            for line in f:
                temp_dict = json.loads(line.strip())
                aid = temp_dict["aid"]
                if not aid in a_dict:
                    print(aid)
                    continue
                content_tokenize = temp_dict["content_tokenize"][:512]
                a_dict[aid]["content_tokenize"] = content_tokenize
        text_list = []
        y_list = []
        for index in df.index:
            y = df.loc[index, "position"]
            temp_text_list = []
            for i in range(10):
                c_text_list = []
                c = "c{}".format(i)
                aid = str(df.loc[index, c]).strip()
                if aid:
                    if aid in a_dict:
                        c_text_list = [
                            *a_dict[aid]["title_tokenize"],
                            "<PADDING>",
                            *a_dict[aid]["content_tokenize"][:512],
                        ]
                    else:
                        print("aid不存在", index, c, aid)
                else:
                    print(index)
                temp_text_list.append(c_text_list)
            text_list.append(temp_text_list)
            y_list.append(y)
        with open(self.position_output_data_path, "w") as f:
            for tag, content_tokenize in zip(y_list, text_list):
                f.write(
                    json.dumps(
                        {"tag": tag, "all_content_tokenize": content_tokenize},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# data_input_path = config_ini_dict["file"]["position_input_path"]
# title_tokenize_path = config_ini_dict["file"]["title_tokenize_path"]
# content_tokenize_path = config_ini_dict["file"]["content_tokenize_path"]
# position_output_data_path = config_ini_dict["file"]["position_output_data_path"]
position = Position_data_deal(
    data_input_path=config_ini_dict["file"]["position_input_path"],
    title_tokenize_path=config_ini_dict["file"]["title_tokenize_path"],
    content_tokenize_path=config_ini_dict["file"]["content_tokenize_path"],
    position_output_data_path=config_ini_dict["file"]["position_output_data_path"],
)
position.position_data_output()
