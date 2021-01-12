# coding=utf-8
import time
import numpy as np
import hashlib


def singleton(cls):
    """单例类装饰器"""
    _instance = {}

    def _deco(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _deco


def try_cathch(func):
    """try catch装饰器"""

    def deco(self, name, dic1={}, url=None):
        try:
            func(self, name, dic1=dic1, url=url)
        except Exception as e:
            print('抓取({})错误: {}'.format(name, e))

    return deco


def count_time(func):
    """计算函数运行时间装饰器"""

    def deco(self, name):
        start = time.time()
        data = func(self, name)
        end = time.time()
        time0 = end - start
        print('({}) 方法运行时间: {}'.format(name, time0))
        return data

    return deco


def is_number(s):
    """
    判断是否为数字，例如["123", "0.124", "10%"]
    :param s: 需要判断是否为数字的字符串
    :return: 
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    # try:
    #     import unicodedata
    #     unicodedata.numeric(s)
    #     return True
    # except (TypeError, ValueError):
    #     pass

    try:
        index = s.find("%")
        if index < 0:
            return False
        else:
            return True
    except (TypeError, ValueError):
        pass
    return False


def compare_set(a, b):
    """判断相同元素个数"""
    if len(a) == 0 or len(b) == 0:
        return 0
    if len(a) >= len(b):
        c = [x for x in b if x in a]
    else:
        c = [x for x in a if x in b]
    return len(c)


# 获取时间时间参数
def get_time(number=0):
    """number: 获取几天前的时间参数，默认为0代表当天"""
    now_date = time.strftime('%Y/%m/%d', time.localtime(time.time()))
    struct_time_date = time.strptime(now_date, u"%Y/%m/%d")
    now_time = int(time.mktime(struct_time_date))
    # data_wday = time.strftime("%A", struct_time_date)
    target_time = now_time - 86400 * number
    target_struct_time_int = time.localtime(target_time)
    target_date = time.strftime("%Y-%m-%d", target_struct_time_int)
    target_wday = time.strftime("%A", target_struct_time_int)
    start_time = target_time
    end_time = start_time + 86400
    return target_date, target_wday, start_time, end_time


def sim_score(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    molecule = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if not molecule:
        return 0
    score = num / molecule
    return score


def md5(s, nlen=32):
    """生成唯一标识"""
    m2 = hashlib.md5()
    m2.update(s.encode("utf-8"))
    res = m2.hexdigest()
    m2.update(res.encode("utf-8"))
    res = m2.hexdigest()
    return res[0:nlen]


def seconds_time(seconds):
    """秒转化成时间格式"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    time1 = "%d:%02d:%02d" % (h, m, s)
    return time1