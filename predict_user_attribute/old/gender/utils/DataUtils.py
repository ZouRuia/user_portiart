# coding=utf-8
import time
import datetime

# 今天日期
today = datetime.date.today()

# 昨天日期
yesterday = today + datetime.timedelta(days=-1)

# 明天日期
tomorrow = today + datetime.timedelta(days=1)

# 后天
acquire = today + datetime.timedelta(days=2)

# 昨天开始时间戳
yesterday_start_time = int(time.mktime(time.strptime(str(yesterday), '%Y-%m-%d')))

# 昨天结束时间戳
# yesterday_end_time = int(time.mktime(time.strptime(str(today), '%Y-%m-%d'))) - 1
yesterday_end_time = int(time.mktime(time.strptime(str(today), '%Y-%m-%d')))

# 今天开始时间戳
# today_start_time = yesterday_end_time + 1
today_start_time = yesterday_end_time

# 今天结束时间戳
# today_end_time = int(time.mktime(time.strptime(str(tomorrow), '%Y-%m-%d'))) - 1
today_end_time = int(time.mktime(time.strptime(str(tomorrow), '%Y-%m-%d')))

# 明天开始时间戳
# tomorrow_start_time = int(time.mktime(time.strptime(str(tomorrow), '%Y-%m-%d')))
tomorrow_start_time = int(time.mktime(time.strptime(str(tomorrow), '%Y-%m-%d')))

# 明天结束时间戳
tomorrow_end_time = int(time.mktime(time.strptime(str(acquire), '%Y-%m-%d')))


def get_date(num=0):
    """获取某一天的日期"""
    result_data = today + datetime.timedelta(days=num)
    # print(result_data)
    return result_data


def get_time(num=0):
    """获取莫一天的前后时间戳"""
    result_data = today + datetime.timedelta(days=num)
    result_start_time = int(time.mktime(time.strptime(str(result_data), '%Y-%m-%d')))
    result_end_time = int(result_start_time + 3600 * 24)
    return result_start_time, result_end_time


def to_time(data=None):
    """获取指定日期的前后时间戳"""
    result_start_time = int(time.mktime(time.strptime(str(today), '%Y-%m-%d')))
    if data:
        if type(data) == str:
            try:
                if '-' in data:
                    result_start_time = int(time.mktime(time.strptime(str(data), '%Y-%m-%d')))
                elif '/' in data:
                    result_start_time = int(time.mktime(time.strptime(str(data), '%Y/%m/%d')))
                elif '.' in data:
                    result_start_time = int(time.mktime(time.strptime(str(data), '%Y.%m.%d')))
                elif data.isalnum():
                    result_start_time = int(time.mktime(time.strptime(str(data), '%Y%m%d')))
                else:
                    raise ValueError('请输入正确的日期格式，例如：2019-01-01')
            except Exception as e:
                raise ValueError('请输入正确的日期格式，例如：2019-01-01')
        else:
            raise TypeError('参数必须是字符串，例如：2019-12-04')
    result_end_time = int(result_start_time + 3600 * 24)
    return result_start_time, result_end_time


if __name__ == "__main__":
    print('昨天日期: {}'.format(yesterday))
    print('昨天开始时间戳: {}'.format(yesterday_start_time))
    print('昨天结束时间戳: {}'.format(yesterday_end_time))

    print('今天日期: {}'.format(today))
    print('今天开始时间戳: {}'.format(today_start_time))
    print('今天结束时间戳: {}'.format(today_end_time))

    print('明天日期: {}'.format(tomorrow))
    print('明天开始时间戳: {}'.format(tomorrow_start_time))
    print('明天结束时间戳: {}'.format(tomorrow_end_time))

    print('目标日期: {}'.format(get_date(10)))

    print('目标时间戳: {}'.format(get_time(10)))

    print('目标日期时间戳: {}'.format(to_time('2019.01.09')))
    print('默认为当天，时间戳: {}'.format(to_time()))
