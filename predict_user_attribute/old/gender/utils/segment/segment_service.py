# coding=utf-8
import os
import jieba
import jieba.analyse
import re
import logging
from utils.BaseUtil import singleton
import jieba.posseg as pseg
from utils.BaseUtil import is_number

jieba.setLogLevel(logging.INFO)
dir_path = os.path.dirname(os.path.abspath(__file__))


@singleton
class SegmentService(object):

    def __init__(self):
        hx_dict_path = os.path.join(dir_path, "hx_dict.txt")
        company_path = os.path.join(dir_path, "company.txt")
        jieba.load_userdict(hx_dict_path)
        jieba.load_userdict(company_path)

        hx_stop_words_path = os.path.join(dir_path, "hx_stop_words.txt")
        jieba.analyse.set_stop_words(hx_stop_words_path)
        self.stopwords = [line.strip() for line in open(hx_stop_words_path, mode='r', encoding="utf-8").readlines()]

    def cut(self, content, cut_all=False, stop_word=True):
        """
        分词
        :param content: 分词文本
        :param cut_all: 默认精确模式
        :param stop_word: 是否使用停用词，默认使用
        :return: 分词结果
        """
        word_list = jieba.cut(content, cut_all=cut_all)
        if stop_word:
            word_list = self.move_stopwords(word_list)
        else:
            word_list = [x.strip() for x in word_list]
        return word_list

    def cut_number(self, content):
        words = jieba.cut(content)
        return set([x for x in words if self.is_number(x) and x.strip()])

    def move_stopwords(self, words):
        """
        移除列表中的停用词
        :param words: 输入词列表
        :return: 去停用词列表
        """
        return [x.strip() for x in words if x not in self.stopwords and len(x.strip()) > 0]

    @staticmethod
    def extract_tags(content, top_n=10, with_weight=False):
        return jieba.analyse.extract_tags(content, withWeight=with_weight, topK=top_n)

    @staticmethod
    def text_rank(content, top_n=10):
        return jieba.analyse.textrank(content, topK=top_n, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))

    @staticmethod
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

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        try:
            index = s.find("%")
            if index < 0:
                return False
            else:
                return True
        except (TypeError, ValueError):
            pass
        return False

    def cut_for_seg(self, content):
        content = re.sub("<.*?>", "", content)
        seg_list = pseg.lcut(content)
        filtered_words_list = []
        """对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词"""
        for seg in seg_list:
            if len(seg.word) <= 1:
                continue
            elif seg.word in self.stopwords:
                continue
            elif seg.flag == "eng":
                if len(seg.word) <= 2:
                    continue
                else:
                    filtered_words_list.append(seg.word)
            elif seg.flag.startswith("n"):
                filtered_words_list.append(seg.word)
            elif seg.flag in ["x", "eng"]:  # 是自定一个词语或者是英文单词
                filtered_words_list.append(seg.word)
        return filtered_words_list

    def cut_for_word2vec(self, content):
        word_list = self.cut(content)
        word_list = [word for word in word_list if len(word) > 1 and not is_number(word)]
        return word_list



if __name__ == '__main__':
    a = SegmentService()
    # words = a.cut("【多地加码停车设施规划，智慧停车再迎风口】近期，山东青岛市发布的《关于进一步加强停车设施规划建设管理工作的实施意见》提出工作目标，到2022年，全市经营性停车场基本纳入智能停车一体化平台。报告预测，到2020年全国停车位数量将达到1.19亿个，若汽车保有量以近五年的复合增速持续增长，届时全国民用汽车保有量将达到2.9亿辆左右，与停车位的比例仅为1：0.4，配比严重偏低。国家发改委城市和小城镇改革发展中心理事会理事长李铁表示，未来随着智慧停车从一线城市向二三四线城市持续渗透，2020年智慧停车产业发展将再度加速。")
    words = a.cut("共享街机“街机超人”宣布，公司完成新一轮战略融资，由上海典商共策投资领投、部分老股东追加投资。对于此轮融资，创始人兼CEO贠垚韬表示，资金将主要用于团队建设、服务能力的提升以及加大在技术方面的投入。")
    print(words)
