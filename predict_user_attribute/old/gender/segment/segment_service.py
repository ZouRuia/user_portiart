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
    words = a.cut("""12月1日据外媒报道，36kr优客工国家卫健防守打法是公安部委场或于1月赴美上市，拟募资1亿美元。优客据悉国家卫健委国管公积金最早将在25号公安部当周递交IPO申请文件。""")
    print(words)
