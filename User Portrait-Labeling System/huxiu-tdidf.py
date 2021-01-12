'''def get_tfidf(self, title: str, content: str, top_n: int):
    """获得tfidf关键词
    Args:
        title:文章标题
        content:文章内容
        top_n:获得top_n个词
    Return:
        [{'tfidf': 0.28155462278770654, 'word': 'AI'},
        {'tfidf': 0.2126913482098162, 'word': '公司'}]
        tfidf: tfidf分数
        word: 词
    """
    pass
'''
import pandas as pd
import jieba
import jieba.analyse
import kashgari

class Get_tag():
    def __init__(self,e,f):
        self.e=e
        self.f=f

    # 去停用词
    # def drop_stopwords(contents, stopwords):
    def drop_stopwords(self,countents,stopwords):
        self.e =  countents
        self.f = stopwords
        contents_clean = []
        all_words = []
        for line in self.e:
            line_clean = []
            for word in line:
                if word in self.f:
                    continue
                line_clean.append(word)
                all_words.append(str(word))  # 所有的词组成一个列表
            contents_clean.append(line_clean)
        return contents_clean, all_words

    # 获取tdidf
    def get_idf(self):
        df_news = self.e
        content = df_news.content.values.tolist()  # 因为jieba要列表格式
        content_S = []  # 存储分完词之后结果
        for line in content:
            current_segment = jieba.lcut(line)  # jieba分词
            if len(current_segment) > 1 and current_segment != "\r\n":
                content_S.append(current_segment)
        # 将分完词的结果转化成DataFrame格式
        df_content = pd.DataFrame({"content_S": content_S})
        stopwords = self.f
        # 调用去除停用词函数
        contents = df_content.content_S.values.tolist()
        stopwords = stopwords.stopword.values.tolist()
        contents_clean, all_words = Get_tag.drop_stopwords(self,contents,stopwords)
       # 将清洗完的数据结果转化成DataFrame格式
        df_content = pd.DataFrame({"contents_clean": contents_clean})
        # c来储存清洗后的文章内容
        c = []
        for line in df_content['contents_clean']:
            c.append("".join(line))
        # 分词上面已经转化成列表格式了直接用
        contentd = c  # 因为jieba要列表格式上面部分已经做好了
        content_D = []  # 存储分完词之后结果
        for line in contentd:
            current_segment = jieba.lcut(line)  # jieba分词
            if len(current_segment) > 1 and current_segment != "\r\n":
                content_D.append(current_segment)
        # 将分完词的结果转化成DataFrame格式
        for line in contentd:
            top_n = 150
            keywords = jieba.analyse.extract_tags(line, topK=top_n, withWeight=True, allowPOS=('ns', 'n'))
            list = []
            for i in range(len(keywords)):
                dict = {'tfidf': keywords[i][1], 'word': keywords[i][0]}
                list.append(dict)
            return list
            print(list)

        # return list
        # print(list)


if __name__ == "__main__":
    a = pd.read_table('/Users/yaolei/学习资料和公司/虎嗅和练习/虎嗅/文本分类和提取标签/ceshi_con.csv', names=['content'], encoding='utf-8')
    a = a.dropna()  # 直接丢弃包括NAN的整条数据
    b = pd.read_csv('/Users/yaolei/学习资料和公司/虎嗅和练习/虎嗅/文本分类和提取标签/data/stopwords.txt', index_col=False, sep='\t',
                    quoting=3, names=["stopword"],
                    encoding="utf-8")  # 读入停用词
    print('-----------')
    g= Get_tag(a,b)
    print(g.get_idf())




