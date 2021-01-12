# -*- coding:utf-8 -*-

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
import os
import random
import jieba
import jieba.analyse

class Get_tag():
    def TextProcessing(self,words,top_n):
        # word_cut = jieba.cut(words, cut_all=False)  # 精简模式，返回一个可迭代的generator
        # word_list = list(word_cut)  # generator转换为list
        # return word_list
        # seg = jieba.cut_for_search(words)  # 搜索引擎模式
        # seg_list = list(seg)
        #
        # print(", ".join(seg_list))
        # return seg_list
        keywords = jieba.analyse.extract_tags(words, topK=top_n, withWeight=True, allowPOS=('n', 'nr', 'ns', 'nz', 'd', 'nt', 'v', 'vn', 'i'))
        # print(keywords)
        # print(type(keywords))
        # <class 'list'>
        item_list = []
        for item in keywords:
            # print(item)
            # print(item[0], item[1])
            all_item = {}
            all_item['words'] = item[0]
            all_item['weight'] = item[1]
            item_list.append(all_item)
        return item_list


    def MakeWordsSet(self,words_file):
        words_set = set()                                            # 创建set集合
        with open(words_file, 'r', encoding='utf-8') as f:        # 打开文件
            for line in f.readlines():                                # 一行一行读取
                word = line.strip()                                    # 去回车
                if len(word) > 0:                                    # 有文本，则添加到words_set中
                    words_set.add(word)
        return words_set                                             # 返回处理结果


    def words_dict(self,all_words_list,deleteN,stopwords_set=set()):
        feature_words = []   # 特征列表
        n = 1
        for t in range(deleteN, len(all_words_list), 1):
            if n > 1000:  # feature_words的维度为1000
                break
            # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
            if not all_words_list[t]['words'].isdigit() and all_words_list[t]['words'] not in stopwords_set and 1 < len(all_words_list[t]['words']) < 5:
                feature_words.append(all_words_list[t])
            n += 1
        # print(feature_words)
        return feature_words


if __name__ == '__main__':
    # 文本预处理
    # folder_path = './SogouC/Sample'                #训练集存放地址
    # all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    reuqest_data = {
        "top_n": 150,
        "title": '中美AI行业发展殊途，中国独角兽命运独立',
        "content": '导语：阿尔法狗退役了，AI可没有走，很多人没有意识到的是，AI早已被植入到诸多生活场景之中，在中国，这种植入和落地是由一群新锐的创业公司和独角兽完成的。更深入这个行业来看，中国的AI行业格局是：BAT中B在搭建平台，A和T开始布局，而中小公司以及独角兽们在这个窗口期内完成了场景落地，和美国形成了完全不一样的局面。独角兽深耕AI细分领域“跟AlphaGo下棋我特别痛苦，他实在太冷静了，没有一丝获胜的希望”，在三连败给对手之后，柯洁这样形容这场对弈，阿尔法狗是自去年以来最火爆的AI概念。但对大多数人来说，陪柯洁一起伤感过后，AI好像又变成一个对他们无关痛痒的概念了，毕竟遥不可及。很多人没有意识到的是，和阿尔法狗更重要的意义在于提示我们机器学习能力进步到了何种形态，以及AI已经深入到生活的各个场景。一个简单的例子就是，女孩子们玩的Faceu、小咖秀，对准自己脸之后，镜头里的自己长出“长舌头”、流出“眼泪”来，这其中就涉及到人脸识别，而这也是人工智能的一种。又比如微博的相册新功能可检测出图片中的面孔，并且能按不同人脸分类归纳，甚至处理图片中的暗光以及雾气，还原出清晰的图片。小米手机的“宝宝相册”、“一人一相册”功能和云端存储照片自动分类功能……这些都涉及图像识别。提供这些功能的是一家叫“商汤科技”的AI公司，按照融资额和估值来说，它已经是一家独角兽，除了Faceu、小咖秀以及微博，还有华为、OPPO、中国移动等都在使用它的技术。如果从行业来看这种无处不在的场景应用，它意味着，AI浪潮已经进入到实际落地的新阶段，而不仅仅是前沿研究上。“没有场景支持的AI研究是空中楼阁。”腾讯集团董事长马化腾在最近的演讲中这样说。而率先把这些技术落地的并非马化腾的腾讯或者是其它几家大公司，而是中小创业公司。就以商汤科技所在的计算机视觉领域而言，还有其它三家公司，包括“格灵深瞳”、旷世科技（Face++）以及依图科技，称为这个领域的四小龙，把人工智能技术推广到娱乐、安防、零售等各个领域。商汤科技等公司的表现足以回答一直以来AI创业面临的一个问题：在大公司包括谷歌、微软以及Facebook和Uber，以及国内百度等公司全部投身这一行业时，创业公司机会何在？“在细分领域做到极致”，YC合伙人Anu Hariharan曾这样给出策略。而商汤的表现也的确在验证这一路径。Anu Hariharan说，“就以图像识别来说，业界识别率到达98%以上的水平，最后一两分的差距能够决定无人车是否能够上路或者别的什么场景的应用的可能性，而只有专注于一个细分领域，才能够越过这两分的门槛”。以商汤科技而言，就是专注于人脸识别。这家公司的联合创始人、CEO徐立博士介绍说，他们依靠自己的核心算法，在2014年把人脸识别准确率做的超过普通团队。目前商汤的商业模式总结起来：是一家拥有原创技术的toB的算法公司，为toC的公司提供API和SDK。就在5月26日，商汤科技宣布与金山云达成合作，把自己的SenseAR增强现实感特效引擎集成到金山云的视频生态新平台上去。由明星转型成为投资人的任泉在2016年4月也投资了商汤科技，他透露说，包括商汤在内，目前计算机视觉领域的AI公司收入很大部分来自安防市场，中国这两年在安防市场的巨大投入也让这些公司未来收入可期。商汤自己提供的一个案例显示，比如在安防领域，在与重庆公安的合作中，商汤科技只用了40天就识别了69个犯罪嫌疑人、抓捕15人，而按照以前的技侦手段，可能一年也就抓几个人，现在一个月把公安系统十年的工作完成。中等公司和BAT划分天下有趣的是，中国AI行业和美国发展出了完全不同的格局。“相比美国，中国这些中等创业公司更有可能获得独立发展的机会”，另一名观察中美两地AI行业的认为，相比美国AI中小公司积极地被收购，中国这些中小公司命运更可能独立。美国市场的AI创业和收购上，整体走在中国市场之前。据市场研究机构CB Insights报告显示，今年第一季度共有34家人工智能创业公司被收购，是去年同期的两倍。CB Insights称，谷歌自2012年以来共收购了11家人工智能创业公司，是所有科技巨头收购最多的公司，苹果、Facebook和英特尔分别排名第二、第三和第四。“中国能够买得起这几家公司的并不多”，他说，而这些公司目前依靠出售自己的SDK已经站稳了市场，“我认为未来更可能走向独立上市，而不是被收购”。商汤科技也更希望保持一个独立的命运，“之前，我记得一年多前有一篇很有意思的报道，说的不管人脸识别哪家做的好的，最后你的我的都是BAT的”，商汤科技联合创始人、CEO徐立博士认为这种判断适用于技术平稳期，“当技术演进没那么快，大公司可以通过资源优势加上次优的技术快速切入，那真的就是你的我的，都是BAT的”。不过，现在不一样了。技术的迅速发展能够让这些公司技术门槛越来越高，“但当技术快速发展的时候，技术发展带来足够时间窗口形成不同壁垒。深度学习带来每年性能的提升都可以抵过深度学习之前的数年甚至十年。当然，行业爆发的前提条件是技术必须过红线，也就是超越普通人的准确率。”那么BAT做什么？目前而言，在国内巨头公司中百度是在AI中布局较早的，这家公司现在也确立了做平台的策略，百度目前在语音识别上投入较大，在有了初期的成果后，希望以语音助手为入口，做自己的平台。麦肯锡近日针对中国的人工智能行业发表了一份报告，它列出了目前在人工智能有影响力的几家公司的应用和商业化场景：如下图所示，能够有所影响的中国公司只有百度一家，而且主要在自动驾驶领域。而其他家包括A和T以及京东等公司，在这个领域的动作都较为滞后，这也是留给中小公司的创业窗口。当然，一些创业公司已经开始站队，比如阿里入股旷视科技，“通过入股的公司，BAT可能会让这些公司站队，但站队也意味着有所限制”，西雅图的一名AI创业者这样看待中国目前的AI格局。麦肯锡对于中国的建议是，扩大在传统行业里面的AI应用。这也意味着，那些在细分领域超过普通水准的公司能够赢得自己的机会。落地之后进一步提高技术差在中国创业再也不是依靠商业模式创新够脱颖而出的，AI时代的到来拉高了创业的门槛。以商汤为例，拥有自主研发的深度学习超算平台，根据徐立介绍，商汤创始团队成员从11年开始做深度学习，但是当时也没有什么好的开源算法可用，现在谷歌所推向市场的Tensorflow以及其他主流的Caffe、Torch都还没有，只能从头开始。做以深度学习为基础的人工智能团队都要去一个名为ImageNet大赛一较高低，2012年，Hinton团队在ImageNet首次使用深度学习完胜其它团队，也正是这次成果，让人们意识到深度学习相比于传统机器学习的长处，让深度学习成为重新回到主流技术舞台；在2014年的时候，Google做了22层成为冠军，深度明显提升了。2015年是来自微软的ResNet做到152层；2016年商汤做到1207层，又得到了一个突破。深度每增加一层，其表达能力都有一个实质性的突破。徐立认为商汤立足的核心在于领先的原创技术，现在的人脸识别技术，还没有饱和到任何人拿了一个现有的东西调一下参数就能够用的。 “为什么我们比较坚持有一些自主研发的平台性的东西，是因为我觉得这个才能对未来的行业有所推动，才能真正地 push the envelope，因为你能做别人不能做的事情。”创业门槛提高的同时，那些有先发优势的创业公司和其他公司技术也在拉大，你可以这样理解以深度学习、神经网络为核心的人工智能技术，基础的算法就像是刚刚生下来的婴儿，而数据则是喂大婴儿的粮食。像商汤科技这样的独角兽公司在多个场景中能够落地的，也意味着不断能够收集重要的数据，而用这些数据去“喂养”他们的算法，能够进一步拉大和其他公司的差距。“如果能够拿到核心的细分领域数据，而且由于处在中国这个劳动力相对便宜的市场，找到便宜的人力去标注数据，这些独角兽公司还是很有机会的。”一名西雅图创业者对商汤科技也有过长期的观察，她的图像识别公司原本和商汤做类似的业务，在2016年被亚马逊收购，她这样看待商汤未来的机会。在麦肯锡的报告中，2015年中国数据总量占全球数据总量的13%，据预测，到2020年中国的数据总量占全球数据总量的比例将达到20%，届时中国将成为世界第一数据资源大国和全球的数据中心。细分领域的有活力的公司能够收集而且整理好这些数据，对于中国各个领域的AI化都会有极大的帮助，而这些公司，也能够获得成长的机会。'
    }
    # response_list = tfidf_model.get_tfidf(reuqest_data["title"], reuqest_data["content"], reuqest_data["top_n"])
    # logger.info(response_list)

    g=Get_tag()
    all_words_list = g.TextProcessing(reuqest_data["content"],reuqest_data["top_n"])
    # 生成stopwords_set
    stopwords_file = './stopwords.txt'
    stopwords_set = g.MakeWordsSet(stopwords_file)
    feature_words = g.words_dict(all_words_list, 2, stopwords_set)
    print(feature_words)