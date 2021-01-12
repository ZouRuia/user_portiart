

import re
re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-\*]+)", re.U)

sentence = "【美股开盘：美股反弹 中概股京东涨超6%】*ST万达美股集体高开，道指涨250点；京东涨超6%，2019年第四季度净利润超预期；特斯拉涨6.5%，报道称Model Y即将开始交付。"

blocks = re_han.split(sentence)
for blk in blocks:
    print(blk)
