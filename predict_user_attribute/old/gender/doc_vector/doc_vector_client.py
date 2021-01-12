# coding=utf-8
import grpc
from doc_vector import doc_vector_pb2, doc_vector_pb2_grpc
from retrying import retry
from configure.ConfigureUtils import *
import configure.constans as ct


class DocVectorClient(object):

    def __init__(self):
        self.ip_port = get_configure_utils().get(ct.grpc_service_ip_port)

    @retry(stop_max_attempt_number=5)
    def avg_vector(self, words):
        # 连接 rpc 服务器
        channel = grpc.insecure_channel(self.ip_port)
        # 调用 rpc 服务
        stub = doc_vector_pb2_grpc.doc_vectorStub(channel)
        response = stub.avg_vector(doc_vector_pb2.doc_vector_request(words=words))
        return response.feature_vec


if __name__ == '__main__':
    init_configure_utils(model="development")
    doc_vector = DocVectorClient()
    # feature_vec = doc_vector.avg_vector(words=['多地', '加码', '停车', '设施', '规划', '智慧', '停车', '迎风', '口', '近期', '山东', '青岛市', '发布', '进一步', '加强', '停车', '设施', '规划', '建设', '管理工作', '实施', '意见', '提出', '工作', '目标', '全市', '经营性', '停车场', '纳入', '智能', '停车', '一体化', '平台', '报告', '预测', '全国', '停车位', '数量', '1.19', '亿个', '汽车', '保有量', '五年', '复合', '增速', '持续增长', '全国', '民用', '汽车', '保有量', '2.9', '亿辆', '停车位', '比例', '仅为', '1', '0.4', '配比', '偏低', '国家发改委', '城市', '小城镇', '改革', '发展', '中心', '理事会', '理事长', '李铁', '未来', '智慧', '停车', '一线', '城市', '二三', '四线', '城市', '持续', '渗透', '智慧', '停车', '产业', '发展', '再度', '加速'])
    feature_vec = doc_vector.avg_vector(words=['共享', '街机', '街机', '超人', '宣布', '公司', '完成', '新一轮', '战略', '融资', '上海', '典商共策', '投资', '领投', '股东', '追加', '投资', '此轮', '融资', '创始人', '兼', 'CEO', '贠垚韬', '资金', '用于', '团队', '建设', '服务', '能力', '提升', '加大', '技术', '投入']
)
    print(len(feature_vec))
    vector = ",".join([str(x) for x in feature_vec])
    print(vector)


