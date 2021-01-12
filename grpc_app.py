import os
import sys
import time
import concurrent.futures
import multiprocessing
import configparser
import grpc

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(APP_DIR)
sys.path.append(os.path.join(APP_DIR,"grpc_file"))

from utils.log_util import logger

from grpc_file import side_feature_pb2_grpc
from service.side_feature_service import SideFeatureService

config_ini_dict = configparser.ConfigParser()
config_ini_dict.read(os.path.join(APP_DIR,"config.ini"))
logger.info(config_ini_dict)

if __name__ == '__main__':
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*5))

    side_feature_pb2_grpc.add_side_feature_serviceServicer_to_server(SideFeatureService(), server)

    server.add_insecure_port("[::]:{}".format(config_ini_dict["SERVER"]["PORT"]))
    server.start()
    logger.info("server start...")
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
