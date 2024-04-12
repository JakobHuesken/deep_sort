import sys
sys.path.append("/usr/local/src/git/tkDNN/deep_sort")
from application_util.kafka_producer import KafkaCalibrationProducer
import json
import os
import configparser

kafka_producer = KafkaCalibrationProducer("141.58.8.236:9092")

# Read in matrix
matrix_data = []
with open("/usr/local/src/git/tkDNN/build/calibrationInvHmat.txt", "r") as file:
    for line in file:
        matrix_data.append(line.strip())
# Read in cam_id
config_file = "/usr/local/src/git/tkDNN/config_visagx-brio.ini"
config = configparser.ConfigParser()
config.read(config_file)
cam_id = config.getint("tkdnn", "cam_id")

message = {"cam_id": cam_id, "matrix_data": matrix_data}
json_message = json.dumps(message)
kafka_producer.publish("calibration", json_message)
kafka_producer.producer.flush(300)
