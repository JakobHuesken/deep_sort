import sys
sys.path.append("/usr/local/src/git/tkDNN/deep_sort")
from application_util.kafka_producer import KafkaDetectionProducer
import json
import os

with open("resources/detections/test/Aufnahme_2024-04-12_13-26-39.json", "r") as json_file:
    all_frames = json.load(json_file)

kafka_producer = KafkaDetectionProducer("141.58.8.236:9092")

for frame in all_frames:
    frame_json = json.dumps(frame)
    kafka_producer.publish("timed-images", frame_json)
kafka_producer.producer.flush(300)