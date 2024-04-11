import sys
sys.path.append("/usr/local/src/git/tkDNN/deep_sort")
from application_util.kafka_producer import KafkaDetectionProducer
import json
import os

with open("resources/detections/test/Aufnahme_2024-04-11_10-06-38.json", "r") as json_file:
    all_frames = json.load(json_file)

kafka_producer = KafkaDetectionProducer("localhost:9092")

for frame in all_frames:
    frame_json = json.dumps(frame)
    kafka_producer.publish("timed-images", frame_json)
kafka_producer.producer.flush(300)