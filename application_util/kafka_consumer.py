from confluent_kafka import Consumer, KafkaException
import json

class KafkaConsumer:
    def __init__(self, broker, group_id):
        self.broker = broker
        self.group_id = group_id
        self.consumer = None
    
    def start_consumer(self.topic):
        self.topic = topic
        conf = {
            'bootstrap.servers': self.broker,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        }

        self.consumer = Consumer(conf)
        self.consumer.subscribe([self.topic])

    def update(batchSize):
        
        parsed_detections = []
        for frame in batchSize:
            msg = self.consumer.poll(1.0)

            if msg is None:
                return 0
            if msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF:
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                        (msg.topic(), msg.partition(), msg.offset()))
                else:
                    sys.stderr.write('Consumer error: %s\n' % msg.error())
                return 0

            value = msg.value().decode('utf-8')
            json_data = json.loads(value)
            for detection in json_data:
                parsed_detections.append({
                    "frame_id": detection["f"],
                    "detection_class": detection["c"],
                    "bbox_x": detection["bX"],
                    "bbox_y": detection["bY"],
                    "bbox_w": detection["bW"],
                    "bbox_h": detection["bH"],
                    "probability": detection["p"],
                    "global_x": detection["gX"],
                    "global_y": detection["gY"],
                    "global_z": detection["gZ"],
                    "features": detection["features"]
                })
        
        return parsed_detections