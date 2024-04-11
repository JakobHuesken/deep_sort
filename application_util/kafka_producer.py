from confluent_kafka import Producer, KafkaException
import json

class KafkaConsumerWrapper:
    def __init__(self, broker):
        self.conf = {
            'bootstrap.servers': broker
        }
        self.producer = Producer(self.conf)

    def publish(self):
        self.producer.produce()
        return None

    def close(self):
        self.producer.close()

class KafkaDetectionProducer(KafkaConsumerWrapper):
    def publish(self, topic, message):
        self.producer.produce(topic, message.encode('utf-8'))
        return None

class KafkaCalibrationProducer(KafkaConsumerWrapper):
    def publish(self, topic, message):
        self.producer.produce(topic, message.encode('utf-8'))
        return None

