from confluent_kafka import Producer, KafkaException
import json

class KafkaConsumerWrapper:
    def __init__(self, broker, group_id, topic):
        self.conf = {
            'bootstrap.servers': broker,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        }
        self.producer = Producer(self.conf)
        self.producer.subscribe([topic])

    def publish(self):
        
        msg = self.consumer.poll(0.1)

        if msg is None:
            return None
        try:
            unprocessed_message = json.loads(msg.value().decode('utf-8'))
        except ValueError as e:
            print(e)
            return None

        processed_message = self.process_message(unprocessed_message)
        return processed_message

    def close(self):
        self.producer.close()

class KafkaDetectionConsumer(KafkaConsumerWrapper):
    def procuce_message(self):
    
        return None
