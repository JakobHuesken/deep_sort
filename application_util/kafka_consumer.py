from confluent_kafka import Consumer, KafkaException
import json
import time

class TimeoutError(Exception):
    pass

class KafkaConsumerWrapper:
    def __init__(self, broker, group_id, topic):
        self.conf = {
            'bootstrap.servers': broker,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        }
        self.consumer = Consumer(self.conf)
        self.consumer.subscribe([topic])

    def update(self, first_time, timeout=1):
        start_time = time.time()
        msg = None
        
        while True:
            if first_time is False:
                if time.time() - start_time > timeout:
                    print("Timeout while waiting for message")
                    break
            msg = self.consumer.poll(0.1)
            print("test")
            if msg is not None:
                first_time = False
                try:
                    unprocessed_message = json.loads(msg.value().decode('utf-8'))
                except ValueError as e:
                    print(e)
                    return None, None
                processed_message = self.process_message(unprocessed_message)
                return processed_message, False
        return None, False

    def process_message(self, json_data):

        print("Received JSON data:", json_data)
        return 0

    def close(self):
        self.consumer.close()

class KafkaDetectionConsumer(KafkaConsumerWrapper):
    def process_message(self, json_data):
        # The detections are not yet in a format supported by deep_sort
        # This function returns them to the right data types and format

        for key, value in json_data.items():
            if isinstance(value, str):
                # Convert string representation of integer to integer
                if value.isdigit():
                    json_data[key] = int(value)
            elif isinstance(value, list):
                # Iterate over each dictionary in the list
                for item in value:
                    # Convert values to integers, except for 'probability' key
                    for k, v in item.items():
                        if isinstance(v, str):
                            if v.isdigit():
                                item[k] = int(v)
                            elif k == 'probability':
                                item[k] = float(v)
                        elif isinstance(v, list):
                            item[k] = [int(x) if isinstance(x, str) and x.isdigit() else x for x in v]

        # print(json_data)
        return json_data

class KafkaCalibrationConsumer(KafkaConsumerWrapper):
    def process_message(self, json_data):
        print("Received JSON data:", json_data)
        return json_data
    def update(self):
        
        msg = None
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
