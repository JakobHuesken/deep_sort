import os
import numpy as np
import sys
sys.path.append("/usr/local/src/git/tkDNN/deep_sort")
from application_util.kafka_consumer import KafkaCalibrationConsumer

def save_calibration_matrix(cam_id, matrix_data):
    # Define the directory to save the calibration matrices
    directory = os.path.join(os.path.dirname(__file__), '..', 'resources', 'calibration')
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the file path for saving the calibration matrix
    file_path = os.path.join(directory, f'calibration_matrix_{cam_id}.npy')

    # Save the calibration matrix as a numpy array
    print(matrix_data)
    np.save(file_path, matrix_data)
    print(f"Calibration matrix for cam_id {cam_id} saved successfully.")

def main():
    broker = 'localhost:9092'
    group_id = '0'
    topic = 'calibration'

    # Create a KafkaCalibrationConsumer instance
    kafka_consumer = KafkaCalibrationConsumer(broker, group_id, topic)

    try:
        while True:
            # Update and process messages from Kafka
            message = kafka_consumer.update()
            if message is not None:
                cam_id = message['cam_id']
                matrix_data = np.array([np.fromstring(row, sep=' ') for row in message['matrix_data']])
                print(matrix_data.shape)
                save_calibration_matrix(cam_id, matrix_data)
    except KeyboardInterrupt:
        pass
    finally:
        # Close the consumer
        kafka_consumer.close()
if __name__ == "__main__":
    main()