# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from application_util.kafka_consumer import KafkaDetectionConsumer
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.multicam_matching import synchronize_frames, filter_out_detections, group_detections, transform_coordinates


def gather_sequence_info(kafka_consumer, batch_size, calibration_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    batchFrames = []
    first_message = True
    last_frame_number = 0
    for i in range(batch_size):
        frame, first_message = kafka_consumer.update(first_message)
        if frame is None: 
            continue
        if isinstance(frame, dict):
            batchFrames.append(frame)

    if batchFrames:
        # Start by synchronizing the frames from different cams
        print("unsynchronized V")
        print(batchFrames)
        print("unsynchronized A")
        # synchronize the frames
        batchFrames, last_frame_number = synchronize_frames(batchFrames, last_frame_number, 30)
        print("synchronized V")
        print(batchFrames)
        print("synchronized A")
        print("transformed V")
        # Now transform the coordinates into global coordinates
        batchFrames = transform_coordinates(batchFrames, calibration_dir)
        print("transformed A")
        print("grouped V")
        batchFrames = group_detections(batchFrames)
        print(batchFrames)
        print("grouped A")
        # # Lastly filter out the same detection comming from different cams
        print("filtered V")
        batchFrames = filter_out_detections(batchFrames)
        print(batchFrames)
        print("filtered A")
    else: 
        print("No frames gathered")
    # Deconstruct dictionary into numpy array
    detections_list = []
    for frame in batchFrames:
        frame_id = frame["frame_id"]
        detections = frame["detections"]

        for detection in detections:
            bbox_x = detection["bbox_x"]
            bbox_y = detection["bbox_y"]
            bbox_w = detection["bbox_w"]
            bbox_h = detection["bbox_h"]
            probability = detection["probability"]
            features = detection["features"]
            detection_data = [frame_id, 0, bbox_x, bbox_y, bbox_w, bbox_h, probability, -1, -1, -1] + features
            detections_list.append(detection_data)
    detections = np.array(detections_list)   

    seq_info = {
        "sequence_name": "MultiCamera",
        "detections": detections,
        "groundtruth": None,
        "image_size": [1080,1920],
        "min_frame_idx": 0,
        "max_frame_idx": batchFrames[-1]["frame_id"],
        "feature_dim": 128,
        "update_ms": 100
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, batch_size, calibration_dir):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    # Kafka
    broker = "141.58.8.236:9092"
    group_id = "1"
    topic = "timed-images"
    kafka_consumer = KafkaDetectionConsumer(broker, group_id, topic) 

    seq_info = gather_sequence_info(kafka_consumer, batch_size, calibration_dir)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    print("run")
    def frame_callback(vis, frame_idx):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=100)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--batch_size", help="Run the tracker on batch_size increments of frames, effect of different values are not yet discovered",
        type=int, default=400)
    parser.add_argument(
        "--calibration_dir", help="Path to the calibration dir.",
        default="resources/calibration")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display, args.batch_size, args.calibration_dir)
