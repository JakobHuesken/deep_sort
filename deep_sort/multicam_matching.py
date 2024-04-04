import numpy as np
from deep_sort.nn_matching import NearestNeighborDistanceMetric
import copy

def _nn_euclidean_distance(x, y):
    x, y = np.atleast_2d(x), np.atleast_2d(y)  # Ensure x and y are 2D arrays
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))



def _pdist(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)  # Ensure numeric data types
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

def synchronize_frames(batchFrames, last_frame_number=None, time_threshold=30): # In milliseconds
    # Sort frames by timestamp
    sorted_frames = sorted(batchFrames, key=lambda x: x['timestamp'])
    
    # Initialize frame_id and last_timestamp
    if last_frame_number is None:
        frame_id = 1
    else:
        frame_id = last_frame_number + 1
    
    last_timestamp = None
    
    # Iterate through frames
    for frame in sorted_frames:
        if last_timestamp is None:
            last_timestamp = frame['timestamp']
        elif frame['timestamp'] - last_timestamp > time_threshold:
            frame_id += 1
        frame['frame_id'] = frame_id
        last_timestamp = frame['timestamp']
    
    # Return synchronized frames and updated last frame number
    return sorted_frames, frame_id


def transform_coordinates(frames, homography_dir):
    for frame in frames:
        cam_id = frame["cam_id"]
        homography_file = f"{homography_dir}/homography_{cam_id}.npy"
        invHmat = np.load(homography_file)

        for detection in frame["detections"]:
            bbox_x, bbox_y, bbox_w, bbox_h = detection["bbox_x"], detection["bbox_y"], detection["bbox_w"], detection["bbox_h"]
            bbox_points = np.array([[bbox_x, bbox_y, 1], 
                                    [bbox_x + bbox_w, bbox_y, 1], 
                                    [bbox_x, bbox_y + bbox_h, 1], 
                                    [bbox_x + bbox_w, bbox_y + bbox_h, 1]])

            transformed_bbox = np.dot(invHmat, bbox_points.T).T
            transformed_bbox[:, 0] /= transformed_bbox[:, 2]
            transformed_bbox[:, 1] /= transformed_bbox[:, 2]

            detection["bbox_x"] = transformed_bbox[0, 0]
            detection["bbox_y"] = transformed_bbox[0, 1]
            detection["bbox_w"] = transformed_bbox[1, 0] - transformed_bbox[0, 0]
            detection["bbox_h"] = transformed_bbox[2, 1] - transformed_bbox[0, 1]

    return frames

def group_detections(frames):
    grouped_detections = []  # List to hold grouped detections by frame ID

    # Iterate over each frame
    for frame in frames:
        frame_id = frame['frame_id']
        detections = frame['detections']

        # Remove cam_id and timestamp from the frame dictionary
        #frame_without_metadata = frame.copy()
        #del frame_without_metadata['cam_id']
        #del frame_without_metadata['timestamp']

        # Check if frame_id already exists in grouped_detections
        found = False
        for entry in grouped_detections:
            if entry['frame_id'] == frame_id:
                found = True
                # Merge detections for the same frame_id
                entry['detections'].extend(detections)
                break

        # If frame_id is not found, add it to grouped_detections
        if not found:
            grouped_detections.append({'frame_id': frame_id, 'detections': detections})

    return grouped_detections

def filter_out_detections(grouped_frames, threshold=0.2, metric="euclidean"):
    """
    Filter out duplicate detections within each grouped frame based on a threshold.

    Parameters:
    -----------
    grouped_frames : list
        A list of dictionaries, where each dictionary represents a grouped frame
        containing the key 'frame_id' and 'detections' holding a list of detections.
    threshold : float, optional
        The threshold for considering two detections as duplicates. Default is 0.2.
    metric : str, optional
        The distance metric to use for comparing features. Currently, only "euclidean"
        metric is supported.

    Returns:
    --------
    filtered_frames : list
        A list of dictionaries with duplicates removed within each frame.
    """
    filtered_frames = []

    distances = NearestNeighborDistanceMetricCustom(metric, threshold)

    for grouped_frame in grouped_frames:
        frame_id = grouped_frame['frame_id']
        detections = grouped_frame['detections']

        # Filter out duplicate detections based on feature similarity
        filtered_detections = []
        for detection in detections:
            if not filtered_detections:
                filtered_detections.append(detection)
            else:
                cost_vector = distances.distance(filtered_detections, detection)
                if np.min(cost_vector) > threshold:
                    filtered_detections.append(detection)

        # Update filtered frame
        filtered_frame = {'frame_id': frame_id, 'detections': filtered_detections}
        filtered_frames.append(filtered_frame)

    return filtered_frames

class NearestNeighborDistanceMetricCustom(NearestNeighborDistanceMetric):
    """
    A modified version of the NearestNeighborDistanceMetric class,
    tailored to fit the different data structure and general usage.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):
        # Copying the original NearestNeighborDistanceMetric initialization
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        else:
            raise ValueError(
                "Invalid metric; must be 'euclidean'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, detections):
        """Update the distance metric with new data.

        Parameters
        ----------
        detections : List[Dict]
            A list of dictionaries, each containing detections for a single frame.

        """
        for frame in detections:
            frame_id = frame['frame_id']
            frame_detections = frame['detections']
            for detection in frame_detections:
                features = detection['features']
                target = frame_id  # Use frame_id as a target
                self.samples.setdefault(target, []).append(features)
                if self.budget is not None:
                    self.samples[target] = self.samples[target][-self.budget:]

    def distance(self, filtered_detections, new_detection):
        num_detections = len(filtered_detections)
        cost_vector = np.zeros(num_detections)

        new_features = np.array(new_detection['features'])  # Convert to numpy array

        for i in range(num_detections):
            features_i = np.array(filtered_detections[i]['features'])  # Convert to numpy array
            cost_vector[i] = self._metric(features_i, new_features)

        return cost_vector