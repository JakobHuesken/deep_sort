import pandas as pd

def load_detections(file_path):
    # Read CSV file containing detections
    df = pd.read_csv(file_path, names=['frame_id', 'object_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h','notneeded','notneeded2','notneeded3','notneeded4'])
    return df

def count_multiple_objects_per_frame(detections):
    # Count the number of frames with multiple objects
    frames_with_multiple_objects = detections.groupby('frame_id')['object_id'].nunique().gt(1).sum()
    return frames_with_multiple_objects

def count_object_id_switches(detections):
    # Initialize object ID switches count
    object_id_switches = 0

    # Initialize previous frame ID and object IDs
    prev_frame_id = 0
    prev_object_ids = []
    current_object_ids = [1]

    # Iterate over detections
    for _, detection in detections.iterrows():
        frame_id = detection['frame_id']
        object_id = detection['object_id']

        # Check if frame ID has changed
        if frame_id != prev_frame_id:
            prev_frame_id = frame_id
            prev_object_ids = current_object_ids
            current_object_ids = []
            current_object_ids.append(object_id)
            # Compare the current object_id to all previous object_ids
            if object_id not in prev_object_ids:
                object_id_switches += 1
                print(f"Frame {frame_id}: New Object ID {object_id}")
            
        else:
            current_object_ids.append(object_id)
            if object_id not in prev_object_ids:
                object_id_switches += 1
                print(f"Frame {frame_id}: New Object ID {object_id}")

    return object_id_switches


# File paths
combined_file = "/home/visadmin/Desktop/Datenerhebung_DeepSort/Tests/15_04_Auseinander_30Grad/15_04_EP/Ergebnisse/hypotheses_test_15.04.24_eineperson_0.2.txt"
cam1_file = "/home/visadmin/Desktop/Datenerhebung_DeepSort/Tests/15_04_Auseinander_30Grad/15_04_EP/Ergebnisse/hypotheses_test_15.04.24_eineperson_cam1.txt"
cam2_file = "/home/visadmin/Desktop/Datenerhebung_DeepSort/Tests/15_04_Auseinander_30Grad/15_04_EP/Ergebnisse/hypotheses_test_15.04.24_eineperson_cam2.txt"

# Load detections
cam1_detections = load_detections(cam1_file)
cam2_detections = load_detections(cam2_file)
combined_detections = load_detections(combined_file)

# Count multiple objects per frame
cam1_multiple_objects = count_multiple_objects_per_frame(cam1_detections)
cam2_multiple_objects = count_multiple_objects_per_frame(cam2_detections)
combined_multiple_objects = count_multiple_objects_per_frame(combined_detections)

# Count object ID switches
cam1_id_switches = count_object_id_switches(cam1_detections)
cam2_id_switches = count_object_id_switches(cam2_detections)
combined_id_switches = count_object_id_switches(combined_detections)

print("Camera 1:")
print("Frames with Multiple Objects:", cam1_multiple_objects)
print("Object ID Switches:", cam1_id_switches)

print("\nCamera 2:")
print("Frames with Multiple Objects:", cam2_multiple_objects)
print("Object ID Switches:", cam2_id_switches)

print("\nCombined Detections:")
print("Frames with Multiple Objects:", combined_multiple_objects)
print("Object ID Switches:", combined_id_switches)
