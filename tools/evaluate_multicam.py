import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
                # print(f"Frame {frame_id}: New Object ID {object_id}")
            
        else:
            current_object_ids.append(object_id)
            if object_id not in prev_object_ids:
                object_id_switches += 1
                # print(f"Frame {frame_id}: New Object ID {object_id}")

    return object_id_switches

def multiple_objects_proportion(detections, multiple_objects):
    total_frames = detections['frame_id'].nunique()
    proportion = multiple_objects / total_frames
    return proportion

def visualize_bounding_boxes(detections, directory, file_path, padding = 100):
    # Calculate middle point of the bottom line of each bounding box
    detections['bbox_middle_x'] = detections['bbox_x'] + detections['bbox_w'] / 2
    detections['bbox_middle_y'] = detections['bbox_y'] + detections['bbox_h']

    # Calculate plot limits with padding
    # x_min = detections['bbox_middle_x'].min() - padding
    # x_max = detections['bbox_middle_x'].max() + padding
    y_min = detections['bbox_middle_y'].min() - padding
    y_max = detections['bbox_middle_y'].max() + padding
    
    # Extract relative path from the file path and remove extension
    relative_path = os.path.relpath(file_path, directory)
    relative_path = os.path.splitext(relative_path)[0]
    
    relative_path = relative_path.replace("/", "_")
    save_path = os.path.join(directory, f"{relative_path}.png")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))  # Adjust figsize as needed
    # Define colormap from yellow to blue
    cmap = plt.cm.viridis
    # Normalize the colors to the range of values
    norm = mcolors.Normalize(vmin=0, vmax=len(detections))
    # Plot each point with gradually changing colors
    for i, (x, y) in enumerate(zip(detections['bbox_middle_x'], detections['bbox_middle_y'])):
        color = cmap(norm(i))
        ax.scatter(x, y, color=color, marker='o')   
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(relative_path)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal

    # Set plot limits
    # ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
   
    def on_key_press(event):
        if event.key == 'escape':
            plt.close(fig)
        elif event.key == 'y':
            plt.savefig(save_path)
            print("Plot saved as:" + save_path)

    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()

def process_files(directory):
    # Find all .txt files starting with "hypotheses" in the specified directory and its subdirectories
    file_paths = glob.glob(os.path.join(directory, '**', 'hypotheses*.txt'), recursive=True)
    
    # Initialize results string
    results_str = ""
    
    # Process each file
    for file_path in file_paths:
        # Load detections
        detections = load_detections(file_path)
        
        # Count multiple objects per frame
        multiple_objects = count_multiple_objects_per_frame(detections)
        
        # Calculate proportion of frames with multiple detections relative to total frames
        proportion = multiple_objects_proportion(detections, multiple_objects)
        
        # Count object ID switches
        id_switches = count_object_id_switches(detections)

        # visualize_bounding_boxes(detections, directory, file_path)
        
        # Append results to the results string
        results_str += f"File: {file_path}\n"
        results_str += f"Frames with Multiple Objects: {multiple_objects}\n"
        results_str += f"Multiple Object Rate: {proportion}\n"
        results_str += f"Object ID Switches: {id_switches}\n\n"
    
    # Write results string to a file
    with open(os.path.join(directory, 'results.txt'), 'w') as f:
        f.write(results_str)

# Specify the directory containing the files
directory = "/home/visadmin/Desktop/Datenerhebung_DeepSort/Tests/"
process_files(directory)
