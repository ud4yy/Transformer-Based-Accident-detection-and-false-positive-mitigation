from ultralytics import YOLO
import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np
from collections import defaultdict
import numpy as np
import os
from collections import defaultdict
import cv2
import numpy as np

def check_overlap(box1, box2):
    # Calculate half-widths and half-heights for both boxes
    half_width1, half_height1 = (box1[2] - box1[0]) / 2, (box1[3] - box1[1]) / 2
    half_width2, half_height2 = (box2[2] - box2[0]) / 2, (box2[3] - box2[1]) / 2

    # Calculate centers of the boxes
    x1_center, y1_center = box1[0] + half_width1, box1[1] + half_height1
    x2_center, y2_center = box2[0] + half_width2, box2[1] + half_height2

    # Check if the distance between centers is less than the sum of half-dimensions
    return (abs(x1_center - x2_center) < (half_width1 + half_width2)) and \
           (abs(y1_center - y2_center) < (half_height1 + half_height2))

def calculate_velocity(prev_center, current_center, time_delta):
    if prev_center is None or current_center is None:
       # return None
       return (0.0, 0.0)
    return ((current_center[0] - prev_center[0]) / time_delta, (current_center[1] - prev_center[1]) / time_delta)

def are_vectors_not_similar_direction(velocity1, velocity2):
    if velocity1 is None or velocity2 is None:
        return False
    # Normalize the velocity vectors
    norm_v1 = np.linalg.norm(velocity1)
    norm_v2 = np.linalg.norm(velocity2)
    if norm_v1 == 0 or norm_v2 == 0:
        return False  # One or both vectors have zero magnitude

    # Calculate the cosine of the angle between the two vectors
    cos_angle = np.dot(velocity1, velocity2) / (norm_v1 * norm_v2)

    # Check if the angle is not close to 0 degrees (cosine close to 1)
    return not np.isclose(cos_angle, 1, atol=0.1)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = '/home/uday/Desktop/ML_project/inputs/Nested05.mp4'
cap = cv2.VideoCapture(video_path)

# Define parameters for VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')


# Define the output directory for saving frames
output_dir = '/home/uday/Desktop/ML_project/outputs/abruptchange'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fgModel = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

# Store the track history
track_history = defaultdict(lambda: [])
prev_centers = defaultdict(lambda: None)
time_of_last_frame = None
frame_index = 0
max_frames_with_abrupt_change = 30
frames_with_abrupt_change_counter = 0
velocity_threshold = 0

# Store the track history
track_history = defaultdict(lambda: [])
prev_centers = defaultdict(lambda: None)
time_of_last_frame = None
frame_index = 0
max_frames_with_abrupt_change = 30
frames_with_abrupt_change_counter = 0

while cap.isOpened() and frames_with_abrupt_change_counter < max_frames_with_abrupt_change:
    success, frame = cap.read()
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    if time_of_last_frame is None:
        time_of_last_frame = current_time

    if success:
        results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[2])

        overlap_detected = False
        if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for i in range(len(boxes)):
                x_center_i, y_center_i = (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2
                velocity_i = calculate_velocity(prev_centers[track_ids[i]], (x_center_i, y_center_i), current_time - time_of_last_frame)
                prev_centers[track_ids[i]] = (x_center_i, y_center_i)

                for j in range(i + 1, len(boxes)):
                    if check_overlap(boxes[i], boxes[j]):
                        x_center_j, y_center_j = (boxes[j][0] + boxes[j][2]) / 2, (boxes[j][1] + boxes[j][3]) / 2
                        velocity_j = calculate_velocity(prev_centers[track_ids[j]], (x_center_j, y_center_j), current_time - time_of_last_frame)
                        prev_centers[track_ids[j]] = (x_center_j, y_center_j)

                        if are_vectors_not_similar_direction(velocity_i, velocity_j):
                            overlap_detected = True
                            break
                    if overlap_detected:
                      break

            annotated_frame = results[0].plot()

            # Write frame to output video if overlap is detected
            if overlap_detected:
                image_path = os.path.join(output_dir, f'collision_frame_{frame_index}.jpg')
                cv2.imwrite(image_path, frame)
                frame_index += 1    # Increment the counter for frames with abrupt changes
                frames_with_abrupt_change_counter += 1

                overlap_detected = False





    else:
        break

cap.release()
cv2.destroyAllWindows()





"""
while cap.isOpened() and frames_with_abrupt_change_counter < max_frames_with_abrupt_change:
    success, frame = cap.read()
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    if time_of_last_frame is None:
        time_of_last_frame = current_time

    if success:
        results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[2,7])
        fgmask = fgModel.apply(frame)  # Apply background subtraction
        fgmask = cv2.morphologyEx(np.float32(fgmask),cv2.MORPH_OPEN,kernel)
         # Check if any foreground is detected
        if np.sum(fgmask) > 0:
         overlap_detected = False
         if results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for i in range(len(boxes)):
                x_center_i, y_center_i = (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2
                velocity_i = calculate_velocity(prev_centers[track_ids[i]], (x_center_i, y_center_i), current_time - time_of_last_frame)
                prev_centers[track_ids[i]] = (x_center_i, y_center_i)
                # Skip objects with low velocity
                print(np.linalg.norm(velocity_i))
                if np.linalg.norm(velocity_i) < velocity_threshold:
                  continue
                for j in range(i + 1, len(boxes)):
                    if check_overlap(boxes[i], boxes[j]):
                        x_center_j, y_center_j = (boxes[j][0] + boxes[j][2]) / 2, (boxes[j][1] + boxes[j][3]) / 2
                        velocity_j = calculate_velocity(prev_centers[track_ids[j]], (x_center_j, y_center_j), current_time - time_of_last_frame)
                        prev_centers[track_ids[j]] = (x_center_j, y_center_j)

                        if are_vectors_not_similar_direction(velocity_i, velocity_j):
                            overlap_detected = True
                            break
                    if overlap_detected:
                      break

            annotated_frame = results[0].plot()

            # Write frame to output video if overlap is detected
            if overlap_detected:

                image_path = os.path.join(output_dir, f'collision_frame_{frame_index}.jpg')
                cv2.imwrite(image_path, frame)
                frame_index += 1    # Increment the counter for frames with abrupt changes
                frames_with_abrupt_change_counter += 1

                overlap_detected = False


    else:
        break

cap.release()
cv2.destroyAllWindows()

"""


