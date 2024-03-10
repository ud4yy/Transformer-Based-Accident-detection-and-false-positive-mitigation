from ultralytics import YOLO
import os
import glob
import re

# Load a model
model = YOLO('/home/uday/Desktop/ML_project/weights/best.pt')  # pretrained YOLOv8n model

def extract_frame_number(file_path):
    # Use regular expression to find the frame number in the file name
    match = re.search(r"collision_frame_(\d+)", file_path)
    return int(match.group(1)) if match else float('inf')

"""Modified function"""

def classify(image_folder_path):
    image_files = glob.glob(os.path.join(image_folder_path, '*.jpg')) + glob.glob(os.path.join(image_folder_path, '*.png'))

    sorted_a = sorted(image_files, key=extract_frame_number)
    image_files = sorted_a.copy()

    if len(image_files) > 10:
        frames_to_process = [image_files[9], image_files[(len(image_files)//2)+1]]
    elif len(image_files) >3 and len(image_files)<=10:
        print(image_files[len(image_files)-1])
        frames_to_process = [image_files[(len(image_files)//2)+1], image_files[(len(image_files)//2)+2], image_files[len(image_files)-1]];
    else:
        frames_to_process = [image_files[0],image_files[1]];
    for image_path in frames_to_process:
        model.predict(image_path, save=True,conf=0.4, save_crop=True)

classify('/home/uday/Desktop/ML_project/outputs/abruptchange')



