import os
import cv2
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", default="./video/", type=str)
parser.add_argument("--data_path", default="./data/Fan/", type=str)
parser.add_argument("--video_info_file", default="./jsons/video_info.json", type=str)
args = parser.parse_args()

# find files in the directory
directories = ['normal', 'abnormal']
info, info_total = [], []
for directory in directories:
    path = os.path.join(args.video_path, directory)
    files = os.listdir(path)

    # create normal and abnormal directory for saving frames
    os.makedirs(os.path.join(args.data_path, directory), exist_ok=True)

    record = {
        "directory": directory,
        "total_frame": 0,
        "files": [],
    }
    
    total_frame = 0
    interval = 1
    for file in files:
        frame_count = 0
        cap = cv2.VideoCapture(os.path.join(path, file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("\n[{} | {}] Fps: {}".format(directory, file, fps))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break

            if total_frame % interval == 0:
                cv2.imwrite(os.path.join(args.data_path, directory, str(total_frame) + '.png'), frame)
                print("[{} | {}] Saving frame: {}".format(directory, file, total_frame))

            frame_count += 1
            total_frame += 1

        record["files"].append({
            "file": file,
            "frame_count": frame_count,
        })

    record["total_frame"] = total_frame
    info.append(record)

with open(args.video_info_file, 'w+') as f:
    json.dump(info, f, indent=4)
print("Save video info to {}".format(args.video_info_file))