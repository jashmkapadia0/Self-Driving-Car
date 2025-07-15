import cv2
import os

video = cv2.VideoCapture('charusat_path.mp4')
output_dir = 'dataset/images'
os.makedirs(output_dir, exist_ok=True)
count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    resized = cv2.resize(frame, (160, 120))
    cv2.imwrite(f"{output_dir}/frame_{count:05d}.jpg", resized)
    count += 1

video.release()
