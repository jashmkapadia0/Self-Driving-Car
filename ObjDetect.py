import cv2
import imutils
import serial
import time
from yoloDet import YoloTRT

# Load YOLO model
model = YoloTRT(
    library="yolov5/build/libmyplugins.so",
    engine="yolov5/build/yolov5s.engine",
    conf=0.5,
    yolo_ver="v5"
)

# Connect to Arduino (check your port: /dev/ttyUSB0 or COM3 etc.)
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # wait for Arduino to initialize

# Start video capture (from Pi camera or USB cam)
cap = cv2.VideoCapture(0)

# Define stop-worthy classes
STOP_CLASSES = [0]  # 0 is usually 'person' in COCO

def is_obstacle_ahead(boxes, class_ids, frame_width):
    """
    Check if a stop-worthy object is in the center zone.
    """
    center_zone = (int(frame_width * 0.4), int(frame_width * 0.6))  # central 20% width
    for box, cls in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        if cls in STOP_CLASSES and center_zone[0] <= center_x <= center_zone[1]:
            return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    boxes, confs, class_ids = model.Infer(frame)

    # Draw boxes and labels
    for box, cls, conf in zip(boxes, class_ids, confs):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Check if we need to stop
    if is_obstacle_ahead(boxes, class_ids, frame.shape[1]):
        print("Obstacle Ahead: STOP")
        arduino.write(b's')  # Send 's' to Arduino to stop motors
    else:
        print("Path Clear: GO")
        arduino.write(b'g')  # Send 'g' to Arduino to go forward

    cv2.imshow("Self-Driving View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
