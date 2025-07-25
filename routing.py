import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import serial
import time
import ctypes
import imutils

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Constants
CONF_THRESH = 0.5
IOU_THRESH = 0.4
INPUT_W = 640  # YOLOv5 input width
INPUT_H = 640  # YOLOv5 input height
LEN_ALL_RESULT = 38001  # Total output size for YOLOv5
LEN_ONE_RESULT = 38  # Single detection size
CATEGORIES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
              "hair drier", "toothbrush"]
STOP_CLASSES = [0]  # Person class ID for stopping

class YoloTRT:
    def __init__(self, library_path, engine_path):
        # Load the shared library
        ctypes.CDLL(library_path)
        
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            serialized_engine = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.batch_size = self.engine.max_batch_size
        
        # Allocate buffers
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        
        self.context = self.engine.create_execution_context()

    def preprocess(self, img):
        """Preprocess image for YOLOv5 input."""
        image_raw = img
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = INPUT_W / w
        r_h = INPUT_H / h
        if r_h > r_w:
            tw = INPUT_W
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((INPUT_H - th) / 2)
            ty2 = INPUT_H - th - ty1
        else:
            tw = int(r_h * w)
            th = INPUT_H
            tx1 = int((INPUT_W - tw) / 2)
            tx2 = INPUT_W - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def postprocess(self, output, origin_h, origin_w):
        """Postprocess YOLOv5 output."""
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, LEN_ONE_RESULT))[:num, :6]
        boxes = self.xywh2xyxy(pred[:, :4], origin_h, origin_w)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        scores = pred[:, 4]
        class_ids = pred[:, 5]
        keep = self.nms(boxes, scores, class_ids)
        return boxes[keep], scores[keep], class_ids[keep]

    def xywh2xyxy(self, x, origin_h, origin_w):
        """Convert xywh to xyxy format."""
        y = np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y

    def nms(self, boxes, scores, class_ids):
        """Apply Non-Maximum Suppression."""
        indices = []
        sorted_idx = np.argsort(-scores)
        while len(sorted_idx):
            idx = sorted_idx[0]
            indices.append(idx)
            if len(sorted_idx) == 1:
                break
            ious = self.bbox_iou(boxes[idx:idx+1], boxes[sorted_idx[1:]])
            keep = ious <= IOU_THRESH
            sorted_idx = sorted_idx[1:][keep]
        return np.array(indices)

    def bbox_iou(self, box1, box2):
        """Calculate IoU between boxes."""
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_x2 - inter_x1 + 1, 0, None) * np.clip(inter_y2 - inter_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou

    def inference(self, img):
        """Run YOLOv5 inference on an image."""
        input_image, image_raw, h, w = self.preprocess(img)
        np.copyto(self.host_inputs[0], input_image.ravel())
        stream = cuda.Stream()
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], stream)
        t1 = time.time()
        self.context.execute_async(self.batch_size, self.bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], stream)
        stream.synchronize()
        t2 = time.time()
        output = self.host_outputs[0]
        boxes, scores, class_ids = self.postprocess(output, h, w)
        return boxes, scores, class_ids, t2 - t1

    def draw_boxes(self, img, boxes, scores, class_ids):
        """Draw bounding boxes on the image."""
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{CATEGORIES[int(class_id)]}: {score:.2f}"
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def is_obstacle_ahead(boxes, class_ids, frame_width):
    """Check if a person is in the center zone."""
    center_zone = (int(frame_width * 0.4), int(frame_width * 0.6))  # Central 20% width
    for box, cls in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) // 2
        if int(cls) in STOP_CLASSES and center_zone[0] <= center_x <= center_zone[1]:
            return True
    return False

def main():
    # Initialize YOLOv5 model
    model = YoloTRT(
        library_path="/home/aiml/Desktop/finaldet/libmyplugins.so",
        engine_path="/home/aiml/Desktop/finaldet/yolov5n.engine"
    )

    # Connect to Arduino
    try:
        arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        time.sleep(2)  # Wait for Arduino to initialize
    except serial.SerialException as e:
        print(f"Error: Could not connect to Arduino: {e}")
        return

    # Start video capture (CSI camera)
    pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        arduino.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = imutils.resize(frame, width=640)
            boxes, scores, class_ids, inference_time = model.inference(frame)
            model.draw_boxes(frame, boxes, scores, class_ids)

            if is_obstacle_ahead(boxes, class_ids, frame.shape[1]):
                print(f"Obstacle Ahead: STOP (Inference time: {inference_time:.3f}s)")
                arduino.write(b's')  # Stop
            else:
                print(f"Path Clear: GO (Inference time: {inference_time:.3f}s)")
                arduino.write(b'g')  # Go

            cv2.imshow("Self-Driving View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        arduino.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
