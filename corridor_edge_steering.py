import cv2
import numpy as np

def nothing(x):
    pass

def initializeTrackbars():
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 150, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 255, 255, nothing)

def getThresholds():
    t1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    t2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    return t1, t2

def thresholding(img):
    t1, t2 = getThresholds()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, t1, t2)
    return canny

def getHistogram(canny_img):
    hist = np.sum(canny_img[canny_img.shape[0]//2:, :], axis=0)
    return hist

def getLaneCenter(histogram):
    midpoint = np.int32(histogram.shape[0] / 2)
    leftx = np.argmax(histogram[:midpoint])
    rightx = np.argmax(histogram[midpoint:]) + midpoint
    center = (leftx + rightx) // 2
    return center, leftx, rightx

def drawSteeringLine(img, center, frame_center):
    cv2.line(img, (center, img.shape[0]), (frame_center, int(img.shape[0] / 2)), (255, 0, 255), 3)
    cv2.circle(img, (center, img.shape[0]), 5, (0, 255, 0), cv2.FILLED)
    return img

# === Main Program ===

cap = cv2.VideoCapture(0)  # Use 0 for webcam, or path to video file
frameWidth = 640
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)

initializeTrackbars()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (frameWidth, frameHeight))
    imgCanny = thresholding(img)

    histogram = getHistogram(imgCanny)
    center, leftx, rightx = getLaneCenter(histogram)

    steering_angle = center - frameWidth // 2

    # Draw colored lane markers
    overlay = img.copy()
    cv2.line(overlay, (leftx, frameHeight), (leftx, frameHeight//2), (0, 255, 0), 4)    # Green for left
    cv2.line(overlay, (rightx, frameHeight), (rightx, frameHeight//2), (0, 0, 255), 4)   # Red for right
    imgOverlay = cv2.addWeighted(img, 0.8, overlay, 1, 0)

    # Draw steering angle
    output = drawSteeringLine(imgOverlay, center, frameWidth // 2)

    cv2.putText(output, f"Steering Angle: {steering_angle}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Lane Detection", output)
    cv2.imshow("Canny", imgCanny)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
