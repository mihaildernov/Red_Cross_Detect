import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    CAM_WIDTH = 640
    CAM_HEIGHT = 480

    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.blur(img_gray, (3, 3))
    img_bin = cv2.threshold(img_blur, 127, 255, cv2.THRESH_OTSU)[1]
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2_HSV)[119:120, 159:160]

    red_low1 = (0, 50, 50)
    red_high1 = (15, 255, 255)

    red_low2 = (165, 50, 50)
    red_high2 = (180, 255, 255)

    red_thresh1 = cv2.inRange(img_hsv, red_low1, red_high1)
    red_thresh2 = cv2.inRange(img_hsv, red_low2, red_high2)
    red_thresh = cv2.bitwise_or(red_thresh1, red_thresh2)

    dst = cv2.Canny(frame, 50, 200, None, 3)

    cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    contours, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        approx_area = cv2.contourArea(approx)

    cv2.imshow("Red Cross Detection", frame)

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
