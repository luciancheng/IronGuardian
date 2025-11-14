import cv2
import numpy as np

lower_hsv = np.array([0, 120, 70])
upper_hsv = np.array([10, 255, 255])

current_frame = None 


def on_click(event, x, y, flags, param):
    global lower_hsv, upper_hsv, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_frame is None:
            return

        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        pixel = hsv[y, x]
        h = int(pixel[0])
        s = int(pixel[1])
        v = int(pixel[2])

        print("Clicked HSV:", pixel)
        lower_hsv = np.array([max(h - 10, 0), max(s - 60, 0), max(v - 60, 0)])
        upper_hsv = np.array([min(h + 10, 179), min(s + 60, 255), min(v + 60, 255)])
        print("lower_hsv =", lower_hsv)
        print("upper_hsv =", upper_hsv)
        print()


def main():
    global current_frame, origin_c
    cap = cv2.VideoCapture(1) 

    origin_center = None

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", on_click)

    dx = 0
    dy = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read from camera.")
            break

        frame = cv2.flip(frame, 1)
        current_frame = frame.copy() 

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea) # pick the largest contour
            M = cv2.moments(c)

            # draw the contour
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = (cx, cy)

                cv2.circle(frame, center, 6, (0, 255, 0), -1)

                if origin_center is None:
                    origin_center = (cx, cy)

                # Absolute displacement from origin
                dx = cx - origin_center[0]
                dy = cy - origin_center[1]

        text = f"dx: {dx}, dy: {dy}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
