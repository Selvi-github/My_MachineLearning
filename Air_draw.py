import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Canvas setup
canvas = None
draw_color = (255, 0, 0)  # Default color Blue
brush_thickness = 10
eraser_thickness = 50
mode = "draw"  # draw / erase

# For connecting previous point
prev_x, prev_y = 0, 0

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)  # Mirror effect

    # Initialize canvas
    if canvas is None:
        canvas = np.zeros_like(img)

    # Convert to RGB for Mediapipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get index finger tip
            h, w, c = img.shape
            cx, cy = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)

            # Draw pointer on fingertip
            cv2.circle(img, (cx, cy), 8, draw_color, cv2.FILLED)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = cx, cy

            # Drawing mode
            if mode == "draw":
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), draw_color, brush_thickness)

            # Eraser mode
            elif mode == "erase":
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), (0, 0, 0), eraser_thickness)

            prev_x, prev_y = cx, cy
    else:
        # Reset previous points if hand not detected
        prev_x, prev_y = 0, 0

    # Merge canvas with webcam feed
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, inv)
    img = cv2.bitwise_or(img, canvas)

    # Show mode info on screen
    cv2.putText(img, f"Mode: {mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Virtual Painter", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('e'):  # Eraser mode
        mode = "erase"
    elif key == ord('d'):  # Draw mode
        mode = "draw"
    elif key == ord('c'):  # Clear canvas
        canvas = np.zeros_like(img)

cap.release()
cv2.destroyAllWindows()
