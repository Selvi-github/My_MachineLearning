import cv2

alg = r"C:/Users/lenovo/Downloads/haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

if haar_cascade.empty():
    print("❌ Error: Cascade file not loaded")
    exit()

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("❌ Error: Could not open camera")
    exit()

while True:
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("FaceDetection", img)

    key = cv2.waitKey(10)
    if key == 27:  # ESC to exit
        break

cam.release()
cv2.destroyAllWindows()
