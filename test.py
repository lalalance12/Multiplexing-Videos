import cv2

# Attempt to capture video from the default camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


if not cap.isOpened():
    print("Cannot open camera")
else:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
    else:
        cv2.imshow('Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

cap.release()
