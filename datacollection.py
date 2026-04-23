import cv2
import os

gesture_name = input("Enter gesture name: ")
dataset_path = "dataset/" + gesture_name

os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)

    cv2.imshow("ROI", roi)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite(f"{dataset_path}/{count}.jpg", roi)
        count += 1
        print("Saved:", count)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
