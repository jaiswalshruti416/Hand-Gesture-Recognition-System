import cv2
import numpy as np
import pickle

# Load model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100,100), (400,400), (255,0,0), 2)

    img = cv2.resize(roi, (100, 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = gray.flatten().reshape(1, -1)

    prediction = model.predict(data)

    cv2.putText(frame, f"Gesture: {prediction[0]}",
                (100, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
