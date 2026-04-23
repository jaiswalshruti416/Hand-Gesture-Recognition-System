import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

dataset_path = "dataset"
data = []
labels = []

for label in os.listdir(dataset_path):
    path = os.path.join(dataset_path, label)

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data.append(gray.flatten())
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)
