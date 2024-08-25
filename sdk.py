import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_images_from_folder(folder):
    images = []
    labels = []
    for category in os.listdir(folder):
        category_path = os.path.join(folder, category)
        if not os.path.isdir(category_path):
            continue
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            img = img.flatten()
            images.append(img)
            labels.append(category)
    return np.array(images), np.array(labels)

# Load the dataset
train_images, train_labels = load_images_from_folder(r"C:\Users\Mazveen\Documents\EAGLE\projects\trashnet_dataset\train")

# Train the SVM model
model = SVC(kernel='linear')
model.fit(train_images, train_labels)

# Save the model to disk
joblib.dump(model, 'svm_trashnet_model.pkl')
