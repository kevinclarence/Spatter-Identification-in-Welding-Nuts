#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Function to get all image file paths from a folder
def get_images_from_folder(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(".jpg")]

# Load and process images with consistent grayscale
def load_images(image_paths, label):
    data = []
    labels = []
    orb = cv2.ORB_create()  # ORB feature detector

    for path in image_paths:
        image = cv2.imread(path, 0)  # Load image in grayscale
        if image is not None:
            keypoints, descriptors = orb.detectAndCompute(image, None)
            if descriptors is not None:
                # Average ORB descriptors for consistent feature length
                feature_vector = np.mean(descriptors, axis=0)
                data.append(feature_vector)
                labels.append(label)
    return data, labels

# Paths to the parent folder containing 'Good' and 'Defective' subfolders
parent_folder = "Pictures"
good_folder = os.path.join(parent_folder, "Good_image")
defective_folder = os.path.join(parent_folder, "Defective_image")

# Load good and defective images
good_images = get_images_from_folder(good_folder)
defective_images = get_images_from_folder(defective_folder)

# Extract features and labels
good_data, good_labels = load_images(good_images, label=1)  # Label 1 for good images
defective_data, defective_labels = load_images(defective_images, label=0)  # Label 0 for defective images

# Combine data and labels
data = good_data + defective_data
labels = good_labels + defective_labels

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Ensure both classes are present
if len(set(labels)) < 2:
    raise ValueError("Not enough classes in the training data. Please ensure both Good and Defective images are provided.")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM classifier (linear kernel)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Define a function to classify an image
def classify_image(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is not None and len(descriptors) > 0:
        feature_vector = np.mean(descriptors, axis=0)
        prediction = svm.predict([feature_vector])
        return prediction[0]
    return None

# Set up camera feed
cap = cv2.VideoCapture(0)  # Use default camera
frame_threshold = 10
consecutive_no_nut_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize frame for consistency
    resized_frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Classify the frame
    result = classify_image(gray_frame)
    if result is None:
        consecutive_no_nut_frames += 1
        if consecutive_no_nut_frames >= frame_threshold:
            message = "Nut not placed properly"
        else:
            message = "Detecting..."
    else:
        consecutive_no_nut_frames = 0
        message = "Good" if result == 1 else "Defective"

    # Display the message on the frame
    cv2.putText(resized_frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Classification', resized_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

