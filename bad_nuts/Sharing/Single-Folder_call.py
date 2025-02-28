#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Function to get all image file paths from a folder
def get_image_paths_from_folder(folder_path):
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(".jpg")]

# Load and process images
def load_images(image_paths, label):
    data = []
    labels = []
    orb = cv2.ORB_create()  # ORB feature detector
    
    for path in image_paths:
        image = cv2.imread(path, 0)  # Load image in grayscale
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None:
            # Average ORB descriptors for consistent feature length
            feature_vector = np.mean(descriptors, axis=0)
            data.append(feature_vector)
            labels.append(label)
        else:
            print(f"No descriptors found for image: {path}")
    
    return data, labels

# Define folder paths
good_images_folder = "Pictures/Good_image"  # Path to the folder with good images
defective_images_folder = "Pictures/Defective_image"  # Path to the folder with defective images

# Call the folder-based loading function
good_image = get_image_paths_from_folder(good_images_folder)
defective_image = get_image_paths_from_folder(defective_images_folder)

# Load images and extract features
good_data, good_labels = load_images(good_image, label=1)  # Label 1 for good images
defective_data, defective_labels = load_images(defective_image, label=0)  # Label 0 for defective images

# Combine data and labels
data = good_data + defective_data
labels = good_labels + defective_labels

# Convert data and labels to numpy arrays for training
data = np.array(data)
labels = np.array(labels)

# Check if both classes are present
if len(set(labels)) < 2:
    raise ValueError("Not enough classes in training data. Ensure both good and defective images are loaded.")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Classify a test image
def classify_image(image_path):
    orb = cv2.ORB_create()
    image = cv2.imread(image_path, 0)  # Load test image in grayscale
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is not None:
        # Average descriptors for consistency with training data
        feature_vector = np.mean(descriptors, axis=0)
        prediction = svm.predict([feature_vector])
        return "Good" if prediction[0] == 1 else "Defective"
    else:
        return "Cannot classify image - no features detected"

# Test the classifier
test_image_path = "Pictures/test_5.jpg"  # Replace with your test image path
result = classify_image(test_image_path)
print("The test image is classified as:", result)


# In[ ]:




