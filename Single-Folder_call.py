import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Print current working directory
print("📂 Current Working Directory:", os.getcwd())

# Define absolute paths for image folders (Modify paths based on your system)
good_images_folder = r"E:\Machine_vision_python_code\good_nuts"
defective_images_folder = r"E:\Machine_vision_python_code\bad_nuts"

# Ensure directories exist before proceeding
if not os.path.exists(good_images_folder):
    raise FileNotFoundError(f"❌ Error: Folder '{good_images_folder}' not found.")
if not os.path.exists(defective_images_folder):
    raise FileNotFoundError(f"❌ Error: Folder '{defective_images_folder}' not found.")

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
            feature_vector = np.mean(descriptors, axis=0)
        else:
            print(f"⚠️ No descriptors found for image: {path}")
            feature_vector = np.zeros(32)  # Fallback feature vector
        
        data.append(feature_vector)
        labels.append(label)
    
    return data, labels

# Get image paths
good_image_paths = get_image_paths_from_folder(good_images_folder)
defective_image_paths = get_image_paths_from_folder(defective_images_folder)

# Load images and extract features
good_data, good_labels = load_images(good_image_paths, label=1)  # Label 1 for good images
defective_data, defective_labels = load_images(defective_image_paths, label=0)  # Label 0 for defective images

# Combine data and labels
data = good_data + defective_data
labels = good_labels + defective_labels

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Ensure data is valid
if data.size == 0:
    raise ValueError("❌ No valid image features extracted. Check dataset and ORB parameters.")

# Ensure both classes are present
if len(set(labels)) < 2:
    raise ValueError("❌ Not enough classes in training data. Ensure both good and defective images are loaded.")

# Normalize features
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train SVM classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = svm.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation results
print("\n🔍 **Model Evaluation Metrics:**")
print(f"✅ Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")
print(f"✅ F1 Score: {f1:.2f}")
print("\n📊 Confusion Matrix:")
print(conf_matrix)

# Function to classify a test image
def classify_image(image_path):
    orb = cv2.ORB_create()
    image = cv2.imread(image_path, 0)  # Load test image in grayscale
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    if descriptors is not None:
        feature_vector = np.mean(descriptors, axis=0)
    else:
        print(f"⚠️ No descriptors found for image: {image_path}")
        feature_vector = np.zeros(32)  # Use a zero vector as fallback
    
    # Normalize the feature vector
    feature_vector = scaler.transform([feature_vector])  

    prediction = svm.predict(feature_vector)
    return "✅ Good" if prediction[0] == 1 else "❌ Defective"

# Test the classifier on a new image
test_image_path = r"test_image\test_5.jpg"  # Replace with your test image path

if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"❌ Test image '{test_image_path}' not found.")

result = classify_image(test_image_path)
print("\n📸 **Test Image Classification Result:**", result)
