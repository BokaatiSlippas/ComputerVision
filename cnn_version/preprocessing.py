import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from random import randint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import pickle as pk


dataset_path = "train"
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
image_size = (48, 48)

faces = []
labels = []

for i, emotion in enumerate(emotions):
    emotion_path = os.path.join(dataset_path, emotion)
    for image_name in os.listdir(emotion_path):
        try:
            image_path = os.path.join(emotion_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # grayscale because later arduino crap
            if image.shape != image_size: # Will later used my actual camera
                image = cv2.resize(image, image_size)
            image = image.astype('float32') / 255.0 # normalise values
            image = np.expand_dims(image, axis=-1) # Add channel dimension cuz apparently CNNs need this shape
            
            faces.append(image)
            labels.append(i)  # Use index as label
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")

faces = np.array(faces)
labels = np.array(labels)

labels_onehot = to_categorical(labels, num_classes=len(emotions)) # Convert labels to one-hot encoding

# Split consistently into training and validation sets
faces_train, faces_val, labels_train, labels_val = train_test_split(
    faces, 
    labels_onehot, 
    test_size=0.2, 
    random_state=randint(1, 100), 
    stratify=labels  # Maintain class proportions
)

# Data Augmentation (for more real-life variance in data)
augmentation_params = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Class Balancing (for more balanced penalties based on class occurrence, i.e. errors on less frequent classes more penalised)
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))


with open('fer2013_preprocessed.pkl', 'wb') as f:
    pk.dump({
        # Raw data splits
        'X_train': faces_train,
        'X_val': faces_val,
        'y_train': labels_train,
        'y_val': labels_val,
        
        # Augmentation configuration
        'augmentation_params': augmentation_params,
        
        # Class balancing
        'class_weights': class_weights,
        
        # Metadata
        'emotions': emotions,
        'image_size': image_size,
        'classes': np.unique(labels)
    }, f)