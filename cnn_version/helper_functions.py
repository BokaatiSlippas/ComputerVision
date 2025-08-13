from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import pickle as pk
import cv2
import numpy as np


emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def load_and_rebuild_generator(filepath, batch_size=32):
    with open(filepath, 'rb') as f:
        data = pk.load(f)
    
    # Recreate the augmentation generator
    train_datagen = ImageDataGenerator(**data['augmentation_params'])
    train_generator = train_datagen.flow(
        data['X_train'],
        data['y_train'],
        batch_size=batch_size
    )
    
    return {
        'train_generator': train_generator,
        'validation_data': (data['X_val'], data['y_val']),
        'class_weights': data['class_weights'],
        'metadata': {
            'emotions': data['emotions'],
            'image_size': data['image_size']
        }
    }


def cnn_model(train_generator, validation_data, class_weights=None):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),  # Batches per epoch
        epochs=30,
        validation_data=validation_data,
        class_weight=class_weights  # Optional
    )
    return model, history


def predict_emotion(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = np.expand_dims(img, axis=-1) / 255.0  # Preprocess
    pred = model.predict(np.array([img]))
    emotion_idx = np.argmax(pred[0])
    return emotions[emotion_idx]  # Your emotion labels