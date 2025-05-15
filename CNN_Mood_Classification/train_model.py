import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create directories if not exist
if not os.path.exists('dataset'):
    raise FileNotFoundError("Please prepare the dataset folder with 'happy' and 'sad' subfolders under 'dataset/train' and 'dataset/validation'.")

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

val_set = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_set, validation_data=val_set, epochs=50)
   
# Save the model
model.save("mood_model.h5")
print("Model saved as mood_model.h5") 
