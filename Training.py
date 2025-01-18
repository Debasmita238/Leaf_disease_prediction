import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Enable eager execution (TensorFlow 2.x uses eager execution by default)
# No need to disable it unless explicitly required
# tf.compat.v1.enable_eager_execution()

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))  # Use 'softmax' for multiclass classification

# Compile the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training and validation datasets
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation datasets
training_set = train_datagen.flow_from_directory(
    r'C:\Users\DEBASMITA SAHA\OneDrive\Desktop\test\project\Plant_Leaf_Disease_Prediction\Dataset\train',
    target_size=(128, 128),
    batch_size=6,
    class_mode='categorical'
)

valid_set = test_datagen.flow_from_directory(
    r'C:\Users\DEBASMITA SAHA\OneDrive\Desktop\test\project\Plant_Leaf_Disease_Prediction\Dataset\val',
    target_size=(128, 128),
    batch_size=3,
    class_mode='categorical'
)

# Display class labels
labels = training_set.class_indices
print("Class Indices:", labels)

# Train the model
classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=50,
    validation_data=valid_set,
    validation_steps=len(valid_set)
)

# Save the model architecture and weights
classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)

classifier.save_weights("my_model_weights.weights.h5")
classifier.save("model.h5")
print("Model saved successfully.")
