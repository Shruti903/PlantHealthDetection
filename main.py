import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import matplotlib.pyplot as plt

# Paths
data_dir = r'C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data'  # Your data folder
batch_size = 32
img_height = 150
img_width = 150

# Image data processing with augmentation (randomly modifying images to help the model generalize)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values between 0 and 1
    rotation_range=40,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Random shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest',  # Fill any missing pixels after transformations
    validation_split=0.2  # Split data into training and validation sets (80% train, 20% validate)
)

# Loading and processing the training data
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # We have two classes: Healthy and Diseased
    subset='training'  # This is the training data subset
)

# Loading and processing the validation data
val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # This is the validation data subset
)

# Defining the model (a simple Convolutional Neural Network)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),  # First Conv layer
    MaxPooling2D((2, 2)),  # Max Pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Second Conv layer
    MaxPooling2D((2, 2)),  # Max Pooling layer
    Flatten(),  # Flatten the 2D matrix to a 1D vector
    Dense(128, activation='relu'),  # Fully connected layer
    Dense(1, activation='sigmoid')  # Output layer (binary classification: 1 or 0)
])

# Compiling the model (using the Adam optimizer and binary cross-entropy loss function)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early stopping to stop training if the validation loss isn't improving
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stopping]  # Early stopping callback
)

# Saving the trained model to a file
model.save('plant_health_model.keras')
print("Model saved as plant_health_model.keras")

# Saving the training history (accuracy and loss values) to a file
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("Training history saved as training_history.pkl")

# Plotting the accuracy graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('accuracy_plot.png')  # Save the accuracy plot as an image
plt.show()

# Plotting the loss graph
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('loss_plot.png')  # Save the loss plot as an image
plt.show()
