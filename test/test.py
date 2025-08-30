import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model('plant_health_model.keras')

# Function to predict plant health
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    print(f"Raw prediction score: {prediction[0][0]}")  # Log the confidence score

    if prediction[0][0] > 0.5:
        print("The plant is diseased.")
    else:
        print("The plant is healthy.")

# Test the model
test_image = r"C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data\healthy\Apple___healthy\00fca0da-2db3-481b-b98a-9b67bb7b105c___RS_HL 7708.JPG"
predict_image(test_image)
