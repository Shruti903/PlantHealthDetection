import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('plant_health_model.keras')

# Function to preprocess the image and make predictions
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    return prediction[0][0]  # Return confidence score

# Function to handle image uploads
def upload_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_paths:
        return

    # Clear previous results
    result_box.delete(1.0, tk.END)

    # Process each selected image
    for file_path in file_paths:
        try:
            confidence = predict_image(file_path)
            status = "Diseased 🌱" if confidence > 0.5 else "Healthy ✅"

            # Display result in the text box
            result_box.insert(tk.END, f"{os.path.basename(file_path)}: {status}\n")

            # Display the last image in the GUI
            img = Image.open(file_path)
            img = img.resize((300, 300))  # Resize for display
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk  # Keep reference to avoid garbage collection

        except Exception as e:
            result_box.insert(tk.END, f"Error processing {file_path}: {str(e)}\n")

# Function to process fixed paths (batch testing)
def batch_process():
    file_paths = [
        r"C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data\healthy\Peach___healthy\0a2ed402-5d23-4e8d-bc98-b264aea9c3fb___Rutg._HL 2471_90deg.JPG",
        r"C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data\healthy\Apple___healthy\00fca0da-2db3-481b-b98a-9b67bb7b105c___RS_HL 7708.JPG",
        r"C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data\healthy\Blueberry___healthy\00fee259-67b7-4dd7-8b36-12503bbdba14___RS_HL 2681_flipTB.JPG",
        r"C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data\healthy\Raspberry___healthy\00a3fc0e-64cc-4e35-ac2f-aef04fda9b22___Mary_HL 9177_90deg.JPG",
        r"C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data\unhealthy\Apple___Apple_scab\0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417_270deg.JPG",
        r"C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data\unhealthy\Grape___Black_rot\00cab05d-e87b-4cf6-87d8-284f3ec99626___FAM_B.Rot 3244_flipLR.JPG",
        r"C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data\unhealthy\Potato___Early_blight\0a47f32c-1724-4c8d-bfe4-986cedd3587b___RS_Early.B 8001.JPG",
        r"C:\Users\shrut\OneDrive\Desktop\PlantHealthDetector\data\unhealthy\Tomato___Target_Spot\0a2de4c5-d688-4f9d-9107-ace1d281c307___Com.G_TgS_FL 7941_180deg.JPG"
    ]

    # Clear previous results
    result_box.delete(1.0, tk.END)

    # Process each predefined path
    for file_path in file_paths:
        try:
            confidence = predict_image(file_path)
            status = "Diseased 🌱" if confidence > 0.5 else "Healthy ✅"
            result_box.insert(tk.END, f"{os.path.basename(file_path)}: {status}\n")
        except Exception as e:
            result_box.insert(tk.END, f"Error processing {file_path}: {str(e)}\n")

# Create the GUI window
root = tk.Tk()
root.title("Plant Health Detector")
root.geometry("500x700")

# Title label
title_label = tk.Label(root, text="Plant Health Detector", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

# Image display area
image_label = tk.Label(root)
image_label.pack(pady=10)

# Buttons for uploading images and batch processing
upload_button = tk.Button(root, text="Upload Images", command=upload_images, font=("Helvetica", 12))
upload_button.pack(pady=10)

batch_button = tk.Button(root, text="Batch Process (Fixed Paths)", command=batch_process, font=("Helvetica", 12))
batch_button.pack(pady=10)

# Results display area
result_box = tk.Text(root, height=15, width=60, font=("Helvetica", 12))
result_box.pack(pady=10)

# Run the GUI
root.mainloop()
