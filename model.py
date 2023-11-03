from tensorflow.keras.models import load_model

model = load_model('skin_cancer_detection_model.h5')  # Replace 'model.h5' with your actual file path

import os
from tensorflow.keras.preprocessing import image
import numpy as np

test_folder = 'test_image'  # Replace with the path to your test image folder

def load_and_preprocess_images(folder_path):
    image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
    images = []
    for img_path in image_paths:
        img = image.load_img(img_path, target_size=(224, 224))  # Adjust the target size as needed
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image
        images.append(img)
    return np.vstack(images)

test_data = load_and_preprocess_images(test_folder)

predictions = model.predict(test_data)

# Interpret the predictions as needed based on your binary classification task
for i, prediction in enumerate(predictions):
    if prediction[0] > 0.5:
        print(f"Image {i + 1} is classified as having skin cancer.")
    else:
        print(f"Image {i + 1} is classified as not having skin cancer.")
