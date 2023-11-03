import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Lists to store accuracy history for plotting
accuracy_history = []

# Function to preprocess and classify an image
def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")],
    )

    if not file_path:
        return

    try:
        # Preprocess the selected image
        img = Image.open(file_path)
        img = img.resize((224, 224))  # Adjust the target size as needed
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make a prediction
        prediction = model.predict(img)

        # Store accuracy in the history list
        accuracy = model.evaluate(img, prediction)[1]
        accuracy_history.append(accuracy)

        # Display the image in the GUI
        photo = ImageTk.PhotoImage(Image.open(file_path).resize((400, 400)))
        image_label.config(image=photo)
        image_label.image = photo

        # Display the classification result
        if prediction[0][0] > 0.5:
            result_label.config(text="This image has cancer.")
        else:
            result_label.config(text="This image does not have cancer.")

        # Plot the accuracy history
        plot_accuracy()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def plot_accuracy():
    plt.clf()
    plt.plot(range(len(accuracy_history)), accuracy_history, marker='o')
    plt.title("Accuracy History")
    plt.xlabel("Image Classification")
    plt.ylabel("Accuracy")
    plt.grid(True)
    accuracy_fig = plt.gcf()
    accuracy_fig.set_size_inches(5, 3)
    accuracy_fig.savefig("accuracy_graph.png")

# Create the main application window
app = tk.Tk()
app.title("Skin Cancer Detection")
app.geometry("800x600")

# Create and configure GUI elements
classify_button = tk.Button(app, text="Classify Image", command=classify_image)
classify_button.pack(pady=20)

image_label = tk.Label(app)
image_label.pack()

result_label = tk.Label(app, text="", font=("Helvetica", 16))
result_label.pack()

# Run the GUI main loop
app.mainloop()
