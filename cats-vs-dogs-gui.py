import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('model1_cats-vs-dogs_10epoch.h5')

# Dictionary to map class indices to labels
classes = {
    0: 'its a cat',
    1: 'its a dog',
}

# Initialize the main window
top = tk.Tk()
top.geometry('800x600')
top.title('Cats VS Dogs Classification')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


# Function to classify an image
def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((128, 128))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image / 255.0  # Ensure the values are in the range [0, 1]

    # Predict the class probabilities
    pred = model.predict(image)[0]

    # Get the index of the class with the highest probability
    class_index = np.argmax(pred)

    # Get the class label from the dictionary
    sign = classes[class_index]

    print(sign)
    label.configure(foreground='#011638', text=sign)


# Function to show the classify button
def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",
                        command=lambda: classify(file_path),
                        padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white',
                         font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


# Function to upload an image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25),
                            (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except Exception as e:
        print(f"Error: {e}")


# Upload button
upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Cats VS Dogs Classification", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

# Start the main event loop
top.mainloop()
