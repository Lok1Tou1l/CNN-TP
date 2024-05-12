import torch
from torchvision import transforms
import model as net
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Load the trained model
model = net.CNNModel(num_classes=4)
model.load_state_dict(torch.load('models/trained_model.pth'))
model.eval()

# Define the labels for the model
labels = ['piece 1', 'piece 2', 'piece ', 'piece 4']

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to PyTorch Tensor data type
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Create a tkinter window
window = tk.Tk()

# Function to handle image recognition
def recognize_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    
    # Load and preprocess the image
    image = Image.open(file_path)
    image = transform(image).unsqueeze(0)  # Add an extra dimension for the batch size

    # Pass the image through the model and get the predictions
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)  # Get the index of the highest value

    # Get the predicted label
    predicted_label = labels[predicted.item()]

    # Create a label widget to display the predicted label
    label_widget = tk.Label(window, text="Predicted label: " + predicted_label)
    label_widget.pack()

    # Print the predicted label
    print('Predicted label:', predicted_label)

# Add a button to the tkinter window to recognize the image
button = tk.Button(window, text="Recognize Image", command=recognize_image)
button.pack()

# Run the tkinter window
window.mainloop()