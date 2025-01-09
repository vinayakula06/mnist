import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np


# Define the updated LeNet5 model class
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the trained model
model = LeNet5()
model.load_state_dict(torch.load("mnist_digit_recognizer.pth", map_location=torch.device('cpu')))
model.eval()


# Define the prediction function
def predict_image(img_data):
    # Load image from bytes and preprocess
    img = Image.open(io.BytesIO(img_data)).convert('L').resize((28, 28))

    # Convert to NumPy array, then to PyTorch tensor
    img = np.array(img, dtype=np.float32)
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).div(255.0)

    # Predict using the model
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return predicted.item()


# Streamlit interface
st.title("LeNet-5 MNIST Prediction")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    prediction = predict_image(uploaded_file.read())
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write(f"Predicted Digit: {prediction}")
else:
    st.write("Please upload an image file.")
