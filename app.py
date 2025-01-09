import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np

# Define the LeNet5 model class
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Updated to 16 * 4 * 4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = nn.functional.avg_pool2d(x, 2)
        x = torch.tanh(self.conv2(x))
        x = nn.functional.avg_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)  # Updated to 16 * 4 * 4
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize and load the trained model
model = LeNet5()
model.load_state_dict(torch.load('mnist_digit_recognizer.pth', map_location=torch.device('cpu')))
model.eval()

# Define the prediction function
def predict_image(img):
    img = Image.open(io.BytesIO(img)).convert('L').resize((28, 28))

    # Convert PIL image to NumPy array
    img = np.array(img, dtype=np.float32)

    # Convert NumPy array to PyTorch tensor
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).div(255.0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Streamlit interface
st.title("LeNet-5 Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    prediction = predict_image(uploaded_file.read())
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write(f"Predicted class: {prediction}")
else:
    st.write("Please upload an image file.")
