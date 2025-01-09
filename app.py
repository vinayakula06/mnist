import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np

# Define the updated LeNet5 model class (matching the original model architecture)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # Input: 28x28, Output: 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)                      # Input: 14x14, Output: 10x10
        self.fc1 = nn.Linear(16 * 4 * 4, 120)                             # Set to match original saved model architecture
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))                      # Output: 28x28
        x = F.avg_pool2d(x, kernel_size=2)                 # Output: 14x14
        x = torch.tanh(self.conv2(x))                      # Output: 10x10
        x = F.avg_pool2d(x, kernel_size=2)                 # Output: 5x5
        x = x.view(-1, 16 * 4 * 4)                         # Updated to match output size of 16*4*4
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to load the model and its state
def load_model(model, model_path="mnist_digit_recognizer.pth"):
    try:
        # Try loading the model state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check for model layer compatibility
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        
        # If layers don't match, print a warning for missing keys
        missing_keys = set(state_dict.keys()) - set(pretrained_dict.keys())
        if missing_keys:
            st.warning(f"Missing keys: {missing_keys}")
        
        # Update model with pretrained weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        st.success("Model loaded successfully.")
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the trained model
model = LeNet5()
model = load_model(model)

# Define the prediction function
def predict_image(img_data):
    try:
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
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None

# Streamlit interface
st.title("LeNet-5 MNIST Prediction")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    prediction = predict_image(uploaded_file.read())
    if prediction is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write(f"Predicted Digit: {prediction}")
else:
    if model is None:
        st.write("Model is not loaded. Please check the logs for errors.")
    else:
        st.write("Please upload an image file.")
