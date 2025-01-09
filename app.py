import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define your MNIST model class (consistent with training)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # Input: 28x28, Output: 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)                      # Input: 14x14, Output: 10x10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                             # Flattened from 16x5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))                      # Output: 28x28
        x = F.avg_pool2d(x, kernel_size=2)                 # Output: 14x14
        x = torch.tanh(self.conv2(x))                      # Output: 10x10
        x = F.avg_pool2d(x, kernel_size=2)                 # Output: 5x5
        x = x.view(-1, 16 * 5 * 5)                         # Flatten for FC layers
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = LeNet5()
model.load_state_dict(torch.load("mnist_digit_recognizer.pth", map_location=torch.device('cpu')))
model.eval()

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit interface
st.title("MNIST Digit Recognizer")
st.write("Upload an image of a digit, and the model will predict it!")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    try:
        input_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_digit = output.argmax(dim=1).item()

        st.write(f"Predicted Digit: {predicted_digit}")
    except Exception as e:
        st.write("Error in processing the image. Please upload a valid digit image.")
