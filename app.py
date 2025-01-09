import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load your trained model
class MNISTModel(torch.nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Define your model architecture here (same as in training)

    def forward(self, x):
        # Define the forward pass here
        pass

# Initialize and load model
model = MNISTModel()
model.load_state_dict(torch.load("mnist_digit_recognizer.pth"))
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

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_digit = output.argmax(dim=1).item()

    st.write(f"Predicted Digit: {predicted_digit}")
