import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Pneumonia Detection Dashboard", page_icon="🩺", layout="centered")

# Load your trained model
model = tf.keras.models.load_model('weights-014-0.1170.keras')

# Function to preprocess image
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))

    # Convert grayscale to RGB by duplicating the single channel
    image = np.stack([image] * 3, axis=-1)  # Make it 3 channels
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# --- Streamlit App UI starts here ---

# Title
st.title("Pneumonia Detection Dashboard")

# Sidebar setup
st.sidebar.header("Upload Chest X-ray Image")
st.sidebar.markdown(
    """
    **Instructions**  
    ➔ Upload a Chest X-ray image (jpg, png, jpeg).  
    ➔ Get prediction whether it's **Normal** or **Pneumonia**.  
    """
)

# File uploader widget
uploaded_file = st.file_uploader("Choose a Chest X-ray Image", type=["jpg", "jpeg", "png"])

# If the user uploads an image
if uploaded_file is not None:
    st.subheader("Uploaded Chest X-ray")
    st.image(uploaded_file, caption="Chest X-ray Image", use_container_width=True)

    # Preprocess and predict
    image = preprocess_image(uploaded_file)
    probabilities = model.predict(image)[0]
    class_labels = ["Normal", "Pneumonia"]
    prediction = class_labels[np.argmax(probabilities)]
    confidence = probabilities[np.argmax(probabilities)] * 100

    # Display result
    st.markdown("---")
    st.subheader("Prediction Results")
    st.success(f"**Prediction:** {prediction}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Visualize the results with a stylish bar chart
    st.markdown("---")
    st.subheader("Prediction Probability")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_labels, probabilities, color=['#3498db', '#e74c3c'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Model Prediction Probability')
    st.pyplot(fig)

# Footer
st.markdown("""
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style="text-align: center; font-size: 14px; color: gray;">
        Built by Sakshi | Powered by Deep Learning
    </div>
""", unsafe_allow_html=True)
