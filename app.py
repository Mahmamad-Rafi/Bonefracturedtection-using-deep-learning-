from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the model path (supports both `.keras` and `.h5`)
MODEL_PATH = 'Model/unet_model_final.h5'

# Try loading the model
model = None
try:
    model = load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")

# Create an upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed image file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Function to check if uploaded file is valid
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocessing function for images
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(256, 256), color_mode="grayscale")
        x = image.img_to_array(img) / 255.0  # Normalize to [0,1]
        x = np.expand_dims(x, axis=-1)  # Ensure shape is (H, W, 1)
        x = np.expand_dims(x, axis=0)   # Add batch dimension (1, H, W, 1)
        
        logger.info(f"✅ Processed Image Shape: {x.shape}, Min: {x.min()}, Max: {x.max()}")
        return x
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")
        return None

# Prediction function
def model_predict(img_path):
    if model is None:
        return "Model not loaded", "index.html"

    x = preprocess_image(img_path)
    if x is None:
        return "Error processing image", "index.html"

    preds = model.predict(x)
    logger.info(f"📊 Raw Model Output: {preds}")

    # Classification Model Case (Binary Classification)
    if preds.shape == (1, 1):
        prob = preds[0, 0]
        logger.info(f"🦴 Predicted Probability: {prob}")
        return ("The bone is fractured", "fractured.html") if prob > 0.5 else ("The bone is not fractured", "not_fractured.html")

    # Segmentation Model Case
    if preds.shape[1:] in [(256, 256, 1), (256, 256)]:
        fracture_percentage = np.mean(preds > 0.5) * 100
        logger.info(f"🦴 Fracture Pixel Percentage: {fracture_percentage}%")
        return ("The bone is fractured", "fractured.html") if fracture_percentage > 10 else ("The bone is not fractured", "not_fractured.html")

    # Multi-class classification
    if preds.shape[1] > 1:
        pred_class = np.argmax(preds, axis=1)[0]
        logger.info(f"🦴 Predicted Class: {pred_class}")
        return ("The bone is fractured", "fractured.html") if pred_class == 0 else ("The bone is not fractured", "not_fractured.html")

    return "Prediction error", "index.html"

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            logger.error("❌ No file part in request")
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            logger.error("❌ No selected file")
            return render_template("index.html", error="No file selected")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Run prediction
            pred, output_page = model_predict(file_path)

            # Send user-uploaded image to result page
            user_image_url = url_for("static", filename=f"uploads/{filename}")
            return render_template(output_page, pred_output=pred, user_image=user_image_url)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5250, debug=True)


