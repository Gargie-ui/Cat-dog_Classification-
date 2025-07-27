from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
import tensorflow as tf
import random

# Fix randomness for reproducibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

app = Flask(__name__)

# Load both models
model1 = load_model('model/cat_dog_neither_classifier1.keras')  # Better at cats/dogs
model2 = load_model('model/cat_dog_neither_classifier2.keras')  # Better at "neither"

class_names = ['cat', 'dog', 'neither']
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    # Preprocess image
    processed = preprocess_image(img_path)

    # Get predictions from both models
    pred1 = model1.predict(processed)[0]
    pred2 = model2.predict(processed)[0]

    # Apply same weights as predict.py
    weights_model1 = np.array([0.6, 0.6, 0.2])  # cat, dog, neither
    weights_model2 = np.array([0.4, 0.4, 0.8])

    combined_pred = (pred1 * weights_model1) + (pred2 * weights_model2)
    combined_pred /= np.sum(combined_pred)  # normalize

    class_index = int(np.argmax(combined_pred))
    confidence = round(float(np.max(combined_pred)) * 100, 2)
    final_class = class_names[class_index]

    return render_template(
        'result.html',
        prediction=final_class,
        confidence=confidence,
        img_path='/' + img_path
    )

if __name__ == '__main__':
    app.run(debug=True)
