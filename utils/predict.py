import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load both models
model_catdog = load_model('model/cat_dog_neither_classifier1.keras')  # Good at cats/dogs
model_neither = load_model('model/cat_dog_neither_classifier2.keras')  # Good at neither

# Class names (order must match training)
class_names = ['cat', 'dog', 'neither']

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    processed_img = preprocess_image(image_path)

    # Get raw predictions
    pred1 = model_catdog.predict(processed_img)[0]
    pred2 = model_neither.predict(processed_img)[0]

    # Apply weights to each model’s prediction per class
    weights_model1 = np.array([0.6, 0.6, 0.2])  # higher for cat/dog
    weights_model2 = np.array([0.4, 0.4, 0.8])  # higher for neither

    # Weighted combination
    combined_pred = (pred1 * weights_model1) + (pred2 * weights_model2)

    # Normalize to make it a valid probability distribution (optional but safer)
    combined_pred /= np.sum(combined_pred)

    class_index = np.argmax(combined_pred)
    confidence = float(np.max(combined_pred))

    return class_names[class_index], round(confidence * 100, 2)

if __name__ == "__main__":
    image_path = "ham.webp"  # Replace with your image path
    label, confidence = predict_image(image_path)
    print(f"Prediction: {label} ({confidence}%)")
