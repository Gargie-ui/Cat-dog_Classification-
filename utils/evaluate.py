from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf

# Load model
model = load_model('model/cat_dog_neither_classifier.keras')  # Updated filename

# Load test data
test_dataset = image_dataset_from_directory(
    'dataset/test_set',
    labels='inferred',
    label_mode='categorical',
    image_size=(128, 128),
    batch_size=32
)

# Normalize & prefetch
test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

# Evaluate
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")
