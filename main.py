import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input,
    RandomFlip, RandomRotation, RandomZoom, BatchNormalization
)
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Constants
img_size = 128
batch_size = 32
epochs = 20

# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)

# Load and prepare datasets
train_dataset = image_dataset_from_directory(
    'dataset/training_set',
    labels='inferred',
    label_mode='categorical',
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    'dataset/test_set',
    labels='inferred',
    label_mode='categorical',
    image_size=(img_size, img_size),
    batch_size=batch_size
)

# Print class names (optional)
class_names = train_dataset.class_names
print("Class indices:", class_names)

# Normalize & prefetch
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

# Build the CNN model
model = Sequential([
    Input(shape=(img_size, img_size, 3)),
    RandomFlip("horizontal"),
    RandomRotation(0.15),
    RandomZoom(0.2),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Changed from 2 to 3 for three classes
])

model.summary()

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs,
    callbacks=[early_stop, lr_reduce]
)

print("✅ Training completed.")
print("Saving the model...")
model.save('cat_dog_neither_classifier.keras')  # Updated filename
