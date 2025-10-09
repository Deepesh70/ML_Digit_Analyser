import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load and Preprocess the MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a channels dimension (for CNN)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 2. Build the Convolutional Neural Network (CNN) Model
model = models.Sequential([
    # Input layer: 28x28 pixel images with 1 color channel (grayscale)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Output layer: 10 neurons for digits 0-9
    layers.Dense(10, activation='softmax')
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the Model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 5. Save the Model in Keras format
model.save('mnist_keras_model.h5')

print("\nModel trained and saved as mnist_keras_model.h5")