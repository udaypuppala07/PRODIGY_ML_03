import os
import cv2
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam

# Step 1: Load and Preprocess Data
batch_size = 32
img_height, img_width = 224, 224

train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'U://prodigy//svm//test_set',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'U://prodigy//svm//test_set',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Step 2: Build a Simple CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
epochs = 10
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Step 4: Predict from a User-Provided Image
def predict_image(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_width, img_height))  # Resize image
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)
    return 'Dog' if prediction > 0.5 else 'Cat'

# Prompt the user for an image file path
image_path = input("Please enter the path to the image you want to classify: ")

# Make a prediction
label = predict_image(model, image_path)

# Display the result
image = cv2.imread(image_path)
cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('Dog vs Cat Classifier', image)
cv2.waitKey(0)  # Wait for key press to display the image
cv2.destroyAllWindows()

print("Prediction complete!")
