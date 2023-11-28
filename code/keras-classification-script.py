# Install Tensorflow and Matplotlib
# import subprocess
# subprocess.run(['pip', 'install', 'tensorflow'])
# subprocess.run(['pip', 'install', 'matplotlib'])

# Imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import sys
import io

# Data Paths
train_data_dir = r'C:/Users/deepk/OneDrive/Documents/College/6th Year/CSCE 580/CSCE580-Fall2023-DeepPatel-Repo/data/pepsico_dataset/Train'
test_data_dir = r'C:/Users/deepk/OneDrive/Documents/College/6th Year/CSCE 580/CSCE580-Fall2023-DeepPatel-Repo/data/pepsico_dataset/Test'


# Set batch size and image size
batch_size = 32
image_size = (250, 250)

# Preprocessing images by rescaling
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Save the original stdout
original_stdout = sys.stdout

# Redirect stdout to a dummy stream to suppress output
sys.stdout = io.StringIO()

# Creating train/test generators for binary classification
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
)

# Restore the original stdout
sys.stdout = original_stdout


# Neural network architecture for the sequential model
model = Sequential() # Create sequential model
model.add(Conv2D(32, (3, 3), input_shape=(250, 250, 3), activation='relu')) # CNN Input layer // 32 filters
model.add(MaxPooling2D((2, 2))) # Downsizing sample
model.add(Conv2D(64, (3, 3), activation='relu')) # 64 Filters
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu')) # 128 Filters
model.add(MaxPooling2D((2, 2)))
model.add(Flatten()) # Converting from 2D to 1D vector
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # CNN Output layer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model checkpoint filepath
checkpoint_filepath = r'C:/Users/deepk/OneDrive/Documents/College/6th Year/CSCE 580/CSCE580-Fall2023-DeepPatel-Repo/code/model_checkpoint.h5'

# Model checkpoint
model_checkpoint = ModelCheckpoint(
    checkpoint_filepath,
    monitor='val_accuracy',  
    save_best_only=True,
    mode='max', 
    verbose=1
)

# Load best model results
model.load_weights(checkpoint_filepath)

# Results
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Loss: {test_loss}")

# Function to classify an image
def classify_image(image_path):
    img = load_img(image_path, target_size=(250, 250))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    return "Pass" if prediction > 0.5 else "Fail"

if __name__ == "__main__":
    # Get the image path from command line arguments
    if len(sys.argv) != 2:
        print("Usage: python keras-classification-script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Perform image classification
    result = classify_image(image_path)

    # Print the result
    print(result)

