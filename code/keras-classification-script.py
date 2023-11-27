#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Install Tensorflow and Matplotlib
get_ipython().run_line_magic('pip', 'install tensorflow')
get_ipython().run_line_magic('pip', 'install matplotlib')


# In[2]:


# Imports
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


# In[3]:


# Data Paths
train_data_dir = '../data/pepsico_dataset/Train/'
test_data_dir = '../data/pepsico_dataset/Test/'


# In[4]:


# Set batch size and image size
batch_size = 32
image_size = (250, 250)


# In[5]:


# Preprocessing images by rescaling
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)


# In[6]:


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


# In[7]:


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


# In[8]:


# Model checkpoint
checkpoint_filepath = 'model_checkpoint.h5'  # Change the filename as needed
model_checkpoint = ModelCheckpoint(
    checkpoint_filepath,
    monitor='val_accuracy',  # You can change this to 'val_loss' or another metric
    save_best_only=True,
    mode='max',  # Use 'min' for loss, 'max' for accuracy
    verbose=1
)


# In[9]:


# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=5,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[model_checkpoint]
)


# In[10]:


# Load best model results
model.load_weights(checkpoint_filepath)


# In[11]:


# Results
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Accuracy: {test_accuracy}")


# In[12]:


# Printing keys to use for plotting
print(history.history.keys())


# In[13]:


# Plotting training and validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)

# Check if accuracy is in the history
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Training Accuracy')

# Check if val_accuracy is in the history
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()

