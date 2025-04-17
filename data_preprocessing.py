from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
import tensorflow as tf

# Set parameters
img_size = (224, 224)  # VGG16 requires 224x224 images
batch_size = 32

# Image Data Generator for training and validation with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r"C:\Users\SNEHA\OneDrive\Desktop\Project1\breast",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    r"C:\Users\SNEHA\OneDrive\Desktop\Project1\breast"
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
