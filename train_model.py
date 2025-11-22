import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set dataset paths
train_dir = r"C:\Users\SNEHA\OneDrive\Desktop\Project1\breast"  # Change this to your actual dataset path
batch_size = 32
img_height, img_width = 224, 224

# Define ImageDataGenerators for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to between 0 and 1
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Set up the data generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(train_dir, 'train'),  # Subfolder for training images (adjust if needed)
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'  # Binary classification (benign vs malignant)
)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(train_dir, 'validation'),  # Subfolder for validation images (adjust if needed)
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Load the VGG16 model (excluding the top layer)
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of VGG16
for layer in vgg_base.layers:
    layer.trainable = False

# Build the custom model
model = models.Sequential()
model.add(vgg_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the model
model.save('breast_cancer_vgg16_model.keras')

# Plotting Accuracy and Loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()

# Confusion Matrix
predictions = model.predict(validation_generator)
y_pred = (predictions > 0.5).astype(int)
y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

