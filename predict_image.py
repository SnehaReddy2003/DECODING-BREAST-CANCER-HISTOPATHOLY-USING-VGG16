from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('breast_cancer_vgg16_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction > 0.5:
        return 'Malignant'
    else:
        return 'Benign'

# Usage
img_path= r'C:\\Users\\SNEHA\\OneDrive\\Desktop\\Project1\\breast\\benign\\SOB\\adenosis\\SOB_B_A_14-29960CD\200X\\SOB_B_A-14-29960CD-200-004.png'  # Replace with actual image path
result = predict_image(img_path)
print(f'The image is: {result}')
