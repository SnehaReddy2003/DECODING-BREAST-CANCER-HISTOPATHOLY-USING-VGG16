from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = 'breast_cancer_vgg16_model.keras'
model = load_model(MODEL_PATH)

def predict_image(img_path):
    try:
        # Preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        prediction = model.predict(img_array)
        confidence = prediction[0][0] * 100
        if prediction > 0.5:
            return f"Malignant "
        else:
            return f"Benign "
    except Exception as e:
        return f"Error during prediction: {str(e)}"

