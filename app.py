import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict_image import predict_image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Check if the file part is in the request
        if 'file' not in request.files:
            return render_template('index.html', result="No file selected")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', result="No file selected")

        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Make a predictionpip
            result = predict_image(file_path)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
    

