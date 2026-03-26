from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model.h5")

class_names = ['Ambulance', 'Bike', 'Bus', 'Car', 'Truck']

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    confidence = float(np.max(predictions))
    class_index = np.argmax(predictions)

    predicted_class = class_names[class_index]

    if confidence >= 0.85:
        decision = "High Confidence ✅"
    elif confidence >= 0.65:
        decision = "Needs Review ❓"
    else:
        decision = "Uncertain ⚠"

    return predicted_class, confidence, decision


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    prediction, confidence, decision = predict_image(file_path)

    return render_template("result.html",
                           image_path=file_path,
                           prediction=prediction,
                           confidence=round(confidence*100, 2),
                           decision=decision)


if __name__ == "__main__":
    app.run(debug=True)