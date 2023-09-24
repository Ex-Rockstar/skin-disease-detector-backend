from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from flask_cors import CORS 


app = Flask(__name__)
CORS(app)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']
        if image_file:
            # Read and preprocess the image
            image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1

            # Predict using the model
            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Return prediction result
            result = {
                "class_name": class_name[2:],
                "confidence_score": float(np.round(confidence_score * 100))
            }
            return jsonify(result)
        else:
            return jsonify({"error": "Image not found in request"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
