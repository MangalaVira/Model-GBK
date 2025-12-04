from flask import Flask, request, jsonify  # Untuk membuat API
from flask_cors import CORS  # Untuk mengaktifkan CORS
from tensorflow.keras.models import load_model  # Untuk memuat model
from tensorflow.keras.utils import img_to_array  # Untuk konversi gambar ke array
from tensorflow.keras.layers import Concatenate  # Untuk menangani merge model
from PIL import Image  # Untuk memproses gambar
import numpy as np  # Untuk manipulasi array
from io import BytesIO  # Untuk membaca file gambar sebagai stream

app = Flask(__name__)  # Inisialisasi Flask
CORS(app)  # Aktifkan CORS

# Load model
try:
    model = load_model(
        "model_guntingbatukertas.keras",
        custom_objects={"Concatenate": Concatenate}
    )
except Exception as e:
    raise ValueError(f"Error loading model: {str(e)}")

LABELS = ['paper', 'rock', 'scissors']

@app.route('/')
def welcome():
    return jsonify({"message": "Selamat Datang di API Model Gunting Batu Kertas"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        image = Image.open(BytesIO(file.read()))
        image = image.resize((160, 160))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0

        prediction = model.predict(image)
        predicted_class = LABELS[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
