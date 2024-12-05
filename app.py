from flask import Flask, request, jsonify
import subprocess
import requests
from PIL import Image
from io import BytesIO
import json
import predict_image
import httpx
import base64
from flask_cors import CORS
from ultralytics import YOLO
import cv2 

app = Flask(__name__)
CORS(app) 

@app.route('/')
def home():
    return "Backend Flask Snailly is running!"

# Route untuk menerima data via POST
@app.route('/link-history', methods=['POST'])
def submit_link():
    data = request.get_json()  # Mengambil data JSON dari body
    url_link = data.get('url')  # Mengambil nilai 'url' dari JSON
    log_id = data.get('log_id')
    print(url_link)
    print(log_id)

    try:
        subprocess.run(["python3", "real_time.py", url_link, log_id], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'Failed to run other script', 'details': str(e)}), 500

    return jsonify({'message': 'URL received and script executed', 'url': url_link}), 200

# Route untuk menerima data via POST
@app.route('/predict-image', methods=['POST'])
def predict():
    try:
        # Mendapatkan URL gambar dari request
        data = request.json
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400


        img = None

        # Check if the image_url is a base64-encoded image
        if image_url.startswith("data:image/"):
            # Extract base64 data from the URL
            try:
                base64_str = image_url.split(",")[1]  # Extract base64 part after the comma
                img_data = base64.b64decode(base64_str)
                img = Image.open(BytesIO(img_data))
            except Exception as e:
                return jsonify({"error": "Failed to decode base64 image: " + str(e)}), 400

        else:
            # If it's a regular URL, download the image
            try:
                with httpx.Client() as client:
                    response = client.get(image_url)
                    img = Image.open(BytesIO(response.content))
            except Exception as e:
                return jsonify({"error": "Failed to download image: " + str(e)}), 400

        try:
            # Simpan gambar sementara di server
            img_path = "temp_image.jpg"
            img.save(img_path)
            print(img_path)
            prediction = predict_image.predict_image(img_path)

            print(json.dumps({"prediction": prediction}))
        #     prediction_output = json.loads(result.stdout)  # Parsing output JSON dari subprocess
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse prediction result"}), 500
        
        # Kembalikan hasil prediksi
        return jsonify({"hasil": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True, port=4638)
