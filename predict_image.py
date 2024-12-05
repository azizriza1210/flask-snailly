# Library Image
from PIL import Image
import numpy as np
import sys
from io import BytesIO
import requests
import tempfile
import json  
from ultralytics import YOLO
import os

model = YOLO("yolo_model.pt")

def save_image(file):
    filename = file.filename
    filepath = os.path.join("/", filename)
    file.save(filepath)
    return filepath

def predict_image_yolo(filepath):
    results = model(filepath)
    return results

def get_label(results):
    top_result = results[0].probs.top1  # Mendapatkan indeks label dengan probabilitas tertinggi
    names = results[0].names
    class_name = names.get(top_result, 'Unknown')  # 'Unknown' jika key tidak ada
    return class_name


def predict_image(image_path):
    print("IMI IMAGE PATH : ",image_path)
    results = predict_image_yolo(image_path)
    print("INI RESULT : ",results)
    class_name = get_label(results)

    return class_name

def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        print(json.dumps({"error": str(e)}))  
        sys.exit(1)

if __name__ == "__main__":
    url_link = sys.argv[1]
    img = download_image(url_link)

    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_image_file:
        img.save(temp_image_file.name)

        prediction = predict_image(temp_image_file.name)

        prediction_list = prediction.tolist()

        print(json.dumps({"prediction": prediction_list}))