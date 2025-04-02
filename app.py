from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from ultralytics import YOLO  # Thư viện YOLOv8

app = Flask(__name__)

# Load hai model YOLO
model1 = YOLO("trainYoLo.pt")
model2 = YOLO("trainYoLo2.pt")

@app.route('/')
def home():
    return "YOLO AI API is running on Render!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Đọc ảnh từ request
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Chạy dự đoán trên cả hai model
    results1 = model1.predict(img)
    results2 = model2.predict(img)

    # Trích xuất thông tin dự đoán
    output1 = results1[0].boxes.data.tolist()  # Dữ liệu bounding box
    output2 = results2[0].boxes.data.tolist()

    return jsonify({"model1_result": output1, "model2_result": output2})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
