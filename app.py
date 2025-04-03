from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

app = Flask(__name__)

# Chỉ load duy nhất 1 model
MODEL_PATH = "trainYoLo2c.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model {MODEL_PATH} không tồn tại!")

model = YOLO(MODEL_PATH)

@app.route('/')
def home():
    return "YOLO AI API is running on Render!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Thiếu ảnh"}), 400

        # Đọc ảnh từ request
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (640, 640))  # Resize để tiết kiệm tài nguyên

        # Dự đoán bằng YOLO
        with torch.no_grad():  # Tắt Autograd để tiết kiệm RAM
            results = model.predict(img)

        output = results[0].boxes.data.tolist() if results[0].boxes else []

        return jsonify({"detections": output})

    except Exception as e:
        return jsonify({"error": f"Lỗi trong server: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
