from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

app = Flask(__name__)

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

    # 🔥 Chỉ load mô hình khi có request (giảm tiêu thụ RAM)
    model1 = YOLO("trainYoLO.pt")
    model2 = YOLO("trainYoLo2c.pt")

    # Chạy dự đoán trên cả hai model
    results1 = model1.predict(img)
    results2 = model2.predict(img)

    # Trích xuất thông tin dự đoán
    output1 = results1[0].boxes.data.tolist() if results1[0].boxes is not None else []
    output2 = results2[0].boxes.data.tolist() if results2[0].boxes is not None else []

    # Xử lý kết quả
    error_message = "No error detected"
    if output1 or output2:
        error_message = "TXNL" if output1 else "STP"

    return jsonify({"error_message": error_message})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
