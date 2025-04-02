from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO  # Thư viện YOLOv8

app = Flask(__name__)

# Tải các mô hình YOLO (kiểm tra đường dẫn file mô hình)
model1 = YOLO("trainYoLO.pt")
model2 = YOLO("trainYoLo2c.pt")

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

    # Trích xuất thông tin dự đoán (bounding boxes)
    output1 = results1[0].boxes.data.tolist()  # Dữ liệu bounding box từ model1
    output2 = results2[0].boxes.data.tolist()  # Dữ liệu bounding box từ model2

    # Logic xử lý kết quả nhận diện lỗi
    # Giả sử rằng bạn có một cách nào đó để xác định lỗi 1 hoặc lỗi 2
    # Ví dụ: nếu có bounding box nào có kích thước nhỏ (chỉ là ví dụ), bạn sẽ coi đó là lỗi 1 (TXNL)
    # Nếu có bounding box nào có tỷ lệ lớn, bạn coi đó là lỗi 2 (STP)

    error_message = None

    if output1 or output2:  # Nếu có ít nhất một lỗi được phát hiện
        for box in output1 + output2:
            x1, y1, x2, y2, confidence, class_id = box  # Tách bounding box và thông tin lớp
            # Giả sử rằng class_id hoặc kích thước bounding box sẽ giúp phân biệt lỗi 1 và lỗi 2
            if confidence > 0.5:  # Nếu confidence > 0.5 (chỉ là ví dụ)
                if x2 - x1 < 100:  # Nếu chiều rộng của box nhỏ (lỗi 1 - TXNL)
                    error_message = "TXNL"
                elif y2 - y1 > 300:  # Nếu chiều cao của box lớn (lỗi 2 - STP)
                    error_message = "STP"
                break  # Nếu đã tìm thấy lỗi, không cần kiểm tra thêm

    # Nếu không phát hiện lỗi, trả về thông báo "Không phát hiện lỗi"
    if error_message is None:
        error_message = "No error detected"

    return jsonify({"error_message": error_message})

if __name__ == '__main__':
    # Đảm bảo ứng dụng lắng nghe đúng cổng Render cấp
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
