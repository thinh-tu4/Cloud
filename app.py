from flask import Flask, render_template, request, send_from_directory, send_file
import os, zipfile, shutil
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'dashboard', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'dashboard', 'Save_pic_upload')
ZIP_OUTPUT = os.path.join(BASE_DIR, 'dashboard', 'results.zip')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ✅ Chỉ dùng 1 model duy nhất
model = YOLO("model/Model_AI.pt")  # Đường dẫn đến model mới

@app.route("/")
def index():
    return render_template("index.html", images=[])

@app.route("/upload", methods=["POST"])
def upload_zip():
    file = request.files['file']
    filename = secure_filename(file.filename)
    zip_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(zip_path)

    extract_dir = os.path.join(UPLOAD_FOLDER, "extracted")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Xóa kết quả cũ
    if os.path.exists(RESULT_FOLDER):
        shutil.rmtree(RESULT_FOLDER)
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    image_results = []

    for img_name in os.listdir(extract_dir):
        img_path = os.path.join(extract_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        result = model.predict(img, imgsz=640, conf=0.25)[0]

        labels = [model.names[int(cls)] for cls in result.boxes.cls]

        # Xác định lỗi nào
        has_txnl = any("txnl" in label.lower() for label in labels)
        has_stp = any("stp" in label.lower() for label in labels)

        result_text = "No Error"
        if has_txnl and has_stp:
            result_text = "TXNL + STP"
        elif has_txnl:
            result_text = "TXNL"
        elif has_stp:
            result_text = "STP"

        img_plot = result.plot()
        cv2.putText(img_plot, result_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        safe_name = secure_filename(img_name)
        out_path = os.path.join(RESULT_FOLDER, safe_name)
        cv2.imwrite(out_path, img_plot)

        image_results.append(safe_name)

    # Tạo file ZIP chứa ảnh kết quả
    with zipfile.ZipFile(ZIP_OUTPUT, 'w') as zipf:
        for filename in image_results:
            filepath = os.path.join(RESULT_FOLDER, filename)
            zipf.write(filepath, arcname=filename)

    return render_template("index.html", images=image_results, zip_available=True)

@app.route('/results/<path:filename>')
def serve_result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/download_zip')
def download_zip():
    return send_file(ZIP_OUTPUT, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
