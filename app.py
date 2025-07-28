from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)
model = YOLO("yolov8_model.pt")  # 載入你的模型
UPLOAD_FOLDER = "uploads"
OUTPUT_PATH = "static/output.jpg"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result_image=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "沒有選擇圖片"

    file = request.files["image"]
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # 執行 YOLO 預測
    results = model(img_path)
    results[0].save(filename=OUTPUT_PATH)

    return render_template("index.html", result_image=OUTPUT_PATH)

if __name__ == "__main__":
    app.run(debug=True)
