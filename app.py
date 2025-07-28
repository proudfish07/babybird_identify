from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 載入訓練好的模型
model = YOLO('yolov8_model.pt')

@app.route('/')
def index():
    return render_template('index.html', result_image=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return '未選擇檔案'

    file = request.files['image']
    if file.filename == '':
        return '未選擇檔案名稱'

    # 儲存上傳的圖片
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # 執行辨識
    results = model(img_path)
    results[0].save(filename=os.path.join(OUTPUT_FOLDER, 'output.jpg'))

    return render_template('index.html', result_image='output.jpg')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
