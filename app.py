from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from PIL import Image
import uuid

app = Flask(__name__)
model = YOLO('yoloweight.pt')  # 換成你自己的模型檔名

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    if img_file:
        filename = f"{uuid.uuid4().hex}.jpg"
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        result_path = os.path.join(RESULT_FOLDER, filename)
        img_file.save(input_path)

        # YOLO 預測
        results = model.predict(source=input_path, save=True, save_txt=False, save_conf=True, project=RESULT_FOLDER, name='', exist_ok=True)

        return render_template('index.html', result_image=os.path.join('results', filename))

    return 'No image uploaded'

if __name__ == '__main__':
    app.run(debug=True)
