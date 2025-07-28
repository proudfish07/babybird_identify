from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import os
import uuid

app = Flask(__name__)

model = YOLO("best.pt")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            results = model.predict(file_path, save=True, project="static", name="predict", exist_ok=True)

            result_path = os.path.join("static", "predict", os.path.basename(file_path))
            return render_template("index.html", result_image=result_path)

    return render_template("index.html", result_image=None)
