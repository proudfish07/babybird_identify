import numpy as np
from PIL import Image
import json

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # 依你的模型調整
    image = np.array(image).astype("float32") / 255.0
    image = np.transpose(image, (2, 0, 1))  # CHW 格式
    image = np.expand_dims(image, axis=0)
    return image

def load_labels(label_path):
    with open(label_path, "r") as f:
        return json.load(f)

def predict_class(session, input_tensor, labels):
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    predicted_idx = int(np.argmax(outputs[0]))
    return labels[str(predicted_idx)]
