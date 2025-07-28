import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
from utils import preprocess_image, load_labels, predict_class

session = ort.InferenceSession("model/best.onnx")
labels = load_labels("data/labels.json")

def classify_image(image):
    input_tensor = preprocess_image(image)
    prediction = predict_class(session, input_tensor, labels)
    return prediction

demo = gr.Interface(fn=classify_image, inputs="image", outputs="label", title="雛鳥辨識器")
demo.launch()
