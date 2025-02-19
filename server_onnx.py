import argparse
import csv
import warnings
from pathlib import Path
from typing import TypedDict
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    FileResponse,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
)
import onnxruntime as ort
import cv2
import numpy as np
import os

warnings.filterwarnings("ignore")

disease_classes = {0: "Healthy", 1: "Down"}

def create_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the directory containing images",
        input_type=InputType.DIRECTORY,
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output directory",
        input_type=InputType.DIRECTORY,
    )
    return TaskSchema(inputs=[input_schema, output_schema], parameters=[])

class Inputs(TypedDict):
    input_dataset: DirectoryInput
    output_file: DirectoryInput

class Parameters(TypedDict):
    pass

server = MLServer(__name__)

server.add_app_metadata(
    name="Syndrome Detection Model",
    author="Priyanka",
    version="1.0.0",
    info="Syndrome detection using ONNX model."
)

onnx_model_path = "syndrome_detection_model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (250, 250))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@server.route("/predict", task_schema_func=create_task_schema)
def predict(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    input_path = inputs["input_dataset"].path
    out = Path(inputs["output_file"].path) / f"predictions_{np.random.randint(1000)}.csv"
    
    results = []
    for filename in os.listdir(input_path):
        image_path = os.path.join(input_path, filename)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            img = preprocess_image(image_path)
            prediction = session.run([output_name], {input_name: img})[0]
            probability = prediction[0][0]
            class_prediction = disease_classes[round(probability)]
            results.append({"image_path": filename, "prediction": class_prediction, "confidence": probability})
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    with open(out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(results)
    
    return ResponseBody(FileResponse(path=str(out), file_type="csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the syndrome detection server.")
    parser.add_argument("--port", type=int, help="Port number to run the server", default=5000)
    args = parser.parse_args()
    server.run(port=args.port)
