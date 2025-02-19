import cv2
import numpy as np
import onnxruntime as ort
import os

disease_classes = {0: "Healthy", 1: "Down"}

def preprocess_image(image_path):
    """
    Load and preprocess an image for ONNX model inference.
    """
    img = cv2.imread(image_path)  
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (250, 250))  
    img = img.astype(np.float32) / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

def load_onnx_model(model_path):
    """
    Load the ONNX model.
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session

def predict_images(folder_path, model_path):
    """
    Perform inference on all images in a folder using an ONNX model.
    """
    session = load_onnx_model(model_path)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  

        try:
            img = preprocess_image(image_path)

            prediction = session.run([output_name], {input_name: img})[0]
            probability = prediction[0][0]  
            class_prediction = disease_classes[round(probability)] 

            results.append((filename, probability, class_prediction))

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    for filename, prob, label in results:
        print(f"Image: {filename} → Probability: {prob:.4f} → Prediction: {label}")

folder_path = "/Users/priyankaba/Project1_596E/data/dataset/test"  
model_path = "/Users/priyankaba/Project1_596E/syndrome_detection_model.onnx"  

predict_images(folder_path, model_path)
