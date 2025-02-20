# Introduction
 I am  using a Flask-ML server and an ONNX model to detect Down Syndrome from facial images.

## Steps to run the project:
Step 1: Cloning the Repository:
```bash
git clone https://github.com/priyankaba15/596_individual_proj.git
cd 596_individual_proj
```


Step 2: Creating a virtual environment:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

Step 3: To install Required packages:
```bash
pip install -r requirements.txt
```

## I have the following structure for the project:

- server_onnx.py: A Flask-ML server that hosts the ONNX model for predicting Down Syndrome from the test dataset.
- app_onnx.py: Handles pre-processing and post-processing, providing a command-line interface to run the ONNX model.
- syndrome_detection_model.onnx: A Keras model converted into ONNX format for Down Syndrome prediction.
- test: Contains the test dataset used to evaluate the model’s accuracy.

## How to Run the Model:
Run the following command to start the server:
```bash
 python server_onnx.py
 OR python3 server_onnx.py
```

Once launched, you can register the model on the rescue box after assigning an IP address and port.

## Using the CLI:
In app_onnx.py, update the values of folder_path and model_path to match the locations where you saved them on your system.

After modifying the paths, run the following command:
```bash
python3 app_onnx.py
OR python app_onnx.py
```

The code processes the test images and determines whether each image indicates the presence of Down Syndrome or not.

## Keras to ONNX Conversion:
To export keras to onnx we use the tf2onnx.

## Code used to convert keras model to onnx:

```bash
import tf2onnx

spec = (tf.TensorSpec((None, 250, 250, 3), tf.float32, name="input"),)  
onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13)
with open(onnx_model_path, "wb") as f:
f.write(onnx_model.SerializeToString())
```


