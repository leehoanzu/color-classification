# Deploy with ONNX Runtime

This document provides a step-by-step guide for deploying a machine learning model on a Raspberry Pi using ONNX Runtime.

## Prerequisites

* A model saved in ONNX format
* Installed Raspberry Pi OS (64-bit version recommended)
* Python 3.7 or above installed
* ONNX Runtime for ARM64

### Required Libraries

1. **Python 3 and pip:**
    * Ensure Python 3 and pip are installed:
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```
2. **ONNX Runtime Installation:**
    * For this tutorial, you will need to install ONNX and ONNX Runtime. You can get binary builds of ONNX and ONNX Runtime with"
    ```bash
    pip install onnx onnxruntime
    ```
3. **Additional Dependencies:**
    * We recommends using the latest stable runtime for PyTorch and additional libraries such as NumPy or openCV:
    ```bash
    pip3 install numpy
    pip3 install opencv-python
    ```

## Load ONNX model

To streamline reusability and simplify future updates, it’s recommended to wrap the model loading functionality into a class. This structure makes it easy to reload models, change paths, or adjust configurations without needing to modify code across multiple files.

### Code Breakdown:

* Class Definition: The OnnxUtils class encapsulates both the loading of the ONNX model and the corresponding label file, making it easy to reuse this functionality in different scripts.

* Method to Load Labels: A separate method loadLabels() is introduced to handle label file loading, ensuring clean code separation and allowing for easy label management.

* Model Inference Session: Inside the class constructor, the ONNX model is loaded into an onnxruntime.InferenceSession, allowing for efficient inference calls later on.

* PyTorch Quantization Engine: If you’re using a quantized model, setting the backend to 'qnnpack' optimizes the inference process on resource-constrained devices like Raspberry Pi.

```python
import numpy as np
import onnxruntime

class OnnxUtils:
    def __init__(self,  onnx_model_path, labelPath):
        self.labels = self.loadLabels(labelPath)
        self.ortSession = onnxruntime.InferenceSession(onnx_model_path, providers = onnxruntime.get_available_providers()) 
        torch.backends.quantized.engine = 'qnnpack'

if __name__ == '__main__':

    def getAbsPath(path):
        # Get absolutely path
        currentDir = os.path.dirname(os.path.abspath(__file__))

        return os.path.abspath(os.path.join(currentDir, path))
    
    # Path of pretrained model
    onnx_model_path = getAbsPath("../model/color-best-edc.onnx")

    # Path of label
    pathLabels = getAbsPath("../label/color-label.txt")

    # Create intance of onnx-runtime
    ortSession = OnnxUtils(onnx_model_path, pathLabels)
```

## Infer

We load and preprocess the input data prior to inference to ensure optimal results.

```python
# Get input from camera
img = cap.read()  

# You can get input data from your device
# pathImage = getAbsPath(f"../img/val/1/white_{i}.png")
# image = cv2.imread(pathImage)  

# Preprocess the image before inference
preImage = ortSession.preProcessImage(image)

# Perform inference on the preprocessed image
preds = ortSession.infer(preImage)

# Display the inference results
ortSession.display(preds)
```

## Run the Script

To execute the script, simply run:

```bash
$ python3 ./utils/main.py
```

> [!NOTE]  
> <sup>1. Replace color-best-edc.onnx with the path to your ONNX model.</sup><br>
> <sup>2. Adjust the input data format based on your model's requirements.</sup>